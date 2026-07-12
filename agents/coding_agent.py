"""
CodingAgent - a real ReAct-style AI agent that can read files, write files,
list directories, and run Python code to complete a coding task.

Loop: LLM thinks -> picks a tool -> tool runs -> result fed back to LLM -> repeat
until the LLM says it's done (Final Answer) or a step limit is hit.

Run it directly:
    python -m agents.coding_agent

Everything the agent touches is sandboxed inside ./agent_workspace/ so it
can't read/write/execute anywhere else on disk.
"""

import os
import re
import json
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.backend.llm_client import LLMClient  # noqa: E402
from agents.memory import AgentMemory  # noqa: E402

WORKSPACE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent_workspace")
MAX_STEPS = 8

SYSTEM_PROMPT = """You are a coding agent. You solve tasks by using tools, one step at a time.
You have access to these tools:

- read_file(path): read a text file
- write_file(path, content): create or overwrite a file
- list_dir(path): list files in a directory ("." for workspace root)
- run_python(code): execute a Python snippet and return stdout/stderr

Respond using EXACTLY this format each turn, nothing else:

Thought: <your reasoning about what to do next>
Action: <one of read_file, write_file, list_dir, run_python, finish>
Action Input: <JSON object with the arguments for the tool>

When the task is fully done, use:
Action: finish
Action Input: {"summary": "<what you did and the result>"}

Rules:
- All paths are relative to your workspace, e.g. "hello.py", not absolute paths.
- One action per turn. Wait for the Observation before continuing.
- Keep write_file content complete and correct - it fully replaces the file.
"""


def _safe_path(path: str) -> str:
    """Resolve a path inside WORKSPACE only, blocking path traversal."""
    os.makedirs(WORKSPACE, exist_ok=True)
    full = os.path.normpath(os.path.join(WORKSPACE, path))
    if not full.startswith(os.path.normpath(WORKSPACE)):
        raise ValueError("Path escapes the workspace sandbox - blocked.")
    return full


class CodingAgent:
    def __init__(self, preferred_model=None):
        self.llm = LLMClient()
        self.preferred_model = preferred_model
        self.memory = AgentMemory()
        if not self.llm.available_providers():
            raise RuntimeError(
                "No LLM provider configured. Set at least one API key "
                "(e.g. HF_TOKEN or NVIDIA_API_KEY) in your .env first."
            )

    # ---------------- Tools ----------------

    def read_file(self, path):
        try:
            with open(_safe_path(path), "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"ERROR: {e}"

    def write_file(self, path, content):
        try:
            full = _safe_path(path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Wrote {len(content)} chars to {path}"
        except Exception as e:
            return f"ERROR: {e}"

    def list_dir(self, path="."):
        try:
            full = _safe_path(path)
            return "\n".join(sorted(os.listdir(full))) or "(empty)"
        except Exception as e:
            return f"ERROR: {e}"

    def run_python(self, code):
        try:
            script_path = _safe_path("_agent_tmp_run.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=WORKSPACE,
                capture_output=True,
                text=True,
                timeout=15,
            )
            out = result.stdout.strip()
            err = result.stderr.strip()
            return f"STDOUT:\n{out}\nSTDERR:\n{err}" if err else f"STDOUT:\n{out}"
        except subprocess.TimeoutExpired:
            return "ERROR: execution timed out (15s limit)"
        except Exception as e:
            return f"ERROR: {e}"

    # ---------------- ReAct loop ----------------

    def _parse_step(self, text):
        thought = re.search(r"Thought:\s*(.*?)(?=\nAction:)", text, re.S)
        action = re.search(r"Action:\s*(\w+)", text)
        action_input = re.search(r"Action Input:\s*(\{.*\})", text, re.S)
        return (
            thought.group(1).strip() if thought else "",
            action.group(1).strip() if action else None,
            action_input.group(1).strip() if action_input else "{}",
        )

    def _memory_context(self, max_episodes: int = 3) -> str:
        """Summarize recent past episodes from this session's memory so the
        LLM has continuity across /agent/run calls that share a session_id,
        instead of starting from a blank slate every time."""
        recent = self.memory.episodic[-max_episodes:]
        if not recent:
            return ""
        lines = []
        for ep in recent:
            result = str(ep.get("result", ""))[:300]
            lines.append(f"- Task: {ep.get('task', '')}\n  Result: {result}")
        return "\n\nContext from earlier tasks in this session (for reference only, don't redo them unless asked):\n" + "\n".join(lines)

    def run_steps(self, task: str):
        """Generator version of run() - yields structured step dicts as the
        agent works, so a UI (CLI or web) can render them live instead of
        waiting for the final answer."""
        memory_context = self._memory_context()
        messages = [
            {"role": "user", "content": f"{SYSTEM_PROMPT}{memory_context}\n\nTask: {task}"}
        ]
        self.memory.add_short_term({"role": "task", "content": task})
        yield {"type": "task", "content": task}

        for step in range(1, MAX_STEPS + 1):
            reply, provider = self.llm.chat(messages, preferred=self.preferred_model)
            thought, action, action_input_raw = self._parse_step(reply)

            if thought:
                yield {"type": "thought", "content": thought, "step": step, "provider": provider}

            try:
                args = json.loads(action_input_raw)
            except json.JSONDecodeError:
                args = {}

            if action == "finish" or action is None:
                summary = args.get("summary", reply)
                self.memory.add_episodic({"task": task, "result": summary})
                yield {"type": "final", "content": summary, "step": step}
                return

            yield {"type": "action", "tool": action, "args": args, "step": step}

            tool_map = {
                "read_file": lambda a: self.read_file(a.get("path", "")),
                "write_file": lambda a: self.write_file(a.get("path", ""), a.get("content", "")),
                "list_dir": lambda a: self.list_dir(a.get("path", ".")),
                "run_python": lambda a: self.run_python(a.get("code", "")),
            }
            tool_fn = tool_map.get(action)
            observation = tool_fn(args) if tool_fn else f"ERROR: unknown tool '{action}'"
            yield {"type": "observation", "content": observation, "step": step}

            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        yield {"type": "final", "content": "Stopped: reached max steps without a Final Answer.", "step": MAX_STEPS}

    def run(self, task: str, verbose=True):
        """Convenience wrapper around run_steps() for simple CLI usage."""
        final_content = ""
        for event in self.run_steps(task):
            if verbose:
                if event["type"] == "thought":
                    print(f"\n--- Step {event['step']} (via {event['provider']}) ---")
                    print(f"Thought: {event['content']}")
                elif event["type"] == "action":
                    print(f"Action: {event['tool']}  Input: {event['args']}")
                elif event["type"] == "observation":
                    print(f"Observation: {event['content']}")
            if event["type"] == "final":
                final_content = event["content"]
        return final_content


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    agent = CodingAgent()
    print(f"Coding agent ready. Providers available: {agent.llm.available_providers()}")
    print(f"Workspace: {WORKSPACE}\n")

    task = input("What should the agent build/do? ").strip()
    result = agent.run(task)
    print("\n=== DONE ===")
    print(result)
