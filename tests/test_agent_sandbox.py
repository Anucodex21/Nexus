import os

import pytest

import agents.coding_agent as ca
from agents.coding_agent import _safe_path, WORKSPACE


def test_normal_relative_path_resolves_inside_workspace():
    p = _safe_path("hello.py")
    assert p.startswith(os.path.normpath(WORKSPACE))


def test_nested_relative_path_resolves_inside_workspace():
    p = _safe_path("subdir/hello.py")
    assert p.startswith(os.path.normpath(WORKSPACE))


@pytest.mark.parametrize(
    "bad_path",
    [
        "../evil.py",
        "../../etc/passwd",
        "../../../root/.ssh/id_rsa",
        "/etc/passwd",
        "subdir/../../evil.py",
    ],
)
def test_path_traversal_is_blocked(bad_path):
    with pytest.raises(ValueError):
        _safe_path(bad_path)


def test_write_then_read_roundtrip(tmp_path, monkeypatch):
    """Uses CodingAgent.__new__ to skip __init__ (which requires a
    configured LLM provider) since these tool methods don't need one."""
    monkeypatch.setattr(ca, "WORKSPACE", str(tmp_path))
    agent = ca.CodingAgent.__new__(ca.CodingAgent)

    write_result = agent.write_file("greeting.txt", "hello world")
    assert "Wrote" in write_result
    assert (tmp_path / "greeting.txt").read_text() == "hello world"

    read_result = agent.read_file("greeting.txt")
    assert read_result == "hello world"


def test_read_missing_file_returns_error_not_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(ca, "WORKSPACE", str(tmp_path))
    agent = ca.CodingAgent.__new__(ca.CodingAgent)
    result = agent.read_file("does_not_exist.txt")
    assert result.startswith("ERROR:")


def test_list_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ca, "WORKSPACE", str(tmp_path))
    agent = ca.CodingAgent.__new__(ca.CodingAgent)
    agent.write_file("a.txt", "1")
    agent.write_file("b.txt", "2")
    listing = agent.list_dir(".")
    assert "a.txt" in listing and "b.txt" in listing


def test_run_python_captures_stdout(tmp_path, monkeypatch):
    monkeypatch.setattr(ca, "WORKSPACE", str(tmp_path))
    agent = ca.CodingAgent.__new__(ca.CodingAgent)
    result = agent.run_python("print('hello from sandbox')")
    assert "hello from sandbox" in result


def test_run_python_captures_stderr_on_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ca, "WORKSPACE", str(tmp_path))
    agent = ca.CodingAgent.__new__(ca.CodingAgent)
    result = agent.run_python("raise ValueError('boom')")
    assert "STDERR" in result and "boom" in result


def test_parse_step_extracts_thought_action_input():
    agent = ca.CodingAgent.__new__(ca.CodingAgent)
    text = (
        "Thought: I should list the directory first\n"
        "Action: list_dir\n"
        'Action Input: {"path": "."}'
    )
    thought, action, action_input = agent._parse_step(text)
    assert thought == "I should list the directory first"
    assert action == "list_dir"
    assert action_input == '{"path": "."}'
