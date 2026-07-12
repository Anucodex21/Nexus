from typing import List, Dict, Any
import json
import time

class AgentMemory:
    """Memory system for agents with short-term and long-term storage."""

    def __init__(self):
        self.short_term = []  # Recent interactions
        self.long_term = {}   # Key-value storage
        self.episodic = []    # Episode memories
        self.max_short_term = 10

    def add_short_term(self, entry: Dict):
        """Add to short-term memory."""
        entry["timestamp"] = time.time()
        self.short_term.append(entry)

        if len(self.short_term) > self.max_short_term:
            # Move oldest to long-term
            oldest = self.short_term.pop(0)
            self._consolidate_to_long_term(oldest)

    def _consolidate_to_long_term(self, entry: Dict):
        """Consolidate short-term memory to long-term."""
        key = f"memory_{int(time.time())}"
        self.long_term[key] = entry

    def add_long_term(self, key: str, value: Any):
        """Add to long-term memory."""
        self.long_term[key] = {
            "value": value,
            "timestamp": time.time()
        }

    def get_long_term(self, key: str):
        """Retrieve from long-term memory."""
        entry = self.long_term.get(key)
        return entry["value"] if entry else None

    def add_episodic(self, episode: Dict):
        """Add an episodic memory."""
        episode["timestamp"] = time.time()
        self.episodic.append(episode)

    def recall_recent(self, n=5):
        """Recall recent short-term memories."""
        return self.short_term[-n:]

    def search_long_term(self, keyword: str):
        """Search long-term memory."""
        results = []
        for key, entry in self.long_term.items():
            value_str = json.dumps(entry["value"])
            if keyword.lower() in value_str.lower():
                results.append({"key": key, "value": entry["value"]})
        return results

    def clear_short_term(self):
        """Clear short-term memory."""
        self.short_term = []

    def get_summary(self):
        """Get memory summary."""
        return {
            "short_term_count": len(self.short_term),
            "long_term_count": len(self.long_term),
            "episodic_count": len(self.episodic)
        }
