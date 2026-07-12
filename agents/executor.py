from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

class TaskExecutor:
    """Execute tasks with monitoring and error handling."""

    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.results = {}
        self.errors = {}

    def execute_single(self, task_id: str, task_func, *args, **kwargs) -> Any:
        """Execute a single task."""
        try:
            print(f"Executing task: {task_id}")
            result = task_func(*args, **kwargs)
            self.results[task_id] = {
                "status": "success",
                "result": result
            }
            return result
        except Exception as e:
            self.errors[task_id] = str(e)
            self.results[task_id] = {
                "status": "failed",
                "error": str(e)
            }
            return None

    def execute_parallel(self, tasks: Dict[str, tuple]) -> Dict[str, Any]:
        """Execute multiple tasks in parallel."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for task_id, (func, args, kwargs) in tasks.items():
                future = executor.submit(self.execute_single, task_id, func, *args, **kwargs)
                futures[future] = task_id

            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    future.result()
                except Exception as e:
                    self.errors[task_id] = str(e)

        return self.results

    def execute_sequential(self, tasks: Dict[str, tuple]) -> Dict[str, Any]:
        """Execute tasks sequentially with dependencies."""
        results = {}

        for task_id, (func, args, kwargs) in tasks.items():
            result = self.execute_single(task_id, func, *args, **kwargs)
            results[task_id] = result

        return results

    def get_status(self):
        """Get execution status."""
        return {
            "completed": len(self.results),
            "errors": len(self.errors),
            "success_rate": len([r for r in self.results.values() if r["status"] == "success"]) / max(len(self.results), 1)
        }
