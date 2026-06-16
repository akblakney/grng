"""Base class for validators."""
import json
import sys
from abc import ABC
from typing import Any, Dict, List


class Validator(ABC):
    """Base class for validators that run a collection of independent checks.

    Subclasses implement one or more `check_*` methods, each taking
    `raw` (the source's native raw data) and `values` (the standardized
    `List[int]`). Each check performs some validation action

    `run_all` discovers and runs every `check_*` method defined on the
    instance, returning a dict mapping method name to its result.

    `accumulate` is called each batch to update running state.
    `finalize` is called once after all batches to report cumulative results.
    """

    def run_all(self, raw: Any, values: List[int]) -> Dict[str, Any]:
        results = {}
        for name in dir(self):
            if name.startswith("check_"):
                method = getattr(self, name)
                if callable(method):
                    results[name] = method(raw, values)
        return results

    def print_results(self, results: Dict[str, Any]) -> None:
        print("===== VALIDATION RESULTS =====", file=sys.stderr)
        for name, result in results.items():
            if result is not None:
                print(f"\n{name}:", file=sys.stderr)
                print(json.dumps(result, indent=2), file=sys.stderr)
        print("==============================\n", file=sys.stderr)

    def accumulate(self, raw: Any, values: List[int]) -> None:
        """Accumulate state across batches. Override in subclasses."""
        pass

    def finalize(self) -> None:
        """Compute and print cumulative results. Called once after all batches."""
        pass