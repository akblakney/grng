
from abc import ABC
from typing import Any, Dict, List
import sys


class Validator(ABC):
    """Base class for validators that run a collection of independent checks.

    Subclasses implement one or more `check_*` methods, each taking
    `raw` (the source's native raw data) and `values` (the standardized
    `List[int]`). Each check performs some validation action — e.g.
    producing a plot, or computing statistics — and may return a result
    (or `None` if its purpose is purely a side effect like plotting).

    `run_all` discovers and runs every `check_*` method defined on the
    instance, returning a dict mapping method name to its result.
    """

    def run_all(self, raw: Any, values: List[int]) -> Dict[str, Any]:
        results = {}
        for name in dir(self):
            if name.startswith("check_"):
                method = getattr(self, name)
                if callable(method):
                    results[name] = method(raw, values)
        return results

    def print_results(self, results):
        print("===== VALIDATION RESULTS =====")
        for name, result in results.items():
            if result is not None:
                print(f"\n{name}:", file=sys.stderr)
                print(result, file=sys.stderr)
        print("==============================\n")
