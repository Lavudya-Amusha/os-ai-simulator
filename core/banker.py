"""
banker.py
---------
Implementation of Dijkstra's Banker's Algorithm for deadlock avoidance.

Checks whether the current resource allocation state is safe,
and if so, returns the safe execution sequence.

Usage
-----
    from core.banker import BankersAlgorithm

    b = BankersAlgorithm(
        allocation = [[0,1,0], [2,0,0], [3,0,2]],
        max_demand = [[7,5,3], [3,2,2], [9,0,2]],
        available  = [3, 3, 2],
    )

    result = b.run()
    print(result.is_safe)           # True / False
    print(result.safe_sequence)     # [1, 3, 0, 2, ...]
    print(result.explanation)       # step-by-step reasoning string
"""

from dataclasses import dataclass, field


# ──────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────

@dataclass
class BankersResult:
    is_safe       : bool
    safe_sequence : list = field(default_factory=list)
    need_matrix   : list = field(default_factory=list)
    explanation   : str  = ""


# ──────────────────────────────────────────────────────────────
# Algorithm
# ──────────────────────────────────────────────────────────────

class BankersAlgorithm:
    """
    Parameters
    ----------
    allocation : 2-D list  shape (n_processes × n_resources)
                 Resources currently held by each process.
    max_demand : 2-D list  shape (n_processes × n_resources)
                 Maximum resources each process may ever request.
    available  : 1-D list  length n_resources
                 Resources currently free in the system.
    """

    def __init__(self, allocation, max_demand, available):
        self._validate_inputs(allocation, max_demand, available)

        self.allocation = allocation
        self.max_demand = max_demand
        self.available  = available

        self.n = len(allocation)             # number of processes
        self.m = len(available)              # number of resource types

    # ──────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _validate_inputs(allocation, max_demand, available):
        if not allocation or not max_demand or not available:
            raise ValueError("Inputs must not be empty.")

        n = len(allocation)
        m = len(available)

        if len(max_demand) != n:
            raise ValueError(
                f"allocation ({n} rows) and max_demand "
                f"({len(max_demand)} rows) must have the same number of processes."
            )

        for i, (alloc_row, max_row) in enumerate(zip(allocation, max_demand)):
            if len(alloc_row) != m:
                raise ValueError(
                    f"allocation[{i}] has {len(alloc_row)} resources; expected {m}."
                )
            if len(max_row) != m:
                raise ValueError(
                    f"max_demand[{i}] has {len(max_row)} resources; expected {m}."
                )
            for j in range(m):
                if alloc_row[j] < 0 or max_row[j] < 0:
                    raise ValueError("Resource values must be non-negative.")
                if alloc_row[j] > max_row[j]:
                    raise ValueError(
                        f"Process {i}: allocation[{j}]={alloc_row[j]} exceeds "
                        f"max_demand[{j}]={max_row[j]}."
                    )

        if any(v < 0 for v in available):
            raise ValueError("Available resources must be non-negative.")

    # ──────────────────────────────────────────────────────────
    # Core algorithm
    # ──────────────────────────────────────────────────────────

    def calculate_need(self):
        """
        Need matrix = Max − Allocation.
        Represents remaining resources each process may still request.
        """
        return [
            [self.max_demand[i][j] - self.allocation[i][j]
             for j in range(self.m)]
            for i in range(self.n)
        ]

    def run(self):
        """
        Execute the safety algorithm.

        Returns
        -------
        BankersResult
            .is_safe       : bool
            .safe_sequence : list of process indices in execution order
            .need_matrix   : computed need matrix
            .explanation   : human-readable step-by-step log
        """
        need   = self.calculate_need()
        work   = self.available.copy()
        finish = [False] * self.n

        safe_sequence = []
        steps         = []           # for human-readable explanation

        while len(safe_sequence) < self.n:
            progress = False

            for i in range(self.n):
                if finish[i]:
                    continue

                # Can this process's remaining need be satisfied?
                can_allocate = all(
                    need[i][j] <= work[j] for j in range(self.m)
                )

                if can_allocate:
                    # Simulate process completion: release its resources
                    work = [work[j] + self.allocation[i][j] for j in range(self.m)]
                    finish[i] = True
                    safe_sequence.append(i)
                    progress = True

                    steps.append(
                        f"P{i} allocated → work becomes {work}"
                    )

            if not progress:
                # No progress in a full pass → unsafe state
                explanation = (
                    "Unsafe state detected.\n"
                    "Steps completed:\n  " + "\n  ".join(steps) + "\n"
                    f"Processes unable to complete: "
                    f"{[i for i in range(self.n) if not finish[i]]}"
                )
                return BankersResult(
                    is_safe=False,
                    need_matrix=need,
                    explanation=explanation
                )

        explanation = (
            "Safe state confirmed.\n"
            "Safe sequence: " + " → ".join(f"P{i}" for i in safe_sequence) + "\n"
            "Steps:\n  " + "\n  ".join(steps)
        )

        return BankersResult(
            is_safe=True,
            safe_sequence=safe_sequence,
            need_matrix=need,
            explanation=explanation
        )

    # ──────────────────────────────────────────────────────────
    # Convenience wrappers (keep existing call-sites working)
    # ──────────────────────────────────────────────────────────

    def is_safe(self):
        """
        Thin wrapper — returns (bool, safe_sequence).
        Kept for backward compatibility with existing UI code.
        """
        result = self.run()
        return result.is_safe, result.safe_sequence