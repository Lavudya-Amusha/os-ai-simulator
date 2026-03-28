"""
simulator.py
------------
Central orchestrator that ties together scheduling, deadlock
detection, and metrics computation.
"""

from __future__ import annotations
import copy
import logging

from core.scheduler    import fcfs, sjf, round_robin, mlfq, priority_scheduling
from core.rag_deadlock import build_rag_from_processes
from core.metrics      import calculate_metrics, get_per_process_stats

logger = logging.getLogger(__name__)

_SCHEDULERS = {
    "FCFS"    : fcfs,
    "SJF"     : sjf,
    "RR"      : round_robin,
    "MLFQ"    : mlfq,
    "Priority": priority_scheduling,
}

_REWARD_W_WAIT       = 0.6
_REWARD_W_TURN       = 0.3
_REWARD_W_THROUGHPUT = 0.1


class OSSimulator:
    """
    Gym-like interface for the RL agent.
    Also supports direct deterministic simulation runs.
    """

    def __init__(self, processes: list[dict]):
        if not processes:
            raise ValueError("Process list must not be empty.")
        self.processes      = processes
        self._rag_deadlock  = self._check_deadlock(processes)
        self._last_avg_wait = 0.0   # updated after every simulation run
        logger.info("OSSimulator initialised with %d processes | deadlock=%s",
                    len(processes), self._rag_deadlock)

    # ── Private helpers ────────────────────────────────────────

    @staticmethod
    def _check_deadlock(processes: list[dict]) -> bool:
        rag = build_rag_from_processes(processes)
        return rag.detect_cycle()

    @staticmethod
    def _run_scheduler(algo: str, processes: list[dict]) -> list:
        if algo not in _SCHEDULERS:
            raise ValueError(
                f"Unknown algorithm '{algo}'. "
                f"Choose from: {list(_SCHEDULERS.keys())}"
            )
        logger.debug("Running scheduler: %s", algo)
        return _SCHEDULERS[algo](copy.deepcopy(processes))

    def _compute_reward(
        self, avg_wait: float, avg_turn: float, throughput: float
    ) -> float:
        return (
            - (_REWARD_W_WAIT * avg_wait)
            - (_REWARD_W_TURN * avg_turn)
            + (_REWARD_W_THROUGHPUT * throughput)
        )

    # ── Public API ─────────────────────────────────────────────

    def get_state(self) -> tuple:
        """
        Snapshot of current system state for the RL agent.
        Returns (num_processes, avg_burst, avg_wait, deadlock_flag)

        avg_wait reflects the LAST simulation run so the RL agent
        sees real performance data, not a hardcoded 0.0.
        """
        n         = len(self.processes)
        avg_burst = sum(p['burst'] for p in self.processes) / n if n else 0
        return (n, round(avg_burst, 2),
                round(self._last_avg_wait, 2), int(self._rag_deadlock))

    def step(self, action: str) -> tuple:
        """
        One RL environment step.

        Returns
        -------
        next_state, reward, timeline, deadlock
        """
        timeline = self._run_scheduler(action, self.processes)
        avg_wait, avg_turn, _, throughput = calculate_metrics(
            self.processes, timeline
        )
        reward     = self._compute_reward(avg_wait, avg_turn, throughput)
        n          = len(self.processes)
        avg_burst  = sum(p['burst'] for p in self.processes) / n
        next_state = (n, round(avg_burst, 2),
                      round(avg_wait, 2), int(self._rag_deadlock))

        self._last_avg_wait = avg_wait   # keep state current
        logger.debug("step action=%s reward=%.3f", action, reward)
        return next_state, reward, timeline, self._rag_deadlock

    def run_with_algorithm(self, algo: str) -> tuple:
        """
        Deterministic single simulation run.

        Returns
        -------
        timeline, deadlock, avg_wait, avg_turn,
        cpu_util, throughput, per_process
        """
        timeline = self._run_scheduler(algo, self.processes)
        avg_wait, avg_turn, cpu_util, throughput = calculate_metrics(
            self.processes, timeline
        )
        per_process = get_per_process_stats(self.processes, timeline)
        self._last_avg_wait = avg_wait   # feed back into state
        logger.info("run_with_algorithm algo=%s avg_wait=%.2f avg_turn=%.2f",
                    algo, avg_wait, avg_turn)
        return (timeline, self._rag_deadlock,
                avg_wait, avg_turn, cpu_util, throughput, per_process)

    def compare_all_algorithms(self) -> dict:
        """
        Run all algorithms and return a comparison dict.
        { algo: {avg_wait, avg_turn, cpu_util, throughput} }
        """
        results = {}
        for algo in _SCHEDULERS:
            timeline = self._run_scheduler(algo, self.processes)
            aw, at, cu, tp = calculate_metrics(self.processes, timeline)
            results[algo] = {
                'avg_wait'  : round(aw, 3),
                'avg_turn'  : round(at, 3),
                'cpu_util'  : round(cu, 3),
                'throughput': round(tp, 3),
            }
        return results