"""
rl_agent.py
-----------
Tabular Q-learning agent for CPU scheduling algorithm selection.

State space  : discretised (num_processes, avg_burst, avg_wait, deadlock)
Action space : {FCFS, SJF, RR, MLFQ, Priority}
Algorithm    : Q-learning with ε-greedy exploration + epsilon decay
"""

from __future__ import annotations
import random
import json
import os
import logging

logger = logging.getLogger(__name__)


class RLSchedulerAgent:
    """
    Tabular Q-learning agent.

    Hyperparameters
    ---------------
    alpha         : learning rate          (0.0 – 1.0)
    gamma         : discount factor        (0.0 – 1.0)
    epsilon       : initial exploration rate
    epsilon_decay : multiplicative decay per episode
    epsilon_min   : exploration floor
    """

    ACTIONS = ["FCFS", "SJF", "RR", "MLFQ", "Priority"]

    def __init__(
        self,
        alpha: float         = 0.1,
        gamma: float         = 0.9,
        epsilon: float       = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float   = 0.05,
    ):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min

        self.q_table: dict        = {}
        self.action_counts: dict  = {a: 0 for a in self.ACTIONS}
        self.reward_history: list = []

    # ── State discretisation ───────────────────────────────────

    @staticmethod
    def _bin(value: float, thresholds: list[float]) -> int:
        for i, t in enumerate(thresholds):
            if value < t:
                return i
        return len(thresholds)

    def get_state_key(self, state: tuple) -> tuple:
        num_processes, avg_burst, avg_wait, deadlock = state
        return (
            self._bin(num_processes, [5, 15]),
            self._bin(avg_burst,     [5, 15]),
            self._bin(avg_wait,      [10, 30]),
            int(bool(deadlock)),
        )

    # ── Q-table helpers ────────────────────────────────────────

    def _init_state(self, key: tuple) -> None:
        if key not in self.q_table:
            self.q_table[key] = {a: 0.0 for a in self.ACTIONS}

    def get_q_values(self, state: tuple) -> dict:
        key = self.get_state_key(state)
        self._init_state(key)
        return dict(self.q_table[key])

    # ── Core RL ────────────────────────────────────────────────

    def choose_algorithm(self, state: tuple) -> str:
        """ε-greedy policy."""
        key = self.get_state_key(state)
        self._init_state(key)

        if random.random() < self.epsilon:
            action = random.choice(self.ACTIONS)
        else:
            action = max(self.q_table[key], key=self.q_table[key].get)

        self.action_counts[action] += 1
        return action

    def update_q_value(
        self,
        state: tuple,
        action: str,
        reward: float,
        next_state: tuple,
    ) -> None:
        """Bellman update: Q(s,a) ← Q(s,a) + α[r + γ·maxQ(s',·) − Q(s,a)]"""
        s  = self.get_state_key(state)
        ns = self.get_state_key(next_state)
        self._init_state(s)
        self._init_state(ns)

        old_q    = self.q_table[s][action]
        next_max = max(self.q_table[ns].values())
        self.q_table[s][action] = old_q + self.alpha * (
            reward + self.gamma * next_max - old_q
        )
        self.reward_history.append(reward)

    def decay_epsilon(self) -> None:
        """Reduce exploration rate — call once per episode."""
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    @property
    def best_action_overall(self) -> str | None:
        if not any(self.action_counts.values()):
            return None
        return max(self.action_counts, key=self.action_counts.get)

    # ── Persistence ────────────────────────────────────────────

    def save(self, path: str = "logs/q_table.json") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "q_table"      : {str(k): v for k, v in self.q_table.items()},
            "epsilon"      : self.epsilon,
            "action_counts": self.action_counts,
            "hyperparams"  : {
                "alpha"        : self.alpha,
                "gamma"        : self.gamma,
                "epsilon_decay": self.epsilon_decay,
                "epsilon_min"  : self.epsilon_min,
            },
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Q-table saved to %s", path)

    def load(self, path: str = "logs/q_table.json") -> None:
        if not os.path.exists(path):
            return
        with open(path, "r") as f:
            payload = json.load(f)
        self.q_table = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in payload.get("q_table", {}).items()
        }
        self.epsilon       = payload.get("epsilon", self.epsilon)
        self.action_counts = payload.get("action_counts", self.action_counts)
        hp = payload.get("hyperparams", {})
        self.alpha         = hp.get("alpha",         self.alpha)
        self.gamma         = hp.get("gamma",         self.gamma)
        self.epsilon_decay = hp.get("epsilon_decay", self.epsilon_decay)
        logger.info("Q-table loaded from %s", path)

    def reset(self) -> None:
        self.q_table        = {}
        self.action_counts  = {a: 0 for a in self.ACTIONS}
        self.reward_history = []
        self.epsilon        = 1.0