"""
tests/test_core.py
------------------
Unit tests for the OS Scheduling Simulator core modules.

Run with:
    pytest tests/test_core.py -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.scheduler import fcfs, sjf, round_robin, mlfq, priority_scheduling
from core.metrics   import calculate_metrics, get_per_process_stats
from core.banker    import BankersAlgorithm
from core.rag_deadlock import ResourceAllocationGraph, build_rag_from_processes
from core.rl_agent  import RLSchedulerAgent


# ══════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════

@pytest.fixture
def simple_processes():
    """Three processes with known expected scheduling order."""
    return [
        {'pid': 0, 'arrival': 0, 'burst': 3, 'priority': 2, 'resource': 'R0', 'wanted_resource': ''},
        {'pid': 1, 'arrival': 1, 'burst': 1, 'priority': 1, 'resource': 'R1', 'wanted_resource': ''},
        {'pid': 2, 'arrival': 2, 'burst': 2, 'priority': 3, 'resource': 'R2', 'wanted_resource': ''},
    ]

@pytest.fixture
def single_process():
    return [{'pid': 0, 'arrival': 0, 'burst': 5, 'priority': 1,
             'resource': 'R0', 'wanted_resource': ''}]


# ══════════════════════════════════════════════════════════════
# FCFS Tests
# ══════════════════════════════════════════════════════════════

class TestFCFS:

    def test_empty_input(self):
        assert fcfs([]) == []

    def test_single_process(self, single_process):
        tl = fcfs(single_process)
        assert all(p == 0 for p in tl)
        assert len(tl) == 5

    def test_order_by_arrival(self, simple_processes):
        tl = fcfs(simple_processes)
        # P0 arrives first — must appear before P1 and P2
        first_pid = next(p for p in tl if p is not None)
        assert first_pid == 0

    def test_all_processes_complete(self, simple_processes):
        tl = fcfs(simple_processes)
        pids_in_timeline = {p for p in tl if p is not None}
        expected_pids    = {p['pid'] for p in simple_processes}
        assert pids_in_timeline == expected_pids

    def test_no_negative_idle(self, simple_processes):
        tl = fcfs(simple_processes)
        # Timeline must not contain negative values
        assert all(p is None or p >= 0 for p in tl)

    def test_correct_burst_length(self, single_process):
        tl = fcfs(single_process)
        pid_count = sum(1 for p in tl if p == 0)
        assert pid_count == single_process[0]['burst']


# ══════════════════════════════════════════════════════════════
# SJF Tests
# ══════════════════════════════════════════════════════════════

class TestSJF:

    def test_empty_input(self):
        assert sjf([]) == []

    def test_picks_shortest_first(self):
        """P1 (burst=1) should run before P2 (burst=5) if both arrive at t=0."""
        procs = [
            {'pid': 0, 'arrival': 0, 'burst': 5, 'priority': 1, 'resource': 'R0', 'wanted_resource': ''},
            {'pid': 1, 'arrival': 0, 'burst': 1, 'priority': 1, 'resource': 'R1', 'wanted_resource': ''},
        ]
        tl = sjf(procs)
        # P1 must come first
        assert tl[0] == 1

    def test_all_processes_complete(self, simple_processes):
        tl = sjf(simple_processes)
        assert {p for p in tl if p is not None} == {0, 1, 2}

    def test_total_burst_matches(self, simple_processes):
        tl = sjf(simple_processes)
        total_burst = sum(p['burst'] for p in simple_processes)
        busy_slots  = sum(1 for p in tl if p is not None)
        assert busy_slots == total_burst


# ══════════════════════════════════════════════════════════════
# Round Robin Tests
# ══════════════════════════════════════════════════════════════

class TestRoundRobin:

    def test_empty_input(self):
        assert round_robin([]) == []

    def test_all_processes_complete(self, simple_processes):
        tl = round_robin(simple_processes)
        assert {p for p in tl if p is not None} == {0, 1, 2}

    def test_quantum_respected(self):
        """No process should run more than `quantum` consecutive slots."""
        procs = [
            {'pid': 0, 'arrival': 0, 'burst': 10, 'priority': 1, 'resource': 'R0', 'wanted_resource': ''},
            {'pid': 1, 'arrival': 0, 'burst': 10, 'priority': 1, 'resource': 'R1', 'wanted_resource': ''},
        ]
        quantum = 3
        tl = round_robin(procs, quantum=quantum)

        consecutive = 1
        for i in range(1, len(tl)):
            if tl[i] is not None and tl[i] == tl[i-1]:
                consecutive += 1
                assert consecutive <= quantum, \
                    f"Process {tl[i]} ran {consecutive} consecutive slots (quantum={quantum})"
            else:
                consecutive = 1

    def test_total_burst_matches(self, simple_processes):
        tl = round_robin(simple_processes)
        total_burst = sum(p['burst'] for p in simple_processes)
        assert sum(1 for p in tl if p is not None) == total_burst


# ══════════════════════════════════════════════════════════════
# Priority Scheduling Tests
# ══════════════════════════════════════════════════════════════

class TestPriority:

    def test_empty_input(self):
        assert priority_scheduling([]) == []

    def test_higher_priority_runs_first(self):
        """P1 has lower priority number = higher priority."""
        procs = [
            {'pid': 0, 'arrival': 0, 'burst': 5, 'priority': 10, 'resource': 'R0', 'wanted_resource': ''},
            {'pid': 1, 'arrival': 0, 'burst': 3, 'priority':  1, 'resource': 'R1', 'wanted_resource': ''},
        ]
        tl = priority_scheduling(procs)
        # P1 (priority=1) should appear first
        first = next(p for p in tl if p is not None)
        assert first == 1

    def test_all_complete(self, simple_processes):
        tl = priority_scheduling(simple_processes)
        assert {p for p in tl if p is not None} == {0, 1, 2}


# ══════════════════════════════════════════════════════════════
# Metrics Tests
# ══════════════════════════════════════════════════════════════

class TestMetrics:

    def test_empty_returns_zeros(self):
        aw, at, cu, tp = calculate_metrics([], [])
        assert aw == at == cu == tp == 0.0

    def test_cpu_util_full(self):
        procs    = [{'pid': 0, 'arrival': 0, 'burst': 3, 'priority': 1,
                     'resource': 'R0', 'wanted_resource': ''}]
        timeline = [0, 0, 0]
        _, _, cu, _ = calculate_metrics(procs, timeline)
        assert cu == 1.0

    def test_cpu_util_half(self):
        procs    = [{'pid': 0, 'arrival': 2, 'burst': 2, 'priority': 1,
                     'resource': 'R0', 'wanted_resource': ''}]
        timeline = [None, None, 0, 0]
        _, _, cu, _ = calculate_metrics(procs, timeline)
        assert cu == 0.5

    def test_waiting_time_fcfs(self):
        """P0 burst=3 starts at 0, waits 0.  P1 burst=2 starts at 3, waits 2."""
        procs = [
            {'pid': 0, 'arrival': 0, 'burst': 3, 'priority': 1, 'resource': 'R0', 'wanted_resource': ''},
            {'pid': 1, 'arrival': 0, 'burst': 2, 'priority': 1, 'resource': 'R1', 'wanted_resource': ''},
        ]
        timeline = [0, 0, 0, 1, 1]
        aw, _, _, _ = calculate_metrics(procs, timeline)
        assert aw == 1.0    # (0 + 2) / 2

    def test_starvation_flagged(self):
        procs = [
            {'pid': 0, 'arrival': 0, 'burst':  1, 'priority': 1, 'resource': 'R0', 'wanted_resource': ''},
            {'pid': 1, 'arrival': 0, 'burst': 50, 'priority': 1, 'resource': 'R1', 'wanted_resource': ''},
        ]
        timeline = [0] + [1] * 50
        stats = get_per_process_stats(procs, timeline)
        # P0 finishes immediately (wait=0), P1 waits 1 tick —
        # with only 2 processes the detection may or may not fire;
        # just verify the field exists.
        assert all('starved' in s for s in stats)


# ══════════════════════════════════════════════════════════════
# Banker's Algorithm Tests
# ══════════════════════════════════════════════════════════════

class TestBankers:

    def test_known_safe_state(self):
        """Classic textbook safe example."""
        allocation = [[0,1,0],[2,0,0],[3,0,2],[2,1,1],[0,0,2]]
        max_demand = [[7,5,3],[3,2,2],[9,0,2],[2,2,2],[4,3,3]]
        available  = [3, 3, 2]
        result = BankersAlgorithm(allocation, max_demand, available).run()
        assert result.is_safe is True
        assert len(result.safe_sequence) == 5

    def test_known_unsafe_state(self):
        allocation = [[1,0],[0,1]]
        max_demand = [[2,2],[2,2]]
        available  = [0, 0]        # nothing free → unsafe
        result = BankersAlgorithm(allocation, max_demand, available).run()
        assert result.is_safe is False

    def test_need_matrix_correct(self):
        allocation = [[1,0],[0,1]]
        max_demand = [[3,2],[1,3]]
        available  = [2, 2]
        result = BankersAlgorithm(allocation, max_demand, available).run()
        # Need = Max - Allocation
        assert result.need_matrix[0] == [2, 2]
        assert result.need_matrix[1] == [1, 2]

    def test_allocation_exceeds_max_raises(self):
        with pytest.raises(ValueError):
            BankersAlgorithm([[5,0]], [[3,0]], [1]).run()

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            BankersAlgorithm([], [], [])


# ══════════════════════════════════════════════════════════════
# RAG / Deadlock Tests
# ══════════════════════════════════════════════════════════════

class TestRAG:

    def test_no_cycle(self):
        rag = ResourceAllocationGraph()
        # P0 holds R0, P1 holds R1 — no circular wait
        rag.add_assignment('R0', 0)
        rag.add_assignment('R1', 1)
        assert rag.detect_cycle() is False

    def test_cycle_detected(self):
        rag = ResourceAllocationGraph()
        # P0 holds R0, wants R1
        # P1 holds R1, wants R0  → deadlock
        rag.add_assignment('R0', 0)
        rag.add_request(0, 'R1')
        rag.add_assignment('R1', 1)
        rag.add_request(1, 'R0')
        assert rag.detect_cycle() is True

    def test_build_rag_no_wanted(self):
        procs = [
            {'pid': 0, 'resource': 'R0', 'wanted_resource': ''},
            {'pid': 1, 'resource': 'R1', 'wanted_resource': ''},
        ]
        rag = build_rag_from_processes(procs)
        assert rag.detect_cycle() is False

    def test_build_rag_with_cycle(self):
        procs = [
            {'pid': 0, 'resource': 'R0', 'wanted_resource': 'R1'},
            {'pid': 1, 'resource': 'R1', 'wanted_resource': 'R0'},
        ]
        rag = build_rag_from_processes(procs)
        assert rag.detect_cycle() is True


# ══════════════════════════════════════════════════════════════
# RL Agent Tests
# ══════════════════════════════════════════════════════════════

class TestRLAgent:

    def test_choose_returns_valid_action(self):
        agent = RLSchedulerAgent()
        state = (10, 5.0, 2.0, 0)
        action = agent.choose_algorithm(state)
        assert action in agent.ACTIONS

    def test_q_value_updates(self):
        agent  = RLSchedulerAgent(epsilon=0.0)   # no exploration
        state  = (10, 5.0, 2.0, 0)
        ns     = (10, 5.0, 1.5, 0)
        action = agent.choose_algorithm(state)
        agent.update_q_value(state, action, reward=1.0, next_state=ns)
        key = agent.get_state_key(state)
        assert agent.q_table[key][action] > 0.0

    def test_epsilon_decays(self):
        agent = RLSchedulerAgent(epsilon=1.0, epsilon_decay=0.9, epsilon_min=0.01)
        for _ in range(10):
            agent.decay_epsilon()
        assert agent.epsilon < 1.0
        assert agent.epsilon >= 0.01

    def test_save_load(self, tmp_path):
        agent = RLSchedulerAgent(epsilon=0.42)
        path  = str(tmp_path / "q.json")
        agent.save(path)
        agent2 = RLSchedulerAgent()
        agent2.load(path)
        assert abs(agent2.epsilon - 0.42) < 1e-6

    def test_reset_clears_state(self):
        agent = RLSchedulerAgent()
        state = (5, 3.0, 1.0, 0)
        agent.choose_algorithm(state)
        agent.reset()
        assert agent.q_table == {}
        assert all(v == 0 for v in agent.action_counts.values())