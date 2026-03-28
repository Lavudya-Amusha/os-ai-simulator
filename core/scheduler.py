"""
scheduler.py
------------
Five CPU scheduling algorithms used by the OS Scheduling Simulator.

Each function accepts:
    processes : list[dict]
        Each dict must contain: pid (int), arrival (int), burst (int),
        priority (int — lower = higher priority)

Each function returns:
    timeline : list
        Ordered list of pid values (one per time unit).
        None entries represent CPU idle slots.

Complexity summary
------------------
    FCFS     : O(n log n)  — sort by arrival
    SJF      : O(n²)       — linear scan per scheduling point
    RR       : O(n · T/q)  — T = total burst, q = quantum
    MLFQ     : O(n · T)    — constant work per time unit
    Priority : O(n²)       — linear scan per scheduling point
"""

from collections import deque
import copy


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _sorted_by_arrival(processes: list[dict]) -> list[dict]:
    return sorted(processes, key=lambda p: (p['arrival'], p['pid']))


def _normalize_arrivals(processes: list[dict]) -> list[dict]:
    """
    Shift arrival times so the earliest process arrives at t=0,
    and cap burst times to prevent MemoryError on large datasets.

    This is essential when the source data contains raw timestamps
    (e.g. nanoseconds from Google Borg traces).
    """
    if not processes:
        return processes

    min_arrival = min(int(p['arrival']) for p in processes)
    MAX_BURST   = 500

    normalized = []
    for p in processes:
        np_ = dict(p)
        np_['arrival'] = max(0, int(p['arrival']) - min_arrival)
        np_['burst']   = max(1, min(int(p['burst']), MAX_BURST))
        normalized.append(np_)
    return normalized


# ──────────────────────────────────────────────────────────────
# 1. First-Come First-Served  (non-preemptive)  O(n log n)
# ──────────────────────────────────────────────────────────────

def fcfs(processes: list[dict]) -> list:
    """
    Non-preemptive FCFS — processes run in strict arrival order.

    Complexity : O(n log n) for the initial sort.
    Best case  : All processes arrive at t=0 — no idle gaps.
    Worst case : Processes arrive far apart — long idle gaps.
    Use when   : Simplicity matters; workload is batch-oriented.
    """
    if not processes:
        return []

    procs        = _sorted_by_arrival(_normalize_arrivals(copy.deepcopy(processes)))
    timeline     = []
    current_time = 0

    for p in procs:
        if current_time < p['arrival']:
            timeline.extend([None] * (p['arrival'] - current_time))
            current_time = p['arrival']
        timeline.extend([p['pid']] * p['burst'])
        current_time += p['burst']

    return timeline


# ──────────────────────────────────────────────────────────────
# 2. Shortest Job First  (non-preemptive)  O(n²)
# ──────────────────────────────────────────────────────────────

def sjf(processes: list[dict]) -> list:
    """
    Non-preemptive SJF — at each scheduling decision pick the
    ready process with the smallest burst time.

    Complexity : O(n²) — linear scan over remaining processes
                 at each scheduling decision.
    Best case  : All processes arrive at t=0 with equal burst.
    Worst case : Long jobs arrive first and block short ones.
    Use when   : Minimising average waiting time; batch workloads
                 where burst times are known in advance.
    """
    if not processes:
        return []

    procs        = _normalize_arrivals(copy.deepcopy(processes))
    timeline     = []
    current_time = 0
    remaining    = list(procs)

    while remaining:
        ready = [p for p in remaining if p['arrival'] <= current_time]

        if not ready:
            next_arr = min(p['arrival'] for p in remaining)
            timeline.extend([None] * (next_arr - current_time))
            current_time = next_arr
            continue

        chosen = min(ready, key=lambda p: (p['burst'], p['arrival'], p['pid']))
        remaining.remove(chosen)
        timeline.extend([chosen['pid']] * chosen['burst'])
        current_time += chosen['burst']

    return timeline


# ──────────────────────────────────────────────────────────────
# 3. Round Robin  (preemptive)  O(n · T/q)
# ──────────────────────────────────────────────────────────────

def round_robin(processes: list[dict], quantum: int = 3) -> list:
    """
    Preemptive Round Robin with configurable time quantum.

    Complexity : O(n · T/q) where T = total burst time,
                 q = quantum.
    Best case  : All processes have burst ≤ quantum — each runs once.
    Worst case : Many long processes with small quantum — high overhead.
    Use when   : Interactive/time-sharing systems; fairness required.

    Parameters
    ----------
    quantum : int   Time slice per process (default 3).
    """
    if not processes:
        return []

    procs           = _sorted_by_arrival(_normalize_arrivals(copy.deepcopy(processes)))
    remaining_burst = {p['pid']: p['burst'] for p in procs}
    timeline        = []
    ready_queue     = deque()
    current_time    = 0
    index           = 0
    n               = len(procs)

    while index < n and procs[index]['arrival'] <= current_time:
        ready_queue.append(procs[index]['pid'])
        index += 1

    while ready_queue or index < n:
        if not ready_queue:
            next_arr = procs[index]['arrival']
            timeline.extend([None] * (next_arr - current_time))
            current_time = next_arr
            while index < n and procs[index]['arrival'] <= current_time:
                ready_queue.append(procs[index]['pid'])
                index += 1

        pid   = ready_queue.popleft()
        burst = min(quantum, remaining_burst[pid])

        for _ in range(burst):
            timeline.append(pid)
            current_time += 1
            while index < n and procs[index]['arrival'] <= current_time:
                ready_queue.append(procs[index]['pid'])
                index += 1

        remaining_burst[pid] -= burst
        if remaining_burst[pid] > 0:
            ready_queue.append(pid)

    return timeline


# ──────────────────────────────────────────────────────────────
# 4. Multi-Level Feedback Queue  O(n · T)
# ──────────────────────────────────────────────────────────────

def mlfq(processes: list[dict], quantums: list[int] | None = None) -> list:
    """
    Three-level MLFQ:
        Level 0 — quantum 2  (highest priority, interactive)
        Level 1 — quantum 4
        Level 2 — FCFS       (lowest priority, CPU-bound)

    A process demotes to the next level on quantum expiry.
    It never promotes upward.

    Complexity : O(n · T) — proportional to total burst time.
    Best case  : All short jobs finish in level 0.
    Worst case : Long jobs cycle through all levels repeatedly.
    Use when   : Mixed workload (interactive + batch); Linux-like
                 scheduling.
    """
    if not processes:
        return []

    if quantums is None:
        quantums = [2, 4, 8]

    procs           = _sorted_by_arrival(_normalize_arrivals(copy.deepcopy(processes)))
    remaining_burst = {p['pid']: p['burst']   for p in procs}
    queues          = [deque(), deque(), deque()]
    timeline        = []
    current_time    = 0
    index           = 0
    n               = len(procs)

    def _enqueue_arrivals():
        nonlocal index
        while index < n and procs[index]['arrival'] <= current_time:
            queues[0].append(procs[index]['pid'])
            index += 1

    _enqueue_arrivals()

    while any(queues) or index < n:
        level = next((i for i in range(3) if queues[i]), None)

        if level is None:
            next_arr = procs[index]['arrival']
            timeline.extend([None] * (next_arr - current_time))
            current_time = next_arr
            _enqueue_arrivals()
            continue

        pid   = queues[level].popleft()
        burst = min(quantums[level] if level < 2 else remaining_burst[pid],
                    remaining_burst[pid])

        for _ in range(burst):
            timeline.append(pid)
            current_time += 1
            _enqueue_arrivals()

        remaining_burst[pid] -= burst
        if remaining_burst[pid] > 0:
            queues[min(level + 1, 2)].append(pid)

    return timeline


# ──────────────────────────────────────────────────────────────
# 5. Preemptive Priority Scheduling  O(n²)
# ──────────────────────────────────────────────────────────────

def priority_scheduling(processes: list[dict]) -> list:
    """
    Preemptive Priority Scheduling — lower priority number =
    higher priority.  At each tick the highest-priority ready
    process runs.  If a higher-priority process arrives mid-
    execution the current process is preempted immediately.

    Complexity : O(n²) — at each time unit we scan ready queue.
    Best case  : All processes have same priority → degrades to FCFS.
    Worst case : Continuous stream of high-priority arrivals causes
                 starvation of low-priority processes.
    Use when   : Real-time systems; OS kernel scheduling where
                 process importance is known.

    Note       : Starvation is possible. The caller can detect it
                 via metrics.get_per_process_stats() by checking
                 waiting time against the starvation threshold.
    """
    if not processes:
        return []

    procs           = _normalize_arrivals(copy.deepcopy(processes))
    remaining_burst = {p['pid']: p['burst']    for p in procs}
    priority_map    = {p['pid']: p['priority'] for p in procs}
    arrival_map     = {p['pid']: p['arrival']  for p in procs}
    all_pids        = [p['pid'] for p in procs]

    timeline     = []
    current_time = 0
    total_burst  = sum(p['burst'] for p in procs)

    while sum(remaining_burst.values()) > 0:
        ready = [
            pid for pid in all_pids
            if arrival_map[pid] <= current_time
            and remaining_burst[pid] > 0
        ]

        if not ready:
            timeline.append(None)
            current_time += 1
            continue

        # Highest priority = lowest numeric value; tie-break by pid
        chosen = min(ready, key=lambda pid: (priority_map[pid], pid))
        timeline.append(chosen)
        remaining_burst[chosen] -= 1
        current_time += 1

    return timeline