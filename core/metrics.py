"""
metrics.py
----------
Scheduling performance metrics derived from a pid timeline.

Timeline format: [pid, pid, None, pid, ...]
    pid  → CPU executing that process at that time unit
    None → CPU idle
"""

from __future__ import annotations

# Starvation threshold: a process is "starved" if its waiting
# time exceeds this multiple of the average waiting time.
STARVATION_MULTIPLIER = 2.0


def calculate_metrics(
    processes: list[dict],
    timeline: list,
) -> tuple[float, float, float, float]:
    """
    Derive standard OS scheduling metrics from the execution timeline.

    Parameters
    ----------
    processes : list[dict]   {'pid', 'arrival', 'burst', ...}
    timeline  : list         output of any scheduler function

    Returns
    -------
    avg_wait    : float   average waiting time across all processes
    avg_turn    : float   average turnaround time
    cpu_util    : float   fraction of time CPU is busy  (0.0 – 1.0)
    throughput  : float   processes completed per unit time
    """
    if not processes or not timeline:
        return 0.0, 0.0, 0.0, 0.0

    arrival    = {p['pid']: int(p['arrival']) for p in processes}
    burst      = {p['pid']: int(p['burst'])   for p in processes}
    completion = {}

    for t, pid in enumerate(timeline):
        if pid is not None:
            completion[pid] = t + 1

    total_wait = 0
    total_turn = 0

    for p in processes:
        pid        = p['pid']
        comp       = completion.get(pid, len(timeline))
        turnaround = max(0, comp - arrival[pid])
        waiting    = max(0, turnaround - burst[pid])
        total_turn += turnaround
        total_wait += waiting

    n          = len(processes)
    avg_wait   = total_wait / n
    avg_turn   = total_turn / n
    total_time = len(timeline)
    busy_units = sum(1 for slot in timeline if slot is not None)
    cpu_util   = busy_units / total_time if total_time > 0 else 0.0
    throughput = n / total_time          if total_time > 0 else 0.0

    return avg_wait, avg_turn, cpu_util, throughput


def get_per_process_stats(
    processes: list[dict],
    timeline: list,
) -> list[dict]:
    """
    Return a per-process breakdown — used for the results table in UI.

    Each dict contains:
        pid, arrival, burst, completion, turnaround, waiting, starved
    """
    if not processes or not timeline:
        return []

    arrival    = {p['pid']: int(p['arrival']) for p in processes}
    burst      = {p['pid']: int(p['burst'])   for p in processes}
    completion = {}

    for t, pid in enumerate(timeline):
        if pid is not None:
            completion[pid] = t + 1

    stats = []
    for p in processes:
        pid  = p['pid']
        comp = completion.get(pid, len(timeline))
        ta   = max(0, comp - arrival[pid])
        wt   = max(0, ta - burst[pid])
        stats.append({
            'pid'        : pid,
            'arrival'    : arrival[pid],
            'burst'      : burst[pid],
            'completion' : comp,
            'turnaround' : ta,
            'waiting'    : wt,
            'starved'    : False,   # filled in below
        })

    # ── Starvation detection ──────────────────────────────────
    # A process is starved if its waiting time exceeds
    # STARVATION_MULTIPLIER × average waiting time.
    if stats:
        avg_wt = sum(s['waiting'] for s in stats) / len(stats)
        threshold = STARVATION_MULTIPLIER * avg_wt
        for s in stats:
            s['starved'] = (s['waiting'] > threshold and avg_wt > 0)

    return sorted(stats, key=lambda x: x['pid'])