"""
load_borg_data.py
-----------------
Loads and cleans Google Borg cluster trace data for use in the
OS scheduling simulator.

Key transformations applied
---------------------------
1. Timestamps (nanoseconds) → normalized integer arrival times (0-based)
2. CPU usage distribution   → integer burst time (1–100 units)
3. Missing / invalid values → safe defaults
4. pid values               → plain integers (required by schedulers)
"""

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

# After normalization, scale arrival times into this range.
# Keeps all idle gaps between 0 and MAX_ARRIVAL_UNITS.
MAX_ARRIVAL_UNITS = 200

# Burst time range (scheduler time units)
MIN_BURST = 1
MAX_BURST = 100

# Number of distinct simulated resources
NUM_RESOURCES = 5


# ──────────────────────────────────────────────────────────────
# CPU usage parser
# ──────────────────────────────────────────────────────────────

def _parse_cpu_distribution(cpu_value):
    """
    Parse the cpu_usage_distribution column which may be:
      - float / int          e.g.  0.034
      - NaN
      - string array         e.g.  "[0.00314 0.00381 0.00401 ...]"
      - scientific notation  e.g.  "[1.23977661e-05 ...]"

    Returns a single float representing mean CPU usage.
    """
    if pd.isna(cpu_value):
        return 0.01                        # safe default: minimal burst

    if isinstance(cpu_value, (int, float)):
        return max(0.0, float(cpu_value))

    if isinstance(cpu_value, str):
        try:
            cleaned = (
                cpu_value
                .replace('[', '')
                .replace(']', '')
                .replace('\n', ' ')
            )
            values = [
                float(x)
                for x in cleaned.split()
                if x.strip()
            ]
            return float(np.mean(values)) if values else 0.01
        except Exception:
            return 0.01

    return 0.01


# ──────────────────────────────────────────────────────────────
# Main loader
# ──────────────────────────────────────────────────────────────

def load_borg_processes(csv_path: str, limit: int = 200) -> list:
    """
    Load at most `limit` rows from the Borg CSV and return a list
    of process dicts ready for OSSimulator.

    Each dict contains
    ------------------
    pid      : int
    arrival  : int    normalized to 0 – MAX_ARRIVAL_UNITS
    burst    : int    1 – MAX_BURST  (scheduler time units)
    priority : int    original Borg priority value
    resource : str    "R0" … "R{NUM_RESOURCES-1}"

    Raises
    ------
    FileNotFoundError if csv_path does not exist.
    ValueError        if the CSV has no usable rows after cleaning.
    """
   df = pd.read_csv(
    csv_path,
    nrows=limit,
    engine="python",
    sep=",",
    on_bad_lines="skip",
    encoding="utf-8",
    low_memory=False
)

    if df.empty:
        raise ValueError(f"CSV at '{csv_path}' is empty.")

    # ── 1. Arrival time ──────────────────────────────────────
    # Raw values are nanosecond timestamps (e.g. 2_517_305_308_183).
    # Normalize: subtract minimum, then scale into [0, MAX_ARRIVAL_UNITS].

    raw_times = pd.to_numeric(df['time'], errors='coerce').fillna(0)

    t_min = raw_times.min()
    t_max = raw_times.max()
    t_range = t_max - t_min

    if t_range > 0:
        # Scale to [0, MAX_ARRIVAL_UNITS] and convert to int
        arrivals = (
            ((raw_times - t_min) / t_range) * MAX_ARRIVAL_UNITS
        ).astype(int)
    else:
        # All timestamps identical — stagger arrivals by index
        arrivals = pd.Series(range(len(df)))

    # ── 2. CPU burst ─────────────────────────────────────────
    # Mean CPU usage is a small float (e.g. 0.034).
    # Scale to [MIN_BURST, MAX_BURST] so schedulers have meaningful work.

    cpu_means = df['cpu_usage_distribution'].apply(_parse_cpu_distribution)

    c_min = cpu_means.min()
    c_max = cpu_means.max()
    c_range = c_max - c_min

    if c_range > 0:
        bursts = (
            MIN_BURST
            + ((cpu_means - c_min) / c_range) * (MAX_BURST - MIN_BURST)
        ).astype(int).clip(MIN_BURST, MAX_BURST)
    else:
        bursts = pd.Series([MIN_BURST] * len(df))

    # ── 3. Priority ──────────────────────────────────────────
    priorities = (
        pd.to_numeric(df.get('priority', pd.Series([1] * len(df))),
                      errors='coerce')
        .fillna(1)
        .astype(int)
    )

    # ── 4. Resource (from collection_id mod NUM_RESOURCES) ───
    collection_ids = (
        pd.to_numeric(df.get('collection_id', pd.Series([0] * len(df))),
                      errors='coerce')
        .fillna(0)
        .astype(int)
    )
    resources = (collection_ids % NUM_RESOURCES).apply(lambda x: f"R{x}")

    # ── 5. Assemble process list ─────────────────────────────
    processes = []
    for i in range(len(df)):
        processes.append({
            'pid'     : i,
            'arrival' : int(arrivals.iloc[i]),
            'burst'   : int(bursts.iloc[i]),
            'priority': int(priorities.iloc[i]),
            'resource': str(resources.iloc[i]),
        })

    if not processes:
        raise ValueError("No valid processes could be parsed from the dataset.")

    # ── 6. Add wanted_resource for RAG deadlock detection ────
    _add_wanted_resources(processes)

    return processes


def _add_wanted_resources(processes):
    """
    Assign a 'wanted_resource' field to each process for realistic RAG
    deadlock detection.

    Logic
    -----
    - A process HOLDS its assigned 'resource'.
    - High-priority processes (above median priority) also WAIT for
      the resource held by the next process in arrival order.
    - We only create a wait edge when wanted != held, ensuring a genuine
      wait-for relationship rather than a trivial self-loop.
    - ~50% of processes get a wait edge → realistic partial deadlock
      (not every run deadlocks, which makes detection meaningful).
    """
    n               = len(processes)
    priorities      = [p['priority'] for p in processes]
    median_priority = sorted(priorities)[n // 2]

    resource_list = [p['resource'] for p in processes]

    for i, p in enumerate(processes):
        if p['priority'] > median_priority:
            # Want the resource held by the next process (circular index)
            wanted = resource_list[(i + 1) % n]
            p['wanted_resource'] = wanted if wanted != p['resource'] else ""
        else:
            p['wanted_resource'] = ""

    return processes