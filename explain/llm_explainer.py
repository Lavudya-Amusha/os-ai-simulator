"""
llm_explainer.py
----------------
AI explanation engine for RL scheduling decisions.
Project: AI-Based OS Scheduler with Predictive Deadlock Management using LLM

API priority:  Groq (free) → Anthropic → Ollama → offline fallback
Setup:         Add GROQ_API_KEY to .streamlit/secrets.toml
"""

from __future__ import annotations
import os, json, urllib.request, urllib.error

_ALGO_INFO = {
    "FCFS":     {"full":"First-Come First-Served","type":"Non-preemptive","complexity":"O(n log n)","strength":"simple, fair ordering, no starvation","weakness":"convoy effect — long jobs block short ones","best_when":"similar burst times, batch workloads","os":"IBM OS/360 batch systems"},
    "SJF":      {"full":"Shortest Job First","type":"Non-preemptive","complexity":"O(n²)","strength":"provably minimises average waiting time","weakness":"starvation of long processes","best_when":"burst times vary widely; avg wait minimisation is goal","os":"HPC cluster batch schedulers"},
    "RR":       {"full":"Round Robin","type":"Preemptive (quantum=3)","complexity":"O(n·T/q)","strength":"fairness, good response time for all processes","weakness":"high context-switch overhead for CPU-bound jobs","best_when":"interactive/time-sharing systems","os":"Traditional Unix time-sharing"},
    "MLFQ":     {"full":"Multi-Level Feedback Queue","type":"Preemptive (3 levels)","complexity":"O(n·T)","strength":"adapts to process behaviour automatically","weakness":"complex tuning, possible starvation at lower levels","best_when":"mixed workloads of unknown type at arrival","os":"Windows NT, macOS kernel scheduler"},
    "Priority": {"full":"Preemptive Priority Scheduling","type":"Preemptive","complexity":"O(n²)","strength":"critical processes get CPU immediately","weakness":"low-priority process starvation without aging","best_when":"real-time systems with clear priority tiers","os":"VxWorks, FreeRTOS real-time OS"},
}


# ── Key retrieval ──────────────────────────────────────────────

def _get_key(name: str) -> str:
    """Read from Streamlit secrets first, then environment variable."""
    try:
        import streamlit as st
        v = st.secrets.get(name, "")
        if v:
            return str(v)
    except Exception:
        pass
    return os.environ.get(name, "")


# ── API callers ────────────────────────────────────────────────

def _call_groq(prompt: str) -> str:
    key = _get_key("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not found in secrets.toml or environment")

    payload = json.dumps({
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.3,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data    = payload,
        headers = {
            "Content-Type" : "application/json",
            "Authorization": f"Bearer {key}",
        },
        method = "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Groq HTTP {e.code}: {body[:200]}")


def _call_anthropic(prompt: str) -> str:
    key = _get_key("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not found")

    payload = json.dumps({
        "model"     : "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "messages"  : [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data    = payload,
        headers = {
            "Content-Type"      : "application/json",
            "x-api-key"         : key,
            "anthropic-version" : "2023-06-01",
        },
        method = "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data["content"][0]["text"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Anthropic HTTP {e.code}: {body[:200]}")


def _call_ollama(prompt: str) -> str:
    payload = json.dumps({
        "model" : "llama3.2",
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data    = payload,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data["response"]
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")


# ── Prompt builder ─────────────────────────────────────────────

def _build_prompt(chosen_algo, state, q_values, comparison):
    n, b, w, d = state
    ranked  = sorted(q_values.items(), key=lambda x: x[1], reverse=True)
    q_lines = "\n".join(
        f"  {a}: {v:.4f}" + (" <- CHOSEN" if a == chosen_algo else "")
        for a, v in ranked
    )
    cmp_lines = ""
    if comparison:
        cmp_lines = "\nAll algorithm performance on this dataset (same 200 processes):\n"
        for a in sorted(comparison, key=lambda x: comparison[x]["avg_wait"]):
            m = comparison[a]
            cmp_lines += (
                f"  {a}: avg_wait={m['avg_wait']:.2f}, "
                f"avg_turn={m['avg_turn']:.2f}, "
                f"cpu_util={m['cpu_util']*100:.1f}%\n"
            )

    info = _ALGO_INFO.get(chosen_algo, {})
    return (
        "You are an expert in operating systems and reinforcement learning, "
        "helping a CS student understand their AI-Based OS Scheduler with "
        "Predictive Deadlock Management project.\n\n"
        "PROJECT CONTEXT:\n"
        "- The student built a CPU scheduling simulator using Q-learning\n"
        "- Dataset: real Google Borg cluster traces (200 processes)\n"
        "- 5 algorithms: FCFS, SJF, Round Robin, MLFQ, Priority Scheduling\n"
        "- Deadlock detected via Resource Allocation Graph (RAG) cycle detection\n"
        "- Banker's Algorithm used for deadlock avoidance (safe state check)\n\n"
        f"CURRENT SIMULATION DATA:\n"
        f"  Processes: {n}, avg_burst={b:.2f} units, deadlock_detected={'Yes' if d else 'No'}\n"
        f"  Algorithm run: {chosen_algo} ({info.get('full','')})\n"
        f"  avg_wait={w:.2f} units\n"
        f"{cmp_lines}\n"
        f"RL AGENT Q-VALUES (after training):\n{q_lines}\n\n"
        f"REWARD FUNCTION: reward = -(0.6 x avg_wait) - (0.3 x avg_turnaround) + (0.1 x throughput)\n\n"
        f"Write a technically precise explanation in Markdown covering:\n"
        f"1. Workload characterisation: what do these numbers (burst={b:.1f}, {n} procs) tell us?\n"
        f"2. Q-value analysis: why did the agent converge on {chosen_algo}? Use actual Q-values.\n"
        f"3. OS theory validation: is this the correct choice according to scheduling theory?\n"
        f"4. Reward function impact: how did the 0.6/0.3/0.1 weights drive this convergence?\n"
        f"5. Deadlock context: {'explain how deadlock coexists with scheduling (separate OS subsystems)' if d else 'note that no deadlock was detected'}\n"
        f"6. One specific insight the student can state confidently in their project viva\n\n"
        f"Use the actual numbers throughout. Be technically precise. Under 450 words."
    )


# ── Offline fallback ───────────────────────────────────────────

def _offline_explanation(chosen_algo, state, q_values, comparison):
    n, b, w, d = state
    info    = _ALGO_INFO.get(chosen_algo, {})
    ranked  = sorted(q_values.items(), key=lambda x: x[1], reverse=True)
    rank    = next((i+1 for i,(a,_) in enumerate(ranked) if a==chosen_algo), 1)
    our_q   = q_values.get(chosen_algo, 0.0)
    suffix  = {1:"st",2:"nd",3:"rd"}.get(rank,"th")
    lbl     = "highest" if rank==1 else f"{rank}{suffix} highest"
    sz      = "small" if n<5 else "medium" if n<15 else "large"
    profile = "short-burst (interactive)" if b<5 else "medium-burst (mixed)" if b<15 else "long-burst (CPU-bound)"
    pb      = 0 if n<5 else 1 if n<15 else 2
    bb      = 0 if b<5 else 1 if b<15 else 2
    wb      = 0 if w<10 else 1 if w<30 else 2

    q_rows = "".join(
        f"| {'**'+a+'**' if a==chosen_algo else a} | {v:.4f} | {'← chosen' if a==chosen_algo else ''} |\n"
        for a, v in ranked
    )

    cmp_section = ""
    if comparison:
        all_w = {a: comparison[a]['avg_wait'] for a in comparison}
        worst = max(all_w, key=all_w.get)
        best  = min(all_w, key=all_w.get)
        our_w = all_w.get(chosen_algo, 0)
        pct   = (all_w[worst]-our_w)/all_w[worst]*100 if all_w[worst]>0 else 0
        rows  = "".join(
            f"| {a} | {comparison[a]['avg_wait']:.2f} | "
            f"{comparison[a]['avg_turn']:.2f} | "
            f"{comparison[a]['cpu_util']*100:.1f}% | "
            f"{'✅ best' if a==best else ('❌ worst' if a==worst else '')} |\n"
            for a in sorted(comparison, key=lambda x: comparison[x]['avg_wait'])
        )
        cmp_section = f"""
---

### Algorithm Comparison (all 5 on same dataset)

| Algorithm | Avg Wait | Avg Turn | CPU Util | |
|-----------|----------|----------|----------|--|
{rows}

**{chosen_algo}** achieves **{pct:.1f}% lower waiting time** than the worst algorithm ({worst}: {all_w[worst]:.2f} units).
"""

    dl_note = (
        "\n> **Deadlock note:** The RAG cycle detection flagged a circular-wait condition "
        "in this workload. However, CPU scheduling and resource management are **separate OS subsystems**. "
        "The scheduler assigns CPU time slots independently — it does not simulate resource acquisition. "
        "Deadlock is detected and reported for the Banker's Algorithm analysis; it does not block scheduling."
        if d else ""
    )

    return f"""## RL Agent Decision Analysis — {chosen_algo}

**Project:** AI-Based OS Scheduler with Predictive Deadlock Management using LLM

**{info.get('full', chosen_algo)}** · {info.get('type','')} · Complexity: {info.get('complexity','')}

---

### Workload Characterisation

| Property | Value | Interpretation |
|----------|-------|----------------|
| Processes | {n} | {sz} workload from Google Borg traces |
| Avg burst | {b:.2f} units | {profile} |
| Avg wait | {w:.2f} units | {'minimal' if w<2 else 'low' if w<10 else 'moderate'} wait pressure |
| Deadlock | {'Yes ⚠️' if d else 'No ✅'} | {'RAG cycle detected — see Deadlock page' if d else 'No circular dependencies in RAG'} |
| State key | `({pb},{bb},{wb},{int(bool(d))})` | Discretised representation used in Q-table lookup |

---

### Q-Value Analysis

The RL agent's learned Q-values for the current state:

| Algorithm | Q-Value | |
|-----------|---------|--|
{q_rows}

**{chosen_algo}** holds the **{lbl} Q-value at {our_q:.4f}**.

Q-values are always negative because the reward penalty (avg_wait × 0.6) dominates the small throughput bonus.
The agent learns **relative differences** — {chosen_algo} consistently produced the least-negative reward
for this workload type across training episodes.

---

### OS Theory Validation

- **Strength**: {info.get('strength','')}
- **Best suited for**: {info.get('best_when','')}
- **Real-world OS usage**: {info.get('os','')}
- **Known limitation**: {info.get('weakness','')}
{cmp_section}
---

### Reward Function Impact

```
reward = −(0.6 × avg_wait) − (0.3 × avg_turnaround) + (0.1 × throughput)
```

The weight **0.6** on `avg_wait` is the dominant term — reflecting that users perceive
waiting time most directly. The agent was incentivised to minimise this above all else.
**{chosen_algo}** earned higher rewards because it {'minimises waiting time by always picking the shortest ready job' if chosen_algo=='SJF' else 'produced the best reward balance for this workload type'}.
{dl_note}

---

### Viva Insight

> For a **{n}-process** workload from Google Borg cluster traces
> (avg burst = {b:.1f} units, {'deadlock present' if d else 'no deadlock'}),
> the Q-learning agent converged on **{chosen_algo}** after training.
> This demonstrates that tabular Q-learning with ε-greedy exploration
> (ε decaying from 1.0 to 0.05) can learn near-optimal scheduling policy
> selection from real cluster workload data — without manual algorithm selection
> or knowledge of burst times in advance.
"""


# ── Public API ─────────────────────────────────────────────────

def explain_decision(
    chosen_algo : str,
    state       : tuple,
    q_values    : dict,
    comparison  : dict | None = None,
) -> tuple[str, str]:
    """
    Returns (markdown_explanation, source_string).

    Tries: Groq → Anthropic → Ollama → offline fallback.
    Errors are logged in source string so the UI can show what went wrong.
    """
    prompt = _build_prompt(chosen_algo, state, q_values, comparison)
    errors = []

    # 1. Groq (free)
    try:
        text = _call_groq(prompt)
        return text, "Groq — llama-3.3-70b (free)"
    except RuntimeError as e:
        errors.append(f"Groq: {e}")
    except Exception as e:
        errors.append(f"Groq: unexpected error — {e}")

    # 2. Anthropic
    try:
        text = _call_anthropic(prompt)
        return text, "Anthropic — claude-sonnet"
    except RuntimeError as e:
        errors.append(f"Anthropic: {e}")
    except Exception as e:
        errors.append(f"Anthropic: unexpected error — {e}")

    # 3. Ollama local
    try:
        text = _call_ollama(prompt)
        return text, "Ollama — llama3.2 (local)"
    except RuntimeError as e:
        errors.append(f"Ollama: {e}")
    except Exception as e:
        errors.append(f"Ollama: {e}")

    # 4. Offline fallback
    text = _offline_explanation(chosen_algo, state, q_values, comparison)
    error_summary = " | ".join(errors)
    return text, f"offline-fallback ({error_summary})"