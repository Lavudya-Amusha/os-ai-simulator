"""
frontend.py
-----------
Streamlit UI for the AI-Based OS Scheduling Simulator.
Matches the updated simulator.py API:
    run_with_algorithm() → (timeline, deadlock, avg_wait, avg_turn,
                             cpu_util, throughput, per_process)
    step()               → (next_state, reward, timeline, deadlock)
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd

from core.simulator     import OSSimulator
from core.rl_agent      import RLSchedulerAgent
from data.load_borg_data import load_borg_processes

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="OS Scheduling Simulator",
    page_icon="🖥️",
    layout="wide",
)

st.title("🖥️ AI-Based OS Scheduling Simulator")

# ──────────────────────────────────────────────────────────────
# Consistent colour palette for all charts
# ──────────────────────────────────────────────────────────────

ALGO_COLORS = {
    "FCFS" : "#4C72B0",
    "SJF"  : "#DD8452",
    "RR"   : "#55A868",
    "MLFQ" : "#C44E52",
}
ALGOS = list(ALGO_COLORS.keys())

# ──────────────────────────────────────────────────────────────
# Session-state initialisation (prevents KeyError on first run)
# ──────────────────────────────────────────────────────────────

_DEFAULTS = {
    "processes"   : None,
    "rewards"     : [],
    "algo_counts" : {},
    "timeline"    : None,
    "metrics"     : None,
    "per_process" : None,
    "agent"       : None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────

st.sidebar.title("⚙️ Controls")

data_limit = st.sidebar.slider("Dataset Size",   50,  500, 200)
iterations = st.sidebar.slider("RL Iterations",  10,  300, 100)

page = st.sidebar.radio(
    "📌 Navigation",
    ["Dashboard", "RL Training", "Simulation", "Analysis", "Deadlock"],
)

st.sidebar.divider()

if st.sidebar.button("📂 Load Dataset", use_container_width=True):
    with st.spinner("Loading Borg traces…"):
        try:
            processes = load_borg_processes(
                "data/borg_traces_data.csv", limit=data_limit
            )
            st.session_state["processes"] = processes
            # Reset downstream state when a new dataset is loaded
            st.session_state["timeline"]    = None
            st.session_state["metrics"]     = None
            st.session_state["per_process"] = None
            st.session_state["rewards"]     = []
            st.session_state["algo_counts"] = {}
            st.session_state["agent"]       = None
            st.sidebar.success(f"Loaded {len(processes)} processes.")
        except FileNotFoundError:
            st.sidebar.error("CSV not found at data/borg_traces_data.csv")
        except Exception as e:
            st.sidebar.error(f"Load failed: {e}")

# ── Guard: nothing works without data ──────────────────────────
if st.session_state["processes"] is None:
    st.info("👈 Click **Load Dataset** in the sidebar to begin.")
    st.stop()

processes = st.session_state["processes"]
sim       = OSSimulator(processes)

# Persist the RL agent across pages so training isn't lost
if st.session_state["agent"] is None:
    st.session_state["agent"] = RLSchedulerAgent()
agent = st.session_state["agent"]


# ══════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ══════════════════════════════════════════════════════════════

if page == "Dashboard":
    st.header("📊 System Overview")

    # Workflow guide
    with st.expander("📖 How to use this simulator", expanded=False):
        st.markdown(
            """
            **Step-by-step workflow**

            1. **Load Dataset** — use the sidebar slider to choose how many
               Borg traces to load, then click *Load Dataset*.
            2. **RL Training** — train the Q-learning agent to discover which
               scheduling algorithm performs best for this workload.
            3. **Simulation** — manually pick an algorithm (or ask the RL agent)
               and run a full simulation.
            4. **Analysis** — inspect the Gantt chart, CPU utilisation, and a
               side-by-side algorithm comparison.
            5. **Deadlock** — visualise the Resource Allocation Graph and run
               the Banker's safety check.
            """
        )

    state = sim.get_state()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Processes", state[0])
    c2.metric("Avg Burst Time",  f"{state[1]:.2f}")
    c3.metric("Avg Wait Time",   f"{state[2]:.2f}")
    c4.metric("Deadlock Risk",   "⚠️ Yes" if state[3] else "✅ No")

    st.divider()

    # Dataset preview
    st.subheader("Dataset Preview")
    df_preview = pd.DataFrame(processes[:10])
    st.dataframe(df_preview, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: RL Training
# ══════════════════════════════════════════════════════════════

elif page == "RL Training":
    st.header("🤖 RL Agent Training")

    col_btn1, col_btn2 = st.columns([2, 1])

    with col_btn1:
        train_btn = st.button("▶ Train RL Agent", use_container_width=True)
    with col_btn2:
        if st.button("🔄 Reset Agent", use_container_width=True):
            st.session_state["agent"] = RLSchedulerAgent()
            st.session_state["rewards"]     = []
            st.session_state["algo_counts"] = {}
            agent = st.session_state["agent"]
            st.success("Agent reset.")

    if train_btn:
        rewards     = []
        algo_counts = {a: 0 for a in ALGOS}

        progress_bar = st.progress(0, text="Training…")

        for i in range(iterations):
            state  = sim.get_state()
            action = agent.choose_algorithm(state)
            algo_counts[action] += 1

            next_state, reward, _, _ = sim.step(action)
            agent.update_q_value(state, action, reward, next_state)
            agent.decay_epsilon()                 # ← epsilon decay each step

            rewards.append(reward)
            progress_bar.progress((i + 1) / iterations,
                                   text=f"Iteration {i+1}/{iterations}")

        st.session_state["rewards"]     = rewards
        st.session_state["algo_counts"] = algo_counts
        progress_bar.empty()
        st.success(
            f"Training complete. "
            f"Best algorithm: **{agent.best_action_overall}** | "
            f"Final ε: {agent.epsilon:.4f}"
        )

    # ── Learning curve ─────────────────────────────────────────
    if st.session_state["rewards"]:
        rewards = st.session_state["rewards"]

        st.subheader("📈 Learning Curve")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(rewards, color="#4C72B0", linewidth=1, alpha=0.8)
        ax.axhline(y=sum(rewards) / len(rewards),
                   color="red", linestyle="--", linewidth=1,
                   label=f"Mean reward: {sum(rewards)/len(rewards):.2f}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Reward")
        ax.set_title("Q-Learning Reward over Training Iterations")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    # ── Algorithm preference ───────────────────────────────────
    if st.session_state["algo_counts"]:
        algo_counts = st.session_state["algo_counts"]

        st.subheader("🏆 Algorithm Selection Frequency")

        best = max(algo_counts, key=algo_counts.get)
        st.info(f"Most preferred algorithm: **{best}**")

        fig, ax = plt.subplots(figsize=(7, 3))
        bars = ax.bar(
            algo_counts.keys(),
            algo_counts.values(),
            color=[ALGO_COLORS[a] for a in algo_counts],
        )
        ax.bar_label(bars, padding=3)
        ax.set_ylabel("Selection count")
        ax.set_title("How often each algorithm was chosen by the RL agent")
        st.pyplot(fig)
        plt.close(fig)

        # Q-table summary
        with st.expander("🔍 Q-Table snapshot (current state)"):
            state    = sim.get_state()
            q_values = agent.get_q_values(state)
            df_q     = pd.DataFrame(
                list(q_values.items()), columns=["Algorithm", "Q-Value"]
            ).sort_values("Q-Value", ascending=False)
            st.dataframe(df_q, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: Simulation
# ══════════════════════════════════════════════════════════════

elif page == "Simulation":
    st.header("⚙️ Run Simulation")

    algo = st.selectbox("Select Scheduling Algorithm", ALGOS)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶ Run Selected Algorithm", use_container_width=True):
            with st.spinner(f"Running {algo}…"):
                (timeline, deadlock,
                 avg_wait, avg_turn,
                 cpu_util, throughput,
                 per_process) = sim.run_with_algorithm(algo)

                st.session_state["timeline"]    = timeline
                st.session_state["metrics"]     = (avg_wait, avg_turn,
                                                    cpu_util, throughput,
                                                    deadlock)
                st.session_state["per_process"] = per_process

    with col2:
        if st.button("🤖 Suggest Best Algorithm (RL)", use_container_width=True):
            state     = sim.get_state()
            suggested = agent.choose_algorithm(state)
            st.success(f"RL recommends: **{suggested}**")

    # ── Metrics ────────────────────────────────────────────────
    if st.session_state["metrics"] is not None:
        avg_wait, avg_turn, cpu_util, throughput, deadlock = \
            st.session_state["metrics"]

        st.divider()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Avg Waiting Time",    f"{avg_wait:.2f}")
        c2.metric("Avg Turnaround Time", f"{avg_turn:.2f}")
        c3.metric("CPU Utilisation",     f"{cpu_util*100:.1f}%")
        c4.metric("Throughput",          f"{throughput:.4f}")
        c5.metric("Deadlock",            "⚠️ Yes" if deadlock else "✅ No")

    # ── Per-process table ──────────────────────────────────────
    if st.session_state["per_process"]:
        st.subheader("Per-Process Breakdown")
        df_pp = pd.DataFrame(st.session_state["per_process"])
        st.dataframe(df_pp, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE: Analysis
# ══════════════════════════════════════════════════════════════

elif page == "Analysis":
    st.header("📈 Performance Analysis")

    if st.session_state["timeline"] is None:
        st.warning("▶ Run a simulation first (go to the Simulation page).")
        st.stop()

    timeline = st.session_state["timeline"]

    # ── Gantt chart ────────────────────────────────────────────
    st.subheader("Gantt Chart — Process Execution Timeline")

    # Filter out None (idle) slots for the legend
    active_pids = sorted({pid for pid in timeline if pid is not None})

    if active_pids:
        cmap   = plt.cm.get_cmap("tab20", max(len(active_pids), 1))
        colors = {pid: cmap(i) for i, pid in enumerate(active_pids)}

        fig, ax = plt.subplots(figsize=(16, 2.5))

        # Merge consecutive same-pid slots into single bars for readability
        t     = 0
        start = 0
        current_pid = timeline[0]

        def _flush_bar(ax, pid, bar_start, bar_end, colors):
            width = bar_end - bar_start
            if pid is None:
                ax.barh(0, width, left=bar_start,
                        color="#e0e0e0", edgecolor="white", linewidth=0.3)
            else:
                ax.barh(0, width, left=bar_start,
                        color=colors[pid], edgecolor="white", linewidth=0.3)
                if width >= 2:
                    ax.text(bar_start + width / 2, 0, f"P{pid}",
                            ha="center", va="center", fontsize=7,
                            color="white", fontweight="bold")

        for t, pid in enumerate(timeline):
            if pid != current_pid:
                _flush_bar(ax, current_pid, start, t, colors)
                start       = t
                current_pid = pid
        _flush_bar(ax, current_pid, start, len(timeline), colors)

        ax.set_yticks([])
        ax.set_xlabel("Time units")
        ax.set_title("CPU Execution Timeline  (grey = idle)")
        ax.set_xlim(0, len(timeline))

        legend_handles = [
            mpatches.Patch(color=colors[p], label=f"P{p}")
            for p in active_pids[:20]     # cap legend to 20 entries
        ]
        ax.legend(handles=legend_handles,
                  bbox_to_anchor=(1.01, 1), loc="upper left",
                  fontsize=7, frameon=False)

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Timeline is empty.")

    # ── CPU utilisation ────────────────────────────────────────
    st.subheader("CPU Utilisation")

    busy    = sum(1 for p in timeline if p is not None)
    total   = len(timeline)
    util_pc = (busy / total * 100) if total > 0 else 0

    col_u1, col_u2 = st.columns(2)
    col_u1.metric("CPU Utilisation", f"{util_pc:.1f}%")
    col_u2.metric("Idle Time",       f"{100 - util_pc:.1f}%")

    # ── Algorithm comparison ───────────────────────────────────
    st.subheader("Algorithm Comparison")

    with st.spinner("Running all algorithms for comparison…"):
        comparison = sim.compare_all_algorithms()

    metrics_labels = {
        "avg_wait"  : "Avg Waiting Time",
        "avg_turn"  : "Avg Turnaround Time",
        "cpu_util"  : "CPU Utilisation",
        "throughput": "Throughput",
    }

    # Summary table
    df_cmp = pd.DataFrame(comparison).T.reset_index()
    df_cmp.columns = ["Algorithm", "Avg Wait", "Avg Turnaround",
                      "CPU Util", "Throughput"]
    st.dataframe(df_cmp, use_container_width=True)

    # Bar charts for wait and turnaround
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, (key, label) in zip(
        axes,
        [("avg_wait", "Avg Waiting Time"),
         ("avg_turn", "Avg Turnaround Time")]
    ):
        values = [comparison[a][key] for a in ALGOS]
        bars   = ax.bar(ALGOS, values,
                        color=[ALGO_COLORS[a] for a in ALGOS])
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set_title(label)
        ax.set_ylabel("Time units")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    best = min(comparison, key=lambda a: comparison[a]["avg_wait"])
    st.success(f"Best algorithm by waiting time: **{best}**")


# ══════════════════════════════════════════════════════════════
# PAGE: Deadlock
# ══════════════════════════════════════════════════════════════

elif page == "Deadlock":
    st.header("🔗 Deadlock Detection")

    tab_rag, tab_banker = st.tabs(["Resource Allocation Graph", "Banker's Algorithm"])

    # ── Tab 1: RAG ─────────────────────────────────────────────
    with tab_rag:
        if st.button("Generate RAG", use_container_width=True):
            G = nx.DiGraph()

            for p in processes:
                pid = f"P{p['pid']}"
                res = str(p['resource'])          # already "R0"…"R4"
                G.add_edge(pid, res)
                G.add_edge(res, pid)

            cycles = list(nx.simple_cycles(G))

            pos = nx.spring_layout(G, seed=42)
            fig, ax = plt.subplots(figsize=(10, 7))

            # Colour nodes: processes = blue, resources = orange
            node_colors = [
                "#4C72B0" if n.startswith("P") else "#DD8452"
                for n in G.nodes()
            ]
            nx.draw(G, pos, with_labels=True, node_size=800,
                    node_color=node_colors, font_color="white",
                    font_size=8, ax=ax)

            if cycles:
                for cycle in cycles:
                    edges = [
                        (cycle[i], cycle[(i + 1) % len(cycle)])
                        for i in range(len(cycle))
                    ]
                    nx.draw_networkx_edges(
                        G, pos, edgelist=edges,
                        edge_color="red", width=2.5, ax=ax
                    )
                st.error(f"⚠️ Deadlock detected — {len(cycles)} cycle(s) found.")
                for cycle in cycles:
                    st.write("🔁 " + " → ".join(cycle))
            else:
                st.success("✅ No deadlock detected.")

            # Legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='#4C72B0', markersize=10, label='Process'),
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor='#DD8452', markersize=10, label='Resource'),
                Line2D([0], [0], color='red', linewidth=2, label='Deadlock cycle'),
            ]
            ax.legend(handles=legend_elements, loc="upper left")
            st.pyplot(fig)
            plt.close(fig)

    # ── Tab 2: Banker's Algorithm ──────────────────────────────
    with tab_banker:
        st.markdown(
            "Enter the resource matrices manually to check whether "
            "the current system is in a **safe state**."
        )

        n_proc = st.number_input("Number of processes",  2, 10, 3, step=1)
        n_res  = st.number_input("Number of resource types", 1, 5, 3, step=1)

        st.markdown("**Allocation matrix** (resources currently held)")
        alloc_df = st.data_editor(
            pd.DataFrame(
                [[0] * n_res for _ in range(n_proc)],
                columns=[f"R{j}" for j in range(n_res)],
                index=[f"P{i}" for i in range(n_proc)],
            ),
            use_container_width=True,
            key="alloc",
        )

        st.markdown("**Max demand matrix** (maximum resources each process may need)")
        max_df = st.data_editor(
            pd.DataFrame(
                [[0] * n_res for _ in range(n_proc)],
                columns=[f"R{j}" for j in range(n_res)],
                index=[f"P{i}" for i in range(n_proc)],
            ),
            use_container_width=True,
            key="maxd",
        )

        avail_input = st.text_input(
            "Available resources (comma-separated)",
            value=", ".join(["3"] * n_res),
        )

        if st.button("▶ Run Banker's Algorithm", use_container_width=True):
            try:
                from core.banker import BankersAlgorithm

                allocation = alloc_df.values.tolist()
                max_demand = max_df.values.tolist()
                available  = [int(x.strip()) for x in avail_input.split(",")]

                banker = BankersAlgorithm(allocation, max_demand, available)
                result = banker.run()

                if result.is_safe:
                    seq = " → ".join(f"P{i}" for i in result.safe_sequence)
                    st.success(f"✅ Safe state.  Safe sequence: {seq}")
                else:
                    st.error("⚠️ Unsafe state — deadlock may occur.")

                with st.expander("Step-by-step explanation"):
                    st.text(result.explanation)

                st.subheader("Need Matrix")
                need_df = pd.DataFrame(
                    result.need_matrix,
                    columns=[f"R{j}" for j in range(n_res)],
                    index=[f"P{i}" for i in range(n_proc)],
                )
                st.dataframe(need_df, use_container_width=True)

            except ValueError as e:
                st.error(f"Input error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")