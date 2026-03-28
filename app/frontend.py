"""
frontend.py — Fully Connected Edition
Every page shares the same state. The RL agent, simulation results,
and Banker's matrices all come from the same dataset and same session.
Claude AI explains decisions using real data from the current session.
"""
import sys
import os

# 👇 ADD THIS FIRST (before custom imports)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import io, json, logging, urllib.request
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx
import pandas as pd
import streamlit as st

# 👇 Your custom modules
from core.simulator      import OSSimulator
from core.rl_agent       import RLSchedulerAgent
from data.load_borg_data import load_borg_processes

# ── Logging ────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler("logs/simulator.log"), logging.StreamHandler()],
)
logger = logging.getLogger("frontend")

# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="AI OS Scheduler — Predictive Deadlock Management",
                   page_icon="🖥️", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}
:root{--bg:#0d1117;--surface:#161b22;--border:#30363d;--accent:#58a6ff;--accent2:#3fb950;--warn:#d29922;--danger:#f85149;--text:#e6edf3;--muted:#8b949e}
.stApp{background:var(--bg);color:var(--text)}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)}
[data-testid="stSidebar"] *{color:var(--text)!important}
[data-testid="stMetric"]{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1rem 1.25rem;transition:border-color .2s}
[data-testid="stMetric"]:hover{border-color:var(--accent)}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:.75rem;letter-spacing:.06em;text-transform:uppercase}
[data-testid="stMetricValue"]{color:var(--text)!important;font-size:1.55rem;font-weight:600;font-family:'JetBrains Mono',monospace}
.stButton>button{background:transparent;border:1px solid var(--accent);color:var(--accent);border-radius:6px;font-weight:500;transition:background .15s,color .15s}
.stButton>button:hover{background:var(--accent);color:#0d1117}
[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:8px;overflow:hidden}
[data-testid="stTabs"] button{color:var(--muted);border-bottom:2px solid transparent;font-size:.9rem;font-weight:500}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--accent)!important;border-bottom-color:var(--accent)!important}
h1{font-size:1.55rem!important;font-weight:600!important;color:var(--text)!important;letter-spacing:-.02em}
[data-testid="stExpander"]{border:1px solid var(--border)!important;border-radius:8px!important;background:var(--surface)!important}
hr{border-color:var(--border)!important;margin:1.5rem 0}
[data-testid="stProgressBar"]>div>div{background:var(--accent)!important}
.ph{display:flex;align-items:center;gap:.75rem;padding:.4rem 0 1.4rem;border-bottom:1px solid var(--border);margin-bottom:1.5rem}
.ph .icon{font-size:1.3rem;width:2.4rem;height:2.4rem;display:flex;align-items:center;justify-content:center;background:var(--surface);border:1px solid var(--border);border-radius:8px}
.ph h1{margin:0!important;padding:0!important}
.ph small{display:block;color:var(--muted);font-size:.82rem;margin-top:1px}
.badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.72rem;font-weight:600;font-family:'JetBrains Mono',monospace;letter-spacing:.03em}
.bb{background:#1f3a5f;color:#58a6ff}.bg{background:#1a3a2a;color:#3fb950}.br{background:#3d1a1a;color:#f85149}.bw{background:#3d2e0f;color:#d29922}
.conn-box{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px 14px;font-size:12px;color:var(--muted);margin-bottom:8px}
.conn-box b{color:var(--text)}
.safe-box{background:#1a3a2a;border:1px solid #3fb950;border-radius:8px;padding:.75rem 1rem;font-family:monospace;font-size:.9rem;color:#3fb950;margin:.5rem 0}
</style>""", unsafe_allow_html=True)

# ── Matplotlib dark theme ──────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":"#161b22","axes.facecolor":"#161b22",
    "axes.edgecolor":"#30363d","axes.labelcolor":"#8b949e",
    "axes.titlecolor":"#e6edf3","axes.titlesize":11,"axes.labelsize":9,
    "axes.grid":True,"grid.color":"#21262d","grid.linewidth":.6,
    "xtick.color":"#8b949e","ytick.color":"#8b949e",
    "xtick.labelsize":8,"ytick.labelsize":8,
    "text.color":"#e6edf3","legend.facecolor":"#161b22",
    "legend.edgecolor":"#30363d","legend.fontsize":8,
    "font.family":"monospace","savefig.facecolor":"#161b22","figure.dpi":130,
})

ALGO_COLORS = {"FCFS":"#4C72B0","SJF":"#DD8452","RR":"#55A868","MLFQ":"#C44E52","Priority":"#8172B2"}
ALGOS       = list(ALGO_COLORS.keys())
ALGO_LABELS = {"FCFS":"First-Come First-Served","SJF":"Shortest Job First",
               "RR":"Round Robin (q=3)","MLFQ":"Multi-Level Feedback Queue",
               "Priority":"Preemptive Priority"}

# ══════════════════════════════════════════════════════════════
# SESSION STATE — single source of truth for the whole app
# ══════════════════════════════════════════════════════════════
_DEFAULTS = {
    # Data
    "processes"        : None,
    "custom_processes" : [],
    # Simulator (shared across ALL pages)
    "sim"              : None,
    # RL Agent (persisted across pages — training is NOT lost on page switch)
    "agent"            : None,
    "rewards"          : [],
    "algo_counts"      : {},
    # Last simulation result (shared with Dashboard, Analysis, Explainer)
    "timeline"         : None,
    "last_algo"        : None,          # ← which algo was actually run
    "last_metrics"     : None,          # (avg_wait, avg_turn, cpu_util, tp, deadlock)
    "per_process"      : None,
    "last_comparison"  : None,          # ← filled by Analysis, used by Explainer
    # Banker's (filled from dataset, not random)
    "banker_alloc"     : None,
    "banker_max"       : None,
    "banker_avail"     : None,
    "banker_n_proc"    : None,
    "banker_n_res"     : None,
    # Explainer
    "last_explanation" : None,
    "expl_algo"        : None,          # which algo was explained
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")
    data_limit = st.slider("Dataset Size",  50, 500, 200, step=50)
    iterations = st.slider("RL Iterations", 10, 300, 100, step=10)

    with st.expander("🧠 RL Hyperparameters"):
        hp_alpha   = st.slider("α  Learning rate",   0.01, 1.0,  0.10, step=0.01)
        hp_gamma   = st.slider("γ  Discount factor", 0.50, 1.0,  0.90, step=0.01)
        hp_eps_dec = st.slider("ε  Decay rate",      0.90, 1.0,  0.995,step=0.001,format="%.3f")
        hp_eps_min = st.slider("ε  Minimum",         0.01, 0.20, 0.05, step=0.01)

    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠  Dashboard","🤖  RL Training","⚙️  Simulation",
        "📈  Analysis","🔗  Deadlock","💡  Explainer","📤  Export","ℹ️  About",
    ], label_visibility="collapsed")

    st.markdown("---")
    if st.sidebar.button("📂  Load Dataset", use_container_width=True):
        with st.spinner("Parsing Borg traces…"):
            try:
                procs = load_borg_processes("data/borg_traces_data.csv", limit=data_limit)
                # Reset everything when new dataset loaded
                for k in _DEFAULTS:
                    st.session_state[k] = _DEFAULTS[k]
                st.session_state["processes"] = procs
                # Create simulator and agent fresh
                st.session_state["sim"]   = OSSimulator(procs)
                st.session_state["agent"] = RLSchedulerAgent(
                    alpha=hp_alpha, gamma=hp_gamma,
                    epsilon_decay=hp_eps_dec, epsilon_min=hp_eps_min)
                st.sidebar.success(f"✓ {len(procs)} processes loaded")
                logger.info("Dataset loaded: %d processes", len(procs))
            except FileNotFoundError:
                st.sidebar.error("data/borg_traces_data.csv not found")
            except Exception as e:
                st.sidebar.error(f"Failed: {e}")
                logger.exception("Load failed")

    n = len(st.session_state["processes"]) if st.session_state["processes"] else 0
    if n:
        st.markdown(f'<span class="badge bg">● {n} processes ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge br">● no data loaded</span>', unsafe_allow_html=True)

    # Show connectivity status
    st.markdown("---")
    st.markdown("**Session Status**")
    trained = bool(st.session_state["rewards"])
    simulated = st.session_state["last_algo"] is not None
    st.markdown(
        f'<div class="conn-box">'
        f'Dataset: <b>{"✅ " + str(n) + " procs" if n else "❌ not loaded"}</b><br>'
        f'RL trained: <b>{"✅ " + str(len(st.session_state["rewards"])) + " episodes" if trained else "❌ not yet"}</b><br>'
        f'Simulation: <b>{"✅ " + str(st.session_state["last_algo"]) if simulated else "❌ not run"}</b><br>'
        f'</div>',
        unsafe_allow_html=True)

# ── Guard ──────────────────────────────────────────────────────
if st.session_state["processes"] is None:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5rem 2rem;text-align:center">
      <div style="font-size:3rem;margin-bottom:1rem">🖥️</div>
      <h2 style="color:#e6edf3;margin-bottom:.5rem">AI-Based OS Scheduling Simulator</h2>
      <p style="color:#8b949e;max-width:420px;line-height:1.6">
        Click <b>Load Dataset</b> in the sidebar to begin.
        All pages share the same data and the same trained RL agent.
      </p>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Shared objects from session state ─────────────────────────
processes = st.session_state["processes"] + st.session_state["custom_processes"]
sim       = st.session_state["sim"]
agent     = st.session_state["agent"]

if sim is None:
    sim = OSSimulator(processes)
    st.session_state["sim"] = sim
if agent is None:
    agent = RLSchedulerAgent(alpha=hp_alpha, gamma=hp_gamma,
                              epsilon_decay=hp_eps_dec, epsilon_min=hp_eps_min)
    st.session_state["agent"] = agent

_page = page.strip().split("  ", 1)[-1]

def _state_key_display(state):
    """Convert continuous state to discretised key string for display."""
    n,b,w,d=state
    pb=0 if n<5 else 1 if n<15 else 2
    bb=0 if b<5 else 1 if b<15 else 2
    wb=0 if w<10 else 1 if w<30 else 2
    return f"({pb},{bb},{wb},{int(bool(d))})"

def ph(icon, title, sub=""):
    sub_html = f"<small>{sub}</small>" if sub else ""
    st.markdown(
        f'<div class="ph"><div class="icon">{icon}</div>'
        f'<div><h1>{title}</h1>{sub_html}</div></div>',
        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# DASHBOARD — shows REAL state from last simulation
# ══════════════════════════════════════════════════════════════
if _page == "Dashboard":
    ph("🏠", "System Overview",
       "All metrics reflect your actual dataset and last simulation run")

    state = sim.get_state()   # avg_wait now comes from last sim run
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Processes", state[0])
    c2.metric("Avg Burst Time",  f"{state[1]:.1f}")
    c3.metric("Avg Wait Time",
              f"{state[2]:.2f}" if state[2] > 0 else "Run simulation first")
    c4.metric("Deadlock Risk",   "⚠️ Yes" if state[3] else "✅ No")

    # Show which algo was last run
    if st.session_state["last_algo"]:
        m = st.session_state["last_metrics"]
        st.info(
            f"Last simulation: **{st.session_state['last_algo']}** — "
            f"avg_wait={m[0]:.2f}, avg_turn={m[1]:.2f}, "
            f"cpu_util={m[2]*100:.1f}%, deadlock={'Yes' if m[4] else 'No'}"
        )
    else:
        st.info("👉 Go to **Simulation** page and run an algorithm to see real metrics here.")

    st.markdown("---")
    col_l, col_r = st.columns([1,2])
    with col_l:
        st.markdown("#### Workflow")
        for i,(icon,lbl) in enumerate([
            ("📂","Load Dataset"),("🤖","Train RL Agent"),
            ("⚙️","Run Simulation"),("📈","Analyse Results"),
            ("🔗","Check Deadlock"),("💡","Explain Decision"),("📤","Export"),
        ],1):
            trained_mark = " ✅" if (
                (i==2 and trained) or (i==3 and simulated) or
                (i==4 and st.session_state["last_comparison"] is not None) or
                (i==6 and st.session_state["last_explanation"])
            ) else ""
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:.6rem;padding:.35rem 0;border-bottom:1px solid #21262d">'
                f'<span class="badge bb">{i}</span>'
                f'<span style="font-size:.88rem">{icon} {lbl}{trained_mark}</span></div>',
                unsafe_allow_html=True)
    with col_r:
        st.markdown("#### Dataset Preview")
        st.dataframe(pd.DataFrame(processes[:8]), use_container_width=True, height=270)

    with st.expander("➕ Add a custom process"):
        cc1,cc2,cc3,cc4 = st.columns(4)
        c_pid     = cc1.number_input("PID",      0,9999,len(processes))
        c_arrival = cc2.number_input("Arrival",  0, 200,0)
        c_burst   = cc3.number_input("Burst",    1, 100,5)
        c_prio    = cc4.number_input("Priority", 0, 500,1)
        if st.button("Add Process"):
            st.session_state["custom_processes"].append({
                'pid':int(c_pid),'arrival':int(c_arrival),'burst':int(c_burst),
                'priority':int(c_prio),'resource':f"R{int(c_pid)%5}",'wanted_resource':""})
            # Rebuild simulator with new process list
            all_p = st.session_state["processes"] + st.session_state["custom_processes"]
            st.session_state["sim"] = OSSimulator(all_p)
            st.success(f"P{c_pid} added. Simulator rebuilt.")
            st.rerun()

# ══════════════════════════════════════════════════════════════
# RL TRAINING — uses shared sim and saves agent to session
# ══════════════════════════════════════════════════════════════
elif _page == "RL Training":
    ph("🤖", "RL Agent Training",
       "Training uses your loaded dataset. Results persist across all pages.")

    ca,cb,cc = st.columns([3,1,1])
    train_btn = ca.button("▶  Train Agent", use_container_width=True)
    save_btn  = cb.button("💾  Save",        use_container_width=True)
    if cc.button("🔄  Reset",               use_container_width=True):
        st.session_state["agent"] = RLSchedulerAgent(
            alpha=hp_alpha,gamma=hp_gamma,
            epsilon_decay=hp_eps_dec,epsilon_min=hp_eps_min)
        st.session_state.update({"rewards":[],"algo_counts":{}})
        agent = st.session_state["agent"]
        st.success("Agent reset.")

    if save_btn:
        agent.save("logs/q_table.json")
        st.success("Saved → logs/q_table.json")

    st.markdown(
        f'<div style="display:flex;gap:.5rem;flex-wrap:wrap;margin:.5rem 0 1rem">'
        f'<span class="badge bb">α={hp_alpha}</span>'
        f'<span class="badge bb">γ={hp_gamma}</span>'
        f'<span class="badge bb">ε-decay={hp_eps_dec}</span>'
        f'<span class="badge bb">ε-min={hp_eps_min}</span>'
        f'</div>', unsafe_allow_html=True)

    if train_btn:
        # Accumulate — don't reset counts from previous training runs
        rewards = list(st.session_state.get("rewards", []))
        counts  = dict(st.session_state.get("algo_counts", {a:0 for a in ALGOS}))
        bar = st.progress(0, text="Starting…")
        for i in range(iterations):
            s  = sim.get_state()          # ← uses real avg_wait from last run
            a  = agent.choose_algorithm(s)
            counts[a] = counts.get(a,0)+1
            ns,r,_,_ = sim.step(a)        # ← runs scheduler, updates sim state
            agent.update_q_value(s,a,r,ns)
            agent.decay_epsilon()
            rewards.append(r)
            bar.progress((i+1)/iterations,
                text=f"Episode {i+1}/{iterations}  ε={agent.epsilon:.4f}  r={r:.2f}")
        bar.empty()
        st.session_state.update({
            "rewards":rewards,"algo_counts":counts,"agent":agent})
        st.success(
            f"Done — best: **{agent.best_action_overall}** | "
            f"ε: **{agent.epsilon:.4f}** | "
            f"mean reward: **{sum(rewards)/len(rewards):.3f}**")
        logger.info("Training done. best=%s", agent.best_action_overall)

    if st.session_state["rewards"]:
        rewards = st.session_state["rewards"]
        mean_r  = sum(rewards)/len(rewards)
        st.markdown("#### Learning Curve")
        fig,ax = plt.subplots(figsize=(11,3))
        ax.plot(rewards,color="#58a6ff",lw=.8,alpha=.6,label="Reward")
        if len(rewards)>=10:
            ax.plot(pd.Series(rewards).rolling(10).mean(),
                    color="#3fb950",lw=1.4,label="Rolling avg (10)")
        ax.axhline(mean_r,color="#d29922",ls="--",lw=1,
                   label=f"Mean={mean_r:.3f}")
        ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
        ax.legend(loc="lower right")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    if st.session_state["algo_counts"]:
        counts = st.session_state["algo_counts"]
        st.markdown("#### Algorithm Selection Frequency")
        cc1,cc2 = st.columns([2,1])
        with cc1:
            fig,ax = plt.subplots(figsize=(7,3))
            bars = ax.bar(counts.keys(),counts.values(),
                          color=[ALGO_COLORS.get(a,"#888") for a in counts],
                          width=.55,zorder=3)
            ax.bar_label(bars,padding=4,fontsize=9,color="#e6edf3")
            ax.set_ylabel("Times selected")
            ax.set_ylim(0,max(counts.values())*1.2)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        with cc2:
            st.markdown("<br>",unsafe_allow_html=True)
            best       = max(counts,key=counts.get)
            total_c    = sum(counts.values())
            for algo,cnt in sorted(counts.items(),key=lambda x:x[1],reverse=True):
                pct = cnt/total_c*100
                col = "#58a6ff" if algo==best else "#8b949e"
                tag = " ✓" if algo==best else ""
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;padding:.3rem 0;'
                    f'border-bottom:1px solid #21262d;font-size:.85rem">'
                    f'<span style="color:{col}">{algo}{tag}</span>'
                    f'<span style="font-family:monospace">{pct:.0f}%</span></div>',
                    unsafe_allow_html=True)

        with st.expander("🔍 Q-Table (current state)"):
            qv   = agent.get_q_values(sim.get_state())
            df_q = (pd.DataFrame(list(qv.items()),columns=["Algorithm","Q-Value"])
                    .sort_values("Q-Value",ascending=False).reset_index(drop=True))
            st.dataframe(df_q,use_container_width=True)
            st.caption(f"State key: {_state_key_display(sim.get_state())}")

# ══════════════════════════════════════════════════════════════
# SIMULATION — stores results for Dashboard, Analysis, Explainer
# ══════════════════════════════════════════════════════════════
elif _page == "Simulation":
    ph("⚙️", "Run Simulation",
       "Results are shared with Dashboard, Analysis, and Explainer pages")

    algo = st.selectbox("Algorithm", ALGOS,
                        format_func=lambda a:f"{a}  —  {ALGO_LABELS[a]}")

    sc1,sc2 = st.columns(2)
    run_btn = sc1.button("▶  Run Algorithm", use_container_width=True)
    rl_btn  = sc2.button("🤖  Ask RL Agent",  use_container_width=True)

    if rl_btn:
        state     = sim.get_state()
        suggested = agent.choose_algorithm(state)
        qv        = agent.get_q_values(state)
        st.info(
            f"RL agent recommends: **{suggested}** — {ALGO_LABELS[suggested]}\n\n"
            f"Q-value for {suggested}: **{qv.get(suggested,0):.4f}** "
            f"(go to Explainer page to understand why)"
        )

    if run_btn:
        with st.spinner(f"Running {algo}…"):
            try:
                tl,dl,aw,at,cu,tp,pp = sim.run_with_algorithm(algo)
                # ← Store everything in session state for ALL pages
                st.session_state.update({
                    "timeline"      : tl,
                    "last_algo"     : algo,
                    "last_metrics"  : (aw,at,cu,tp,dl),
                    "per_process"   : pp,
                    "last_comparison": None,  # reset comparison — needs re-run
                    "last_explanation": None, # reset explanation — data changed
                })
                logger.info("Ran %s: aw=%.2f at=%.2f", algo, aw, at)
                st.success(
                    f"✅ {algo} complete. "
                    f"Dashboard and Analysis pages are now updated with these results."
                )
            except Exception as e:
                st.error(f"Simulation failed: {e}")

    if st.session_state["last_metrics"] is not None:
        aw,at,cu,tp,dl = st.session_state["last_metrics"]
        st.markdown(f"#### Results — {st.session_state['last_algo']}")
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Avg Wait",       f"{aw:.2f}")
        m2.metric("Avg Turnaround", f"{at:.2f}")
        m3.metric("CPU Util",       f"{cu*100:.1f}%")
        m4.metric("Throughput",     f"{tp:.4f}")
        m5.metric("Deadlock",       "⚠️ Yes" if dl else "✅ No")

    if st.session_state["per_process"]:
        st.markdown("#### Per-Process Breakdown")
        df_pp = pd.DataFrame(st.session_state["per_process"])
        def _wt(v):
            if v>df_pp["waiting"].quantile(.75): return "color:#f85149"
            if v<df_pp["waiting"].quantile(.25): return "color:#3fb950"
            return "color:#e6edf3"
        starved = df_pp[df_pp["starved"]==True]
        if not starved.empty:
            st.warning(
                f"⚠️ Starvation in {len(starved)} process(es): "
                + ", ".join(f"P{p}" for p in starved['pid'].tolist()[:10]))
        st.dataframe(df_pp.style.map(_wt,subset=["waiting"]),
                     use_container_width=True,height=300)

# ══════════════════════════════════════════════════════════════
# ANALYSIS — reads from shared simulation results
# ══════════════════════════════════════════════════════════════
elif _page == "Analysis":
    ph("📈","Performance Analysis",
       f"Showing results for: {st.session_state['last_algo'] or 'no simulation run yet'}")

    if st.session_state["timeline"] is None:
        st.warning("▶ Run a simulation first (Simulation page).")
        st.stop()

    timeline    = st.session_state["timeline"]
    active_pids = sorted({p for p in timeline if p is not None})

    st.markdown(f"#### Gantt Chart — {st.session_state['last_algo']}")
    if active_pids:
        cmap   = plt.cm.get_cmap("tab20",max(len(active_pids),1))
        colors = {pid:cmap(i) for i,pid in enumerate(active_pids)}
        fig,ax = plt.subplots(figsize=(16,2))
        segs,cur_pid,cur_s = [],timeline[0],0
        for t,pid in enumerate(timeline):
            if pid!=cur_pid:
                segs.append((cur_pid,cur_s,t)); cur_pid,cur_s=pid,t
        segs.append((cur_pid,cur_s,len(timeline)))
        for pid,s,e in segs:
            w=e-s
            if pid is None:
                ax.barh(0,w,left=s,color="#21262d",edgecolor="#30363d",linewidth=.3)
            else:
                ax.barh(0,w,left=s,color=colors[pid],edgecolor="#0d1117",linewidth=.3)
                if w>=3:
                    ax.text(s+w/2,0,f"P{pid}",ha="center",va="center",
                            fontsize=6.5,color="white",fontweight="bold")
        ax.set_yticks([]); ax.set_xlabel("Time units"); ax.set_xlim(0,len(timeline))
        handles=[mpatches.Patch(color=colors[p],label=f"P{p}") for p in active_pids[:24]]
        ax.legend(handles=handles,bbox_to_anchor=(1.01,1),
                  loc="upper left",fontsize=7,frameon=True,ncol=2)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    busy    = sum(1 for p in timeline if p is not None)
    total_t = len(timeline)
    uc1,uc2,uc3 = st.columns(3)
    uc1.metric("CPU Utilisation",f"{busy/total_t*100:.1f}%" if total_t else "0%")
    uc2.metric("Busy Slots",busy); uc3.metric("Idle Slots",total_t-busy)

    st.markdown("---")
    st.markdown("#### Algorithm Comparison")
    with st.spinner("Running all algorithms for comparison…"):
        cmp = sim.compare_all_algorithms()
        # ← Save comparison for Explainer page
        st.session_state["last_comparison"] = cmp

    df_cmp = (pd.DataFrame(cmp).T.reset_index()
              .rename(columns={"index":"Algorithm","avg_wait":"Avg Wait",
                               "avg_turn":"Avg Turnaround",
                               "cpu_util":"CPU Util","throughput":"Throughput"}))
    st.dataframe(df_cmp,use_container_width=True)
    fig,axes = plt.subplots(1,2,figsize=(12,3.5))
    for ax,(key,lbl) in zip(axes,[("avg_wait","Avg Waiting Time (lower=better)"),
                                    ("avg_turn","Avg Turnaround (lower=better)")]):
        vals=[cmp[a][key] for a in ALGOS]
        bars=ax.bar(ALGOS,vals,color=[ALGO_COLORS[a] for a in ALGOS],width=.55,zorder=3)
        ax.bar_label(bars,fmt="%.2f",padding=4,fontsize=8.5,color="#e6edf3")
        ax.set_title(lbl); ax.set_ylabel("Time units")
        ax.set_ylim(0,max(vals)*1.2 if max(vals)>0 else 1)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    bw=min(cmp,key=lambda a:cmp[a]["avg_wait"])
    bc=max(cmp,key=lambda a:cmp[a]["cpu_util"])
    st.success(f"Lowest wait: **{bw}** | Highest CPU util: **{bc}**")
    st.info("💡 Go to **Explainer** page — Claude AI will explain why the RL agent chose its algorithm using these results.")

# ══════════════════════════════════════════════════════════════
# DEADLOCK — Banker's matrices from dataset
# ══════════════════════════════════════════════════════════════
elif _page == "Deadlock":
    ph("🔗","Deadlock Detection",
       "RAG built from your dataset. Banker's matrices auto-filled from simulation data.")

    tab_rag,tab_banker = st.tabs(["  Resource Allocation Graph  ","  Banker's Algorithm  "])

    with tab_rag:
        st.markdown("Cycles = **circular wait** = deadlock present.")
        rag_limit = st.slider("Processes to show in RAG", 10, min(len(processes),50), 30, step=5)
        st.caption(f"Showing {rag_limit} of {len(processes)} processes for readability.")
        if st.button("Generate RAG",use_container_width=True):
            try:
                from core.rag_deadlock import build_rag_from_processes
                rag = build_rag_from_processes(processes[:rag_limit])
                G   = nx.DiGraph()
                for node,nbrs in rag.graph.items():
                    for nb in nbrs:
                        src=f"P{node}" if isinstance(node,int) else str(node)
                        dst=f"P{nb}"   if isinstance(nb,int)   else str(nb)
                        G.add_edge(src,dst)
                cycles=list(nx.simple_cycles(G))
                pos=nx.spring_layout(G,seed=42,k=1.5)
                fig,ax=plt.subplots(figsize=(10,7))
                pnodes=[n for n in G.nodes() if n.startswith("P")]
                rnodes=[n for n in G.nodes() if not n.startswith("P")]
                nx.draw_networkx_nodes(G,pos,nodelist=pnodes,node_color="#1f3a5f",node_size=900,ax=ax)
                nx.draw_networkx_nodes(G,pos,nodelist=rnodes,node_color="#3d2a0f",node_size=700,node_shape="s",ax=ax)
                nx.draw_networkx_labels(G,pos,font_color="#e6edf3",font_size=8,ax=ax)
                nx.draw_networkx_edges(G,pos,edge_color="#30363d",arrows=True,arrowsize=12,width=1,ax=ax)
                if cycles:
                    cedges=[]
                    for c in cycles:
                        cedges+=[(c[i],c[(i+1)%len(c)]) for i in range(len(c))]
                    nx.draw_networkx_edges(G,pos,edgelist=cedges,
                                           edge_color="#f85149",width=2.5,arrows=True,arrowsize=14,ax=ax)
                    st.error(f"⚠️ {len(cycles)} deadlock cycle(s) detected in your dataset.")
                    for c in cycles:
                        st.code(" → ".join(c)+" → "+c[0])
                else:
                    st.success("✅ No deadlock in your dataset.")
                ax.legend(handles=[
                    Line2D([0],[0],marker='o',color='w',markerfacecolor='#1f3a5f',markersize=11,label='Process'),
                    Line2D([0],[0],marker='s',color='w',markerfacecolor='#3d2a0f',markersize=11,label='Resource'),
                    Line2D([0],[0],color='#f85149',lw=2,label='Deadlock cycle'),
                ],loc="upper left",framealpha=.9)
                ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117"); ax.axis("off")
                st.pyplot(fig); plt.close(fig)
            except Exception as e:
                st.error(f"RAG error: {e}")

    with tab_banker:
        st.markdown("Matrices are **derived from your loaded dataset** (same data as the simulation).")

        def _build_banker(procs,n_res=5):
            import math
            ri  = {f"R{j}":j for j in range(n_res)}
            n   = min(len(procs),10)
            alloc=[[ 0]*n_res for _ in range(n)]
            maxd =[[ 0]*n_res for _ in range(n)]
            mb   = max((p["burst"] for p in procs),default=1)
            for i,p in enumerate(procs[:n]):
                j=ri.get(str(p["resource"]),0)
                alloc[i][j]=1
                maxd[i][j]=1+max(1,math.ceil((p["burst"]/mb)*3))
            held=[sum(alloc[i][j] for i in range(n)) for j in range(n_res)]
            avail=[max(2,h) for h in held]
            return n,alloc,maxd,avail

        N_RES=5
        if st.session_state["banker_alloc"] is None or \
                st.button("🔄  Rebuild from current dataset"):
            n_p,da,dm,dav=_build_banker(processes,N_RES)
            st.session_state.update({
                "banker_n_proc":n_p,"banker_alloc":da,"banker_max":dm,
                "banker_avail":dav,"banker_n_res":N_RES})

        n_proc=st.session_state["banker_n_proc"]
        n_res =st.session_state["banker_n_res"]
        cols  =[f"R{j}" for j in range(n_res)]
        idx   =[f"P{i}" for i in range(n_proc)]
        st.markdown(f'<span class="badge bb">● {n_proc} processes · {n_res} resources · from dataset</span>',
                    unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        bm1,bm2=st.columns(2)
        with bm1:
            st.markdown("**Allocation** (currently held)")
            alloc_df=st.data_editor(pd.DataFrame(st.session_state["banker_alloc"],columns=cols,index=idx),
                                    use_container_width=True,key="alloc")
        with bm2:
            st.markdown("**Max demand** (maximum needed)")
            max_df=st.data_editor(pd.DataFrame(st.session_state["banker_max"],columns=cols,index=idx),
                                  use_container_width=True,key="maxd")
        avail_input=st.text_input("Available (comma-separated)",
                                   value=", ".join(str(v) for v in st.session_state["banker_avail"]))
        if st.button("▶  Run Banker's Algorithm",use_container_width=True):
            try:
                from core.banker import BankersAlgorithm
                result=BankersAlgorithm(
                    alloc_df.values.tolist(),max_df.values.tolist(),
                    [int(x.strip()) for x in avail_input.split(",")]).run()
                if result.is_safe:
                    seq=" → ".join(f"P{i}" for i in result.safe_sequence)
                    st.success("✅ Safe state")
                    st.markdown(f'<div class="safe-box">Safe sequence: {seq}</div>',unsafe_allow_html=True)
                else:
                    st.error("⚠️ Unsafe state — deadlock risk.")
                with st.expander("Step-by-step trace"):
                    st.code(result.explanation,language="text")
                st.markdown("**Need matrix**")
                st.dataframe(pd.DataFrame(result.need_matrix,columns=cols,index=idx),use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

# EXPLAINER — uses REAL data from current session
# ══════════════════════════════════════════════════════════════
elif _page == "Explainer":
    ph("💡","RL Decision Explainer",
       "LLM explains why the RL agent selected this scheduling algorithm")

    last_algo = st.session_state["last_algo"]
    last_m    = st.session_state["last_metrics"]
    last_cmp  = st.session_state["last_comparison"]
    state     = sim.get_state()
    q_values  = agent.get_q_values(state)

    if not last_algo:
        st.warning("❌ Run a simulation first (Simulation page), then come back here.")
        st.stop()

    # ── Status panel — shows exactly what will be used ──────────
    st.markdown("#### What the AI will explain")
    d1,d2,d3,d4 = st.columns(4)
    d1.metric("Algorithm Ran",    last_algo)
    d2.metric("Avg Wait (real)",  f"{last_m[0]:.2f}" if last_m else "—")
    d3.metric("Avg Turn (real)",  f"{last_m[1]:.2f}" if last_m else "—")
    d4.metric("Deadlock",         "Yes ⚠️" if (last_m and last_m[4]) else "No ✅")

    trained_eps = len(st.session_state["rewards"])
    col_a, col_b = st.columns(2)
    with col_a:
        if trained_eps > 0:
            st.success(f"✅ RL agent trained — {trained_eps} episodes. Q-values ready.")
        else:
            st.warning("⚠️ RL agent NOT trained. Train it first for Q-value analysis.")
    with col_b:
        if last_cmp:
            st.success("✅ Algorithm comparison available (from Analysis page)")
        else:
            st.warning("⚠️ No comparison yet — visit Analysis page first")

    # ── Q-Table ─────────────────────────────────────────────────
    if q_values and any(v!=0 for v in q_values.values()):
        st.markdown(f"#### Q-Table ({trained_eps} training episodes)")
        df_q=(pd.DataFrame(list(q_values.items()),columns=["Algorithm","Q-Value"])
              .sort_values("Q-Value",ascending=False).reset_index(drop=True))
        st.dataframe(df_q,use_container_width=True)
        st.caption(f"State key: {_state_key_display(state)}  |  ε = {agent.epsilon:.4f}")

    st.markdown("---")

    # ── API status check — shows which LLM will be used ─────────
    def _check_api_status():
        """Check which API key is available and return status info."""
        try:
            groq_key = st.secrets.get("GROQ_API_KEY","") if hasattr(st, "secrets") else ""
        except Exception:
            groq_key = ""
        groq_key = groq_key or os.environ.get("GROQ_API_KEY","")

        try:
            anth_key = st.secrets.get("ANTHROPIC_API_KEY","") if hasattr(st, "secrets") else ""
        except Exception:
            anth_key = ""
        anth_key = anth_key or os.environ.get("ANTHROPIC_API_KEY","")

        if groq_key:
            return "groq", f"Groq API (llama-3.3-70b) — key ends ...{groq_key[-6:]}"
        elif anth_key:
            return "anthropic", f"Anthropic Claude — key ends ...{anth_key[-6:]}"
        else:
            # Check Ollama
            try:
                urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
                return "ollama", "Ollama local (llama3.2)"
            except Exception:
                return "offline", "Offline analysis engine (no API key set)"

    api_mode, api_label = _check_api_status()

    # Show the AI source clearly
    if api_mode == "offline":
        st.markdown(
            f'''<div style="background:#3d2e0f;border:1px solid #d29922;border-radius:8px;
            padding:12px 16px;margin-bottom:12px">
            <b style="color:#d29922">⚠️ No API key found — will use offline analysis</b><br>
            <span style="color:#8b949e;font-size:12px">
            To get free AI explanations: create <code>.streamlit/secrets.toml</code>
            and add <code>GROQ_API_KEY = "gsk_..."</code> (free at console.groq.com)
            </span></div>''',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'''<div style="background:#1a3a2a;border:1px solid #3fb950;border-radius:8px;
            padding:12px 16px;margin-bottom:12px">
            <b style="color:#3fb950">✅ AI source: {api_label}</b><br>
            <span style="color:#8b949e;font-size:12px">
            Click Generate to get a real AI explanation of your simulation data.
            </span></div>''',
            unsafe_allow_html=True)

    # ── Single generate button ───────────────────────────────────
    if st.button("✨  Generate AI Explanation", use_container_width=True):
        with st.spinner(f"Generating via {api_label}…"):
            try:
                from explain.llm_explainer import explain_decision
                text, source = explain_decision(last_algo, state, q_values, last_cmp)
                st.session_state["last_explanation"] = text
                st.session_state["expl_algo"]        = last_algo
                st.session_state["expl_source"]      = source
            except Exception as e:
                st.error(f"Generation failed: {e}")

    # ── Show result ──────────────────────────────────────────────
    if st.session_state["last_explanation"]:
        source = st.session_state.get("expl_source","")

        if source.startswith("offline-fallback"):
            # Extract the real error reason
            error_part = source.replace("offline-fallback","").strip("() ")
            st.warning(
                f"⚠️ AI APIs not reachable — showing offline analysis.\n\n"
                f"**Error details:** {error_part}\n\n"
                f"**To fix:** Check your GROQ_API_KEY in `.streamlit/secrets.toml`"
            )
        elif "Groq" in source:
            st.success(f"✅ Generated by **{source}**")
        elif "Anthropic" in source:
            st.success(f"✅ Generated by **{source}**")
        elif "Ollama" in source:
            st.success(f"✅ Generated by **{source}** (local)")
        else:
            st.info(f"Source: {source}")

        if st.session_state.get("expl_algo") != last_algo:
            st.warning(
                f"⚠️ This explanation is for **{st.session_state['expl_algo']}**. "
                f"You have since run **{last_algo}**. Click Generate again to update.")

        st.markdown("---")
        st.markdown(st.session_state["last_explanation"])


# ══════════════════════════════════════════════════════════════
# EXPORT — uses shared session data
# ══════════════════════════════════════════════════════════════
elif _page == "Export":
    ph("📤","Export Results","Download simulation data from the current session")

    if not st.session_state["last_algo"]:
        st.warning("Run a simulation first.")
        st.stop()

    st.markdown(f"Exporting results for: **{st.session_state['last_algo']}**")
    c1,c2,c3 = st.columns(3)

    with c1:
        st.markdown("**Per-Process CSV**")
        if st.session_state["per_process"]:
            csv=pd.DataFrame(st.session_state["per_process"]).to_csv(index=False).encode()
            st.download_button("⬇  results.csv",data=csv,
                               file_name=f"results_{st.session_state['last_algo']}.csv",
                               mime="text/csv",use_container_width=True)

    with c2:
        st.markdown("**Gantt Chart PNG**")
        if st.session_state["timeline"]:
            timeline=st.session_state["timeline"]
            active_pids=sorted({p for p in timeline if p is not None})
            cmap=plt.cm.get_cmap("tab20",max(len(active_pids),1))
            colors={pid:cmap(i) for i,pid in enumerate(active_pids)}
            fig_e,ax_e=plt.subplots(figsize=(18,2.5))
            segs,cur_pid,cur_s=[],timeline[0],0
            for t,pid in enumerate(timeline):
                if pid!=cur_pid: segs.append((cur_pid,cur_s,t)); cur_pid,cur_s=pid,t
            segs.append((cur_pid,cur_s,len(timeline)))
            for pid,s,e in segs:
                w=e-s
                ax_e.barh(0,w,left=s,color="#21262d" if pid is None else colors[pid],
                          edgecolor="#0d1117",linewidth=.3)
                if pid is not None and w>=3:
                    ax_e.text(s+w/2,0,f"P{pid}",ha="center",va="center",
                              fontsize=6,color="white",fontweight="bold")
            ax_e.set_yticks([]); ax_e.set_xlabel("Time")
            ax_e.set_title(f"Gantt — {st.session_state['last_algo']}")
            fig_e.tight_layout()
            buf=io.BytesIO(); fig_e.savefig(buf,format="png",dpi=150,bbox_inches="tight")
            plt.close(fig_e)
            st.download_button("⬇  gantt.png",data=buf.getvalue(),
                               file_name=f"gantt_{st.session_state['last_algo']}.png",
                               mime="image/png",use_container_width=True)

    with c3:
        st.markdown("**Q-Table JSON**")
        q_json=json.dumps({
            "q_table"      :{str(k):v for k,v in agent.q_table.items()},
            "epsilon"      :agent.epsilon,
            "action_counts":agent.action_counts,
            "training_episodes":len(st.session_state["rewards"]),
            "last_algo_run":st.session_state["last_algo"],
        },indent=2).encode()
        st.download_button("⬇  q_table.json",data=q_json,
                           file_name="q_table.json",mime="application/json",
                           use_container_width=True)

# ══════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════
elif _page == "About":
    ph("ℹ️","About This Project")
    st.markdown("""
## AI-Based OS Scheduler with Predictive Deadlock Management using LLM

A reinforcement-learning-guided CPU scheduling simulator with LLM-powered
decision explanation and predictive deadlock management, built on real
**Google Borg cluster trace data**. All pages share the same dataset,
the same trained RL agent, and the same simulation results.

### Data Flow
1. **Load Dataset** → processes stored in session, simulator built
2. **RL Training** → agent learns on the same simulator, Q-table persists
3. **Simulation** → results stored, feed back into Dashboard and Explainer
4. **Analysis** → comparison data stored, used by Explainer
5. **Deadlock** → RAG and Banker's built from same dataset
6. **Explainer** → Claude AI uses actual Q-values and simulation metrics
7. **Export** → downloads from current session results
""")
    techs=["Python 3.11","Streamlit","Pandas","NetworkX","Matplotlib","Pytest","Claude AI"]
    st.markdown("".join(
        f'<span style="display:inline-block;background:#161b22;border:1px solid #30363d;'
        f'border-radius:6px;padding:4px 12px;font-size:.78rem;font-family:monospace;'
        f'color:#8b949e;margin:3px">{t}</span>' for t in techs),
        unsafe_allow_html=True)
    st.markdown("""
---
*B.Tech Computer Science — Operating Systems Project · RGUKT Basar · 2024–2025*
""")