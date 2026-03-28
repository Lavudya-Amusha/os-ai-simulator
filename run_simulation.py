from core.rl_agent import RLSchedulerAgent
from data.load_borg_data import load_borg_processes
from core.simulator import OSSimulator
from explain.llm_explainer import explain_decision
from core.metrics import calculate_metrics


# Load dataset
processes = load_borg_processes("data/borg_traces_data.csv", limit=200)

# Initialize simulator
sim = OSSimulator(processes)

# Initialize RL agent
agent = RLSchedulerAgent()

iterations = 300


# ------------------------------
# RL TRAINING
# ------------------------------

for i in range(iterations):

    state = sim.get_state()

    action = agent.choose_algorithm(state)

    next_state, reward, order, deadlock = sim.step(action)

    agent.update_q_value(state, action, reward, next_state)

    print("Iteration:", i+1, "| Algorithm:", action, "| Reward:", round(reward,2))


# ------------------------------
# FINAL DECISION
# ------------------------------

state = sim.get_state()
algo = agent.choose_algorithm(state)

order, deadlock, avg_waiting_time, avg_turnaround_time = sim.run_with_algorithm(algo)

print("\nFinal Scheduling Decision")
print("-------------------------")
print("Chosen Scheduling Algorithm:", algo)
print("Deadlock Risk Detected:", deadlock)
print("Waiting Time:", avg_waiting_time)
print("Turnaround Time:", avg_turnaround_time)
