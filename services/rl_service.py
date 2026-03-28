class RLService:

    def __init__(self, simulator, agent):
        self.sim = simulator
        self.agent = agent

    def train(self, iterations):
        rewards = []

        for _ in range(iterations):
            state = self.sim.get_state()
            action = self.agent.choose_algorithm(state)

            next_state, reward, _, _ = self.sim.step(action)

            self.agent.update_q_value(state, action, reward, next_state)

            rewards.append(reward)

        return rewards