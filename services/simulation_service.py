from core.simulator import OSSimulator

class SimulationService:

    def __init__(self, processes):
        self.sim = OSSimulator(processes)

    def get_state(self):
        return self.sim.get_state()

    def run_algorithm(self, algo):
        return self.sim.run_with_algorithm(algo)