class FeedForward:
    def __init__(self, model, trajectory):
        self.model = model

        self.trajectory = trajectory

    def initialize(self):
        pass

    def execute(self, state, input, index):
        status = True

        if index >= len(self.trajectory.t):
            return False, [0.0, 0.0]

        u = self.trajectory.u[:, index]

        return status, u
