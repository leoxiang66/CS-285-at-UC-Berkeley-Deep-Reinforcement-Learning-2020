class BaseEnvironment(object):
    def __init__(self, init_state):
        self.states = [init_state]

    def produce_next_state(self,action):
        raise NotImplementedError

