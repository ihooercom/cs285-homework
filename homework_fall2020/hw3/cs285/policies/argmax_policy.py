import numpy as np


class ArgMaxPolicy(object):
    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]

        qa_values = self.critic.qa_values(observation)
        actions = np.argmax(qa_values, axis=-1)

        return actions.squeeze()