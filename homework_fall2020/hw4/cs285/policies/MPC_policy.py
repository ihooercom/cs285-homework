import numpy as np

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):
    def __init__(self, env, ac_dim, dyn_models, horizon, N, **kwargs):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

    def sample_action_sequences(self, num_sequences, horizon):
        # uniformly sample trajectories and return an array of
        # dimensions (num_sequences, horizon, self.ac_dim) in the range
        # [self.low, self.high]

        random_action_sequences = np.random.uniform(low=self.low,
                                                    high=self.high,
                                                    size=(num_sequences, horizon, self.ac_dim))
        return random_action_sequences

    def get_action(self, obs):

        if self.data_statistics is None:
            # print("WARNING: performing random actions.")
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        # sample random actions (N x horizon)
        candidate_action_sequences = self.sample_action_sequences(num_sequences=self.N, horizon=self.horizon)

        # for each model in ensemble:
        predicted_sum_of_rewards_per_model = []
        for model in self.dyn_models:
            sum_of_rewards = self.calculate_sum_of_rewards(obs, candidate_action_sequences, model)
            predicted_sum_of_rewards_per_model.append(sum_of_rewards)

        # calculate mean_across_ensembles(predicted rewards)
        predicted_rewards = np.mean(predicted_sum_of_rewards_per_model, axis=0)  # [ens, N] --> N

        # pick the action sequence and return the 1st element of that sequence
        best_action_sequence = candidate_action_sequences[np.argmax(predicted_rewards)]
        action_to_take = best_action_sequence[0]
        return action_to_take[None]  # Unsqueeze the first index

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        """

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        obs = np.tile(obs, (self.N, 1))
        active_mask = np.ones(self.N)
        rewards_for_actoin_seqs = np.zeros(self.N)
        for i in range(self.horizon):
            acs = candidate_action_sequences[:, i]
            res, dones = self.env.get_reward(obs, acs)
            rewards_for_actoin_seqs += res * active_mask
            obs = model.get_prediction(obs, acs, self.data_statistics)
            active_mask *= (1.0 - dones)
        sum_of_rewards = rewards_for_actoin_seqs

        return sum_of_rewards
