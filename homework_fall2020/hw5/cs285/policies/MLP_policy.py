import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(input_size=self.ob_dim,
                                           output_size=self.ac_dim,
                                           n_layers=self.n_layers,
                                           size=self.size)
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(), self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(input_size=self.ob_dim,
                                          output_size=self.ac_dim,
                                          n_layers=self.n_layers,
                                          size=self.size)
            self.logstd = nn.Parameter(torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device))
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(itertools.chain([self.logstd], self.mean_net.parameters()), self.learning_rate)

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        observation = torch.as_tensor(observation, dtype=torch.float32).to(ptu.device)
        with torch.no_grad():
            pi = self.forward(observation)
        a = pi.sample()
        return a.cpu().numpy()

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        observations = torch.as_tensor(observations, dtype=torch.float32).to(ptu.device)
        actions = torch.as_tensor(actions, dtype=torch.float32).to(ptu.device)
        if self.discrete:
            output = self.logits_na(observations)
        else:
            mu = self.mean_net(observations)
            sigma = torch.exp(self.logstd)
            noise = distributions.Normal(0, 1).sample(mu.shape).to(ptu.device)
            output = mu + sigma * noise
        loss_fn = kwargs['loss_fn']
        loss = loss_fn(output, actions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        if self.discrete:
            logits = self.logits_na(observation)
            pi = distributions.Categorical(logits=logits)
        else:
            mu = self.mean_net(observation)
            sigma = torch.exp(self.logstd)
            pi = distributions.Normal(mu, sigma)
        return pi


#####################################################
#####################################################


class MLPPolicyAC(MLPPolicy):
    # MJ: cut acs_labels_na and qvals from the signature if they are not used
    def update(self, observations, actions, adv_n=None, acs_labels_na=None, qvals=None):
        raise NotImplementedError
        # Not needed for this homework

    ####################################
    ####################################
