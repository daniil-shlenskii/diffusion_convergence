import torch as th
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily

import matplotlib.pyplot as plt

class TwoGaussians:
    def __init__(self, distance_frac=3., sigma=1.,):
        n_modes = 2
        distance = distance_frac * sigma

        sigmas = th.tensor([sigma] * n_modes)
        means = th.tensor([i * distance for i in range(n_modes)])

        mix = Categorical(th.ones(n_modes,))
        comp = Normal(means, sigmas)
        self.gmm = MixtureSameFamily(mix, comp)

    def sample(self, n_samples, seed=None):
        if seed is not None:
            th.manual_seed(seed)
        return self.gmm.sample_n(n_samples)