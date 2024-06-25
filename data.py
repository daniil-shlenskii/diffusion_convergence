import torch as th
from torch.distributions.normal import Normal, Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily

class GM1d:
    def __init__(self, n_modes=1, width=6):
        sub_width = width / n_modes
        sigmas = th.tensor([sub_width / 3] * n_modes)
        means = th.linspace(-width/2 + sub_width/2, width/2 - sub_width/2, steps=n_modes)

        mix = Categorical(th.ones(n_modes,))
        comp = Normal(means, sigmas)
        gmm = MixtureSameFamily(mix, comp)

    def sample(self, n_samples):
        return self.gmm.sample_n(n_samples)