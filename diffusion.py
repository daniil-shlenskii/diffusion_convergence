import re
import numpy as np

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from typing import Optional, Tuple


def _extract_into_tensor(arr: th.Tensor, timesteps: th.Tensor, broadcast_shape: Tuple):
    """
    Extract values from a 1-D torch tensor for a batch of indices.
    :param arr: 1-D torch tensor.
    :param timesteps: a tensor of indices to extract from arr.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = arr.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def get_beta_schedule(num_diffusion_timesteps: int) -> th.Tensor:
    scale = 1000 / num_diffusion_timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    betas = th.from_numpy(betas).double()
    return betas


class BaseDiffusion:
    def __init__(self, betas: th.Tensor) -> None:
        self.betas = betas
        self.alphas = 1 - self.betas
        self.alphas_cumprod = th.cumprod(self.alphas, dim=-1)
        self.num_timesteps = len(self.betas)


class ForwardDiffusion(BaseDiffusion):
    def q_mean_variance(self, x0: th.Tensor, t: th.Tensor) -> th.Tensor:
        # ====
        # your code
        # calculate mean and variance of the distribution q(x_t | x_0) (use equation (1))
        sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        mean = _extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x0.shape)
        # ====
        return mean, variance

    def q_sample(self, x0: th.Tensor, t: th.Tensor, noise: Optional[th.Tensor]=None) -> th.Tensor:
        # ====
        # your code
        # sample from the distribution q(x_t | x_0) (use equation (1))
        if noise is None:
            noise = th.randn_like(x0)
        mean, variance = self.q_mean_variance(x0=x0, t=t)
        samples = mean + variance.sqrt() * noise
        # ====
        return samples


class ForwardDiffusion(BaseDiffusion):
    def q_mean_variance(self, x0: th.Tensor, t: th.Tensor) -> th.Tensor:
        # ====
        # your code
        # calculate mean and variance of the distribution q(x_t | x_0) (use equation (1))
        sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        mean = _extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x0.shape)
        # ====
        return mean, variance

    def q_sample(self, x0: th.Tensor, t: th.Tensor, noise: Optional[th.Tensor]=None) -> th.Tensor:
        # ====
        # your code
        # sample from the distribution q(x_t | x_0) (use equation (1))
        if noise is None:
            noise = th.randn_like(x0)
        mean, variance = self.q_mean_variance(x0=x0, t=t)
        samples = mean + variance.sqrt() * noise
        # ====
        return samples
    

class ReverseDiffusion(BaseDiffusion):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alphas_cumprod_prev = th.cat(
            [th.tensor([1.0], device=self.betas.device), self.alphas_cumprod[:-1]], dim=0
        )

        # ====
        # your code
        # calculate variance of the distribution q(x_{t-1} | x_t, x_0) mean (use equation (3))
        self.variance = (
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * self.betas
        )
        # ====

        # ====
        # your code
        # calculate coefficients of the distribution q(x_{t-1} | x_t, x_0) mean (use equation (2))
        self.xt_coef = (
            self.alphas.sqrt() * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )
        self.x0_coef = (
            self.betas * (self.alphas_cumprod_prev).sqrt() / (1 - self.alphas_cumprod)
        )
        # ====

    def get_x0(self, xt: th.Tensor, eps: th.Tensor, t: th.Tensor) -> th.Tensor:
        # ====
        # your code
        # get x_0 (use equations (4) and (2))
        alphas_cumprod = _extract_into_tensor(self.alphas_cumprod, t, xt.shape)
        x0 = (xt - (1 - alphas_cumprod).sqrt() * eps) / alphas_cumprod.sqrt()
        # ====
        return x0
        
    def q_posterior_mean_variance(
        self, xt: th.Tensor, eps: th.Tensor, t: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        # ====
        # your code
        # get mean and variance of the distribution q(x_{t-1} | x_t, x_0) mean (use equations (2) and (3))
        variance = _extract_into_tensor(self.variance, t, xt.shape)

        x0 = self.get_x0(xt, eps, t)

        xt_coef = _extract_into_tensor(self.xt_coef, t, xt.shape)
        x0_coef = _extract_into_tensor(self.x0_coef, t, xt.shape)
        mean = xt_coef * xt + x0_coef * x0
        # ====
        return mean, variance

    def p_sample(self, xt: th.Tensor, eps: th.Tensor, t: th.Tensor) -> th.Tensor:
        # read this code carefully
        mean, variance = self.q_posterior_mean_variance(xt=xt, eps=eps, t=t)
        noise = th.randn_like(xt, device=xt.device)
        
        nonzero_mask = th.ones_like(t)  # to not add any noise while predicting x0
        nonzero_mask[t == 0] = 0
        nonzero_mask = _extract_into_tensor(
            nonzero_mask, th.arange(nonzero_mask.shape[0]), xt.shape
        )
        nonzero_mask = nonzero_mask.to(xt.device)
        sample = mean + nonzero_mask * variance.sqrt() * noise
        return sample.float()
    

class DDPM(nn.Module):
    def __init__(
        self,
        betas: th.Tensor,
        model: nn.Module, 
        shape: Optional[th.Tensor] = None,
    ) -> None:
        super().__init__()

        self.forward_diffusion = ForwardDiffusion(betas=betas)
        self.reverse_diffusion = ReverseDiffusion(betas=betas)
        self.model = model
        self.num_timesteps = len(betas)

        self.register_buffer("betas", betas)
        self.register_buffer("shape", shape)

    @property
    def device(self) -> None:
        return next(self.parameters()).device

    @th.no_grad()
    def sample(self, num_samples) -> th.Tensor:
        assert self.shape is not None

        x = th.randn((num_samples, *self.shape), device=self.device, dtype=th.float32)
        indices = list(range(self.num_timesteps))[::-1]

        for i in tqdm(indices):
            t = th.tensor([i] * num_samples, device=x.device)
            # ====
            # your code
            # 1) get epsilon from the model
            # 2) sample from the reverse diffusion
            eps = self.model(x=x, t=t)
            x = self.reverse_diffusion.p_sample(xt=x, eps=eps, t=t)
            # ====
        return x

    def train_loss(self, x0: th.Tensor) -> th.Tensor:
        if self.shape is None:
            self.shape = th.tensor(list(x0.shape)[1:], device="cpu")
        
        t = th.randint(0, self.num_timesteps, size=(x0.size(0),), device=x0.device)
        noise = th.randn_like(x0)
        # ====
        # your code
        # 1) get x_t
        # 2) get epsilon from the model
        xt = self.forward_diffusion.q_sample(x0=x0, t=t, noise=noise)
        eps = self.model(x=xt, t=t)
        # ====
        loss = F.mse_loss(eps, noise)
        return loss

    @classmethod
    def from_pretrained(cls: "DDPM", model: nn.Module, ckpt_path: str) -> "DDPM":
        ckpt = th.load(ckpt_path)
        model_state_dict = {
            re.sub("model.", "", key): 
            val for key, val in ckpt.items() if "ema." in key
        }
        model.load_state_dict(model_state_dict)
        return cls(
            betas=ckpt["betas"],
            model=model,
            shape=ckpt["shape"],
        )