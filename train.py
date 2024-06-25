import torch as th
from torch.utils.data import DataLoader

import yaml
from argparse import ArgumentParser
from pathlib import Path

from models import SimpleMLP
from datasets import TwoGaussians
from diffusion import DDPM, get_beta_schedule
from trainer import Trainer

parser = ArgumentParser()
parser.add_argument("--params", type=str) 
args = parser.parse_args()

with open(args.params) as file:
    config = yaml.safe_load(file)

# prepare dataset
ds_config = config["dataset"]
ds_kwargs = {"distance_frac": ds_config["distance_frac"]}
train_loader = DataLoader(
    TwoGaussians(**ds_kwargs).sample(
        ds_config["train"]["n_samples"],
        seed=ds_config["train"].get("seed", 0),
    )[:, None],
    batch_size=64,
    shuffle=True,
)
test_samples = TwoGaussians(**ds_kwargs).sample(
    ds_config["test"]["n_samples"],
    seed=ds_config["test"].get("seed", 0),
)[:, None]


# prepare diffusion
diff_config = config["diffusion"]
betas = get_beta_schedule(diff_config["T"])
denoiser = SimpleMLP(d_in=1, T=diff_config["T"], hidden_dim=diff_config["hidden_dim"])
ddpm = DDPM(betas, denoiser)

opt = th.optim.Adam(ddpm.parameters(), lr=3e-4)

device = "cuda" if th.cuda.is_available() else "cpu"

# prepare trainer
trainer_config = config["trainer"]
log_every = trainer_config["log_every"]
log_dir = Path(trainer_config["log_dir"])
sub_dir = args.params.split("/")[-1].split(".")[0]
log_dir = log_dir / sub_dir
log_dir.mkdir(exist_ok=True, parents=True)

trainer = Trainer(ddpm, opt, device, log_every, log_dir)

n_iters = trainer_config["n_iters"]
trainer.train(train_loader, n_iters, test_samples)