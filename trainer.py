import torch as th
import matplotlib.pyplot as plt

import json
from IPython.display import clear_output
from collections import defaultdict

from metrics import FID

class Trainer:
    def __init__(
        self, ddpm, opt, device, log_every, log_dir
    ):
        self.ddpm = ddpm.to(device)
        self.opt = opt
        self.device = device

        self.log_every = log_every
        self.log_dir = log_dir

        self.history = defaultdict(list)
    
    def train(self, train_loader, n_iters, test_samples=None):
        step = 0
        data_iter = iter(train_loader)
        loss_acc = 0
        while step < n_iters:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            loss = self._train_step(batch)
            loss_acc += loss / self.log_every

            if (step + 1) % self.log_every == 0:
                self._log(loss_acc, test_samples)
                self._plot()
                loss_acc = 0

            step += 1

    def _train_step(self, batch):
        self.ddpm.train()
        batch = batch.to(self.device)
        loss = self.ddpm.train_loss(batch)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        return loss.item()

    def _log(self, loss, test_samples=None):
        self.history["loss"].append(loss)
        if test_samples is not None:
            self.ddpm.eval()
            with th.no_grad():
                samples = self.ddpm.sample(test_samples.shape[0])
            self.history["FID"].append(FID()(samples.to(self.device), test_samples.to(self.device)).detach().cpu().item())
        with open(f"{self.log_dir}/history.json", 'w') as file:
            json.dump(self.history, file)

    def _plot(self):
        clear_output()
        nrows, ncols = 1, len(self.history)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
        fig.set_figheight(4)
        fig.set_figwidth(4 * ncols)

        for key, ax in zip(self.history.keys(), axes[0]):
            ax.plot(self.history[key], label=f"{key}")
            ax.legend()
        plt.show()
        plt.savefig(f"{self.log_dir}/history_plot.png")
