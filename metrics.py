import torch as th

class FID:
    def __init__(self, preprocess_fn=None):
        self.preprocess_fn = preprocess_fn

    def __call__(self, samples1, samples2):
        if self.preprocess_fn:
            samples1 = self.preprocess_fn(samples1)
            samples2 = self.preprocess_fn(samples2)
        mean1, std1 = samples1.mean(dim=0), samples1.std(dim=0)
        mean2, std2 = samples2.mean(dim=0), samples2.std(dim=0) 
        return th.norm(mean1 - mean2) + th.norm(std1 - std2)