import torch
from torch import nn

def make_brain(answers):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, answers)
    )

brain_a = make_brain(10)
brain_b = make_brain(11)