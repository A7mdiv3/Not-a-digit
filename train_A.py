import torch
from torch import nn
from torchvision import datasets
from brainy import make_brain

mnist=datasets.MNIST("data", train=True, download=True)
pictures=mnist.data.float()/255
plabels=mnist.targets

brainy=make_brain(10)
wrong=nn.CrossEntropyLoss()
teach=torch.optim.Adam(brainy.parameters(), lr=0.001)

for step in range(501):
    guesses=brainy(pictures)
    mistake=wrong(guesses, plabels)
    teach.zero_grad()
    mistake.backward()
    teach.step()

    if step %100==0:
        print(step,mistake.item())

torch.save(brainy.state_dict(), "baseline")
