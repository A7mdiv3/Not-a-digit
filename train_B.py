import torch
from torch import nn
from brainy import make_brain
from torchvision import datasets

mnist=datasets.MNIST("data", train=True, download=True)
digits=mnist.data.float()/255
dlabels=mnist.targets

emnist=datasets.EMNIST("data", split="letters", train=True, download=True)
pick=torch.randperm(len(emnist.data))[:30000]
letters=emnist.data[pick].transpose(1, 2).float()/255
llabels=torch.full((30000,), 10)

pictures=torch.cat((digits, letters))
labels=torch.cat((dlabels, llabels))

brainy=make_brain(11)
wrong=nn.CrossEntropyLoss()
teach=torch.optim.Adam(brainy.parameters(), lr=0.001)

for step in range(2000):
    guesses=brainy(pictures)
    mistake=wrong(guesses, labels)
    teach.zero_grad()
    mistake.backward()
    teach.step()

    if step %100==0:
        print(step,mistake.item())

torch.save(brainy.state_dict(), "baseline_B")

