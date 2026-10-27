import torch
from torchvision import datasets
from brainy import make_brain

torch.set_grad_enabled(False)

mnist=datasets.MNIST("data", train=False, download=True)
digits=mnist.data.float()/255
correct=mnist.targets

emnist=datasets.EMNIST("data", split="letters", train=False, download=True)
letters=emnist.data.transpose(1, 2).float()/255

fashion=datasets.FashionMNIST("data", train=False, download=True)
clothes=fashion.data.float()/255

brain_A=make_brain(10)
brain_A.load_state_dict(torch.load("baseline"))

brain_B=make_brain(11)
brain_B.load_state_dict(torch.load("baseline_B"))

def percent(scrore_list):
    return round(scrore_list.float().mean().item()*100, 1)

def sure_A(pictures):
    return torch.softmax(brain_A(pictures), dim=1).max(dim=1).values


answers_A=brain_A(digits).argmax(dim=1)
print(" ")
print("Accuracy of Brain A:",percent(answers_A==correct),"%")
print("A digits wrongly thrown out:",percent(sure_A(digits) < 0.9),"%")
print("A letters let through:",percent(sure_A(letters) >= 0.9),"%")
print("A clothes let through:",percent(sure_A(clothes) >= 0.9),"%")
print("====================================================")

answers_B=brain_B(digits).argmax(dim=1)
print("Accuracy of Brain B:",percent(answers_B==correct),"%")
print("B digits wrongly thrown out:",percent(brain_B(digits).argmax(dim=1) == 10),"%")
print("B letters let through:",percent(brain_B(letters).argmax(dim=1) != 10),"%")
print("B clothes let through:",percent(brain_B(clothes).argmax(dim=1) != 10),"%")