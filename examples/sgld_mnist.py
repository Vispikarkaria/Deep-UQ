# Example: SGLD MCMC sampling on MNIST (toy quick demo)
import torch, torchvision
from torch import nn
from torchvision import transforms
from deepuq.models import MLP
from deepuq.methods import collect_posterior_samples, predict_with_samples

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True)

model = MLP(28*28, [128], 10, p_drop=0.0)
samples = collect_posterior_samples(model, train_loader, n_steps=50, lr=1e-4, weight_decay=1e-4, burn_in=0.2)

x,_ = next(iter(train_loader))
mean, var = predict_with_samples(model, samples, x)
print(mean.shape, var.shape)
