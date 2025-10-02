# Example: Variational Inference (Bayes by Backprop) on MNIST
import torch, torchvision
from torch import optim, nn
from torchvision import transforms
from deepuq.methods import BayesByBackpropMLP, vi_elbo_step

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)

model = BayesByBackpropMLP(28*28, [256], 10, prior_sigma=0.1)
opt = optim.Adam(model.parameters(), lr=1e-3)
n_batches = len(train_loader)
for epoch in range(1):
    for x,y in train_loader:
        opt.zero_grad()
        loss, nll, kl = vi_elbo_step(model, x, y, n_batches)
        loss.backward(); opt.step()
print('trained one epoch')
