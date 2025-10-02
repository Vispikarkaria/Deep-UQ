# Example: Laplace Approximation with laplace-torch
import torch, torchvision
from torch import nn, optim
from torchvision import transforms
from deepuq.models import MLP
from deepuq.methods import LaplaceWrapper

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False)

model = MLP(28*28, [256], 10, p_drop=0.0)
opt = optim.Adam(model.parameters(), lr=1e-3)
for x,y in train_loader:
    opt.zero_grad(); logits = model(x); loss = nn.CrossEntropyLoss()(logits,y); loss.backward(); opt.step()
    break

la = LaplaceWrapper(model, 'classification', 'diag')
la.fit(train_loader, prior_precision=1.0)
for x,y in test_loader:
    probs, _ = la.predict(x)
    print('probs', probs.shape)
    break
