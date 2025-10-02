# Example: MC Dropout on MNIST (logits -> probs)
import torch, torchvision
from torch import nn, optim
from torchvision import transforms
from deepuq.models import MLP
from deepuq.methods import MCDropoutWrapper

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False)

model = MLP(28*28, [512,256], 10, p_drop=0.2)
opt = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1):
    for x,y in train_loader:
        opt.zero_grad(); logits = model(x); loss = nn.CrossEntropyLoss()(logits,y); loss.backward(); opt.step()

uq = MCDropoutWrapper(model, n_mc=50)
for x,y in test_loader:
    mean, var = uq.predict(x)
    print('batch mean shape', mean.shape, 'var shape', var.shape)
    break
