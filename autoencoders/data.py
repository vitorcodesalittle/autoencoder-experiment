import torch
import torchvision
from torchvision.transforms import transforms

def load_data(batch_size=10):
    transform = transforms.Compose(
      [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      ]
    )
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    print(f"""Datasetinfo
      [+] Training size {trainloader.dataset.data.shape}
      [+] Testing size {testloader.dataset.data.shape}
    """)
    return trainloader, testloader

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
