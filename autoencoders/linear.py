from torch import nn
from torch.functional import F
class AutoencoderLinear(nn.Module):
  def __init__(self):
    super(AutoencoderLinear, self).__init__()
    self.f = nn.Sequential(
      nn.Flatten(),
      nn.Linear(3072, 512),
      nn.ReLU(),
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, 256),
      nn.ReLU(),
      nn.Linear(256, 512),
      nn.ReLU(),
      nn.Linear(512, 3072),
      nn.ReLU(),
      nn.Unflatten(1, (3, 32, 32))
    )

  def forward(self, x):
    x = self.f(x)
    return x


model = AutoencoderLinear()
summary(model, (32, 32, 3))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
history = trainAutoEncoder(model, epochs=20, momentum=0.9, debug=True, criterion=nn.MSELoss())
trainingStats(history)

