from torch import nn
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

