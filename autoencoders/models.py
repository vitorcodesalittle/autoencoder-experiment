from torch import nn
from torch.nn.functional import relu, linear

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

# How to calculate input dimensions of the layer after a convolutional layer?
# n_out = floor( (n_in + 2p - k) / s ) + 1
# n_out: number of output features
# n_in: number of input features
# p : padding
# k: kernel size
# s: stride
class AutoencoderConv(nn.Module):
  def __init__(self):
    super(AutoencoderConv, self).__init__()
    self.downsample = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4, stride=2), # ((32, 32) - (4, 4)) / 2 + 1 = (28, 28) / 2 + 1 = (14, 14) + 1  = (14, 14) + (1, 1) = (15, 15)
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1), # ((15, 15) - (3, 3)) / 1 + 1 = (12, 12) + 1 = (12, 12) + (1, 1) = (13, 13)
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1), # (11, 11)
        nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1), # (9, 9)
        nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1), # (7, 7)
        nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=(1,1)), # ((7, 7) + 2(1, 1) - (3, 3) / 1 + 1 = (6, 6) / 2 + 1 = 
        nn.ReLU(),
    )

    self.upsample = nn.Sequential(
        nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=3, stride=2, padding=(1,1)), # ((7, 7) + 2(1, 1) - (3, 3) / 1 + 1 = (6, 6) / 2 + 1 = 
        nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=1), # (7, 7)
        nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=1), # (9, 9)
        nn.ConvTranspose2d(in_channels=16, out_channels=64, kernel_size=3, stride=1), # (11, 11)
        nn.ConvTranspose2d(in_channels=64, out_channels=256, kernel_size=3, stride=1), # ((15, 15) - (3, 3)) / 1 + 1 = (12, 12) + 1 = (12, 12) + (1, 1) = (13, 13)
        nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=4, stride=2), # ((32, 32) - (4, 4)) / 2 + 1 = (28, 28) / 2 + 1 = (14, 14) + 1  = (14, 14) + (1, 1) = (15, 15)
    )
  def forward(self, x):
    x = self.downsample(x)
    x = self.upsample(x)
    return x



class CodeClassifier(nn.Module):
    """
    module that optimizes the classification of image based on it's encoded version
    """
    def __init__(self, autoencoder):
        super(CodeClassifier, self).__init__()
        autoencoder.require_grad = False
        self.encoder = autoencoder.downsample
        self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(1024, 10),
                nn.ReLU(),
                nn.Softmax(dim=1)
        )
    # x is of shape (batch_size, 32, 32, 3)"
    def forward(self, x):
        x = self.encoder(x) # x is of shape (4, 4, 64)
        x = self.classifier(x)
        return x

class AutoencoderConvClassifier(nn.Module):
    pass
