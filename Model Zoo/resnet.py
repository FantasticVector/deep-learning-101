import torch
import torch.nn as nn

class block(nn.Module):
  def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1) -> None:
    super().__init__()
    