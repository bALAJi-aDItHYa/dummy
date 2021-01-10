import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import unfold

inp = torch.randn(4,2,5,5)
out_channels = 6
conv = nn.Conv2d(in_channels=2, out_channels=6, kernel_size=(3,3), padding=1, stride=1, bias=False)

f = conv.weight
print(f.shape)

k = f.view(out_channels, -1)
print(k[2].shape)

un = inp.unfold(2,3,1).unfold(3,3,1).reshape(4,-1,2,3,3)
un = un.reshape(un.size(0),un.size(1),-1)
print(un[3][2].shape)

i = torch.zeros(2,2)
r = 2.0
i[0][1] = r
print(i)

j = torch.Tensor(5,8)
print(j)

#for b in inp.size(0):
#	for i in inp.size(2):
#		for j in inp.size(3):
#			result = inp[b][i][j] * f[]