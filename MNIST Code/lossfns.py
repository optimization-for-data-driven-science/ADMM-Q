from __future__ import print_function
from __future__ import division
from builtins import range
from builtins import int
from builtins import dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as V
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.backends.cudnn as cudnn

import torch.nn.functional as F

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

# torch.manual_seed(11)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(11)

def hinge_loss(pred, y, dtype):

	N = pred.shape[0]
	C = 10
	# y_one_hot = V(torch.zeros(N, C).type(dtype), requires_grad=False)
	# y_one_hot[torch.arange(N), y] += 1
	# y_one_hot = 1 - y_one_hot

	pred -= pred[torch.arange(N), y].view(-1, 1)
	pred += 1
	# pred *= y_one_hot
	pred = torch.max(torch.tensor(0.0).type(dtype), pred)
	loss = (pred.pow(2).sum(dim=1) - 1).mean()

	# loss = (1 - pred[torch.arange(N), y])
	# loss = torch.max(torch.ones(N).type(dtype), loss)
	# loss = loss.pow(2).mean()

	return loss
