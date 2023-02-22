from __future__ import print_function
from __future__ import division
from builtins import range
from builtins import int
from builtins import dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.backends.cudnn as cudnn

import torch.nn.functional as F

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np
import random

# random.seed(11)
# np.random.seed(11)
# torch.manual_seed(11)
# torch.cuda.manual_seed(11)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False



class ChunkSampler(sampler.Sampler):
	def __init__(self, num_samples, start=0):
		self.num_samples = num_samples
		self.start = start
	
	def __iter__(self):
		return iter(range(self.start, self.start + self.num_samples))
	
	def __len__(self):
		return self.num_samples

def loadData(args):


	# transform_augment_train = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.ToTensor()])
	transform = T.Compose([T.ToTensor()])

	MNIST_train = dset.MNIST('./dataset', train=True, transform=T.ToTensor(), download=True)

	MNIST_test = dset.MNIST('./dataset', train=False, transform=T.ToTensor(), download=True)

  
	loader_train = DataLoader(MNIST_train, batch_size=args.batch_size, shuffle=True)
	
	loader_test = DataLoader(MNIST_test, batch_size=args.batch_size)

	return loader_train, loader_test