from __future__ import print_function
from __future__ import division
from builtins import range
from builtins import int
from builtins import dict


import argparse
import pickle
import random

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

from model import ConvNet

import numpy as np

import matplotlib.pyplot as plt

from lossfns import *
from dataprocess import *

import os
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# random.seed(11)
# np.random.seed(11)
# torch.manual_seed(11)
# torch.cuda.manual_seed(11)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def main(args):
	
	loader_train, loader_test = loadData(args)
	dtype = torch.cuda.FloatTensor
	fname = "model/MNIST_normal.pth"
	
	model = normal_train(args, loader_train, loader_test, dtype)

	torch.save(model, fname)

	print("Training done, model save to %s :)" % fname)

	model = torch.load(fname)
	
	model = binarize(args, model, loader_train, loader_test, dtype)
	test(model, loader_test, dtype)

	fname = "model/MNIST_BN.pth"
	torch.save(model, fname)

	print("Binarization done, model save to %s :)" % fname)

	

def binarize(args, model, loader_train, loader_test, dtype):
	
	# Uncomment the following line to override pre-training
	# model = ConvNet()

	loss_file = {}
	loss_file["loss1"] = []
	loss_file["loss2"] = []
	loss_file["loss3"] = []
	loss_file["loss"] = []

	model = ConvNet()
	model = model.type(dtype)
	model.train()

	loss_f = nn.CrossEntropyLoss()

	rho = 0.00001
	beta = 0.95

	XX = []
	YY = []
	Lambda = []
	scale = []


	for m in model.modules():

		if isinstance(m, (nn.Conv2d)) or isinstance(m, (nn.Linear)):
			
			tmpWeight = V(m.weight.data.clone().type(dtype), requires_grad=False)
			tmpWeight = (tmpWeight > 0).float() * 2 - 1
			tmpBias = V(m.bias.data.clone().type(dtype), requires_grad=False)
			tmpWeightLambda = V(torch.zeros(m.weight.shape).type(dtype), requires_grad=False)
			tmpBiasLambda = V(torch.zeros(m.bias.shape).type(dtype), requires_grad=False)

			XX.append([m.weight, m.bias])
			YY.append([tmpWeight, tmpBias])
			Lambda.append([tmpWeightLambda, tmpBiasLambda])

	SCHEDULE_EPOCHS = [40, 40, 40]
	SCHEDULE_LRS = [0.01, 0.01, 0.001]
	rhos = [0.00001, 0.0001, 0.001]
	
	for (num_epochs, learning_rate) in zip(SCHEDULE_EPOCHS, SCHEDULE_LRS):

		optimizer = optim.Adam(model.parameters(), lr=learning_rate)
		
		for epoch in range(num_epochs):

			print('\nTraining epoch %d / %d with LR %.7f ...\n' % (epoch + 1, num_epochs, learning_rate))
			
			#### Update on Y ####

			if epoch % 5 == 0:

				for j in range(len(YY)):

					# mask = (torch.rand(YY[j][0].shape).type(dtype) < beta).float()
					# YY[j][0] = mask * (((XX[j][0].data + (1 / rho) * Lambda[j][0]) > 0).float() * 2 - 1) + (1 - mask) * YY[j][0]
					

					YY[j][0] = ((XX[j][0].data + (1 / rho) * Lambda[j][0]) > 0).float() * 2 - 1
					# YY[j][1] = ((XX[j][1].data + (1 / rho) * Lambda[j][1]) > 0).float() * 2 - 1

			
			for i, (X_, y_) in enumerate(loader_train):

				X = V(X_.type(dtype), requires_grad=False)
				y = V(y_.type(dtype), requires_grad=False).long()

				#### update on X ####

				preds = model(X)

				# loss1 = loss_f(preds, y)
				loss1 = hinge_loss(preds, y, dtype)
				loss2 = 0
				loss3 = 0

				for j in range(len(YY)):

					loss2 += Lambda[j][0].mul(XX[j][0] - YY[j][0]).sum()
					loss3 += rho / 2 * (XX[j][0] - YY[j][0]).norm().pow(2)

				loss = loss1 + loss2 + loss3

				loss_file["loss"].append(loss.item())
				loss_file["loss1"].append(loss1.item())
				loss_file["loss2"].append(loss2.item())
				loss_file["loss3"].append(loss3.item())

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				if (i + 1) % args.print_every == 0:
					print('Batch %d done, loss1 = %.7f, loss2 = %.7f, loss3 = %.7f' % (i + 1, loss1.item(), loss2.item(), loss3.item()))


			print('Batch %d done, loss = %.7f' % (i+ 1, loss.item()))
			test(model, loader_test, dtype)

			## update on lambda
			if (epoch + 1) % 5 == 0:
				for j in range(len(YY)):

					Lambda[j][0] += rho * (XX[j][0].data - YY[j][0].data)
					# Lambda[j][1] += rho * (XX[j][1].data - YY[j][1].data)
		

	pickle.dump(loss_file, open("loss.log", "wb"))

	return model

def binarize_random(args, model, loader_train, loader_test, dtype):

	loss_file = {}
	loss_file["loss1"] = []
	loss_file["loss2"] = []
	loss_file["loss3"] = []
	loss_file["loss"] = []

	model = model.type(dtype)
	model.train()

	loss_f = nn.CrossEntropyLoss()

	rho = 0.001
	beta = 0.3

	XX = []
	YY = []
	Lambda = []
	scale = []

	for m in model.modules():

		if isinstance(m, (nn.Conv2d)) or isinstance(m, (nn.Linear)):
			
			tmpWeight = V(m.weight.data.clone().type(dtype), requires_grad=False)
			tmpWeight = (tmpWeight > 0).float() * 2 - 1
			tmpBias = V(m.bias.data.clone().type(dtype), requires_grad=False)
			tmpWeightLambda = V(torch.zeros(m.weight.shape).type(dtype), requires_grad=False)
			tmpBiasLambda = V(torch.zeros(m.bias.shape).type(dtype), requires_grad=False)

			XX.append([m.weight, m.bias])
			YY.append([tmpWeight, tmpBias])
			Lambda.append([tmpWeightLambda, tmpBiasLambda])


	SCHEDULE_EPOCHS = [20, 20]
	SCHEDULE_LRS = [0.01, 0.001]
		
	for (num_epochs, learning_rate) in zip(SCHEDULE_EPOCHS, SCHEDULE_LRS):

		optimizer = optim.Adam(model.parameters(), lr=learning_rate)
		
		for epoch in range(num_epochs):

			print('\nTraining epoch %d / %d with LR %.7f ...\n' % (epoch + 1, num_epochs, learning_rate))
			
			#### Update on Y ####

			if epoch % 5 == 0:

				for j in range(len(YY)):

					mask = (torch.rand(YY[j][0].shape).type(dtype) < beta).float()
					YY[j][0] = mask * (((XX[j][0].data + (1 / rho) * Lambda[j][0]) > 0).float() * 2 - 1) + (1 - mask) * YY[j][0]

			
			for i, (X_, y_) in enumerate(loader_train):

				X = V(X_.type(dtype), requires_grad=False)
				y = V(y_.type(dtype), requires_grad=False).long()

				#### update on X ####

				preds = model(X)

				# loss1 = loss_f(preds, y)
				loss1 = hinge_loss(preds, y, dtype)
				loss2 = 0
				loss3 = 0

				for j in range(len(YY)):

					loss2 += Lambda[j][0].mul(XX[j][0] - YY[j][0]).sum()
					loss3 += rho / 2 * (XX[j][0] - YY[j][0]).norm().pow(2)

				loss = loss1 + loss2 + loss3

				loss_file["loss"].append(loss.item())
				loss_file["loss1"].append(loss1.item())
				loss_file["loss2"].append(loss2.item())
				loss_file["loss3"].append(loss3.item())

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				if (i + 1) % args.print_every == 0:
					print('Batch %d done, loss1 = %.7f, loss2 = %.7f, loss3 = %.7f' % (i + 1, loss1.item(), loss2.item(), loss3.item()))


			print('Batch %d done, loss = %.7f' % (i+ 1, loss.item()))
			test(model, loader_test, dtype)

			## update on lambda
			if (epoch + 1) % 5 == 0:
				for j in range(len(YY)):

					Lambda[j][0] += rho * (XX[j][0].data - YY[j][0].data)
					# Lambda[j][1] += rho * (XX[j][1].data - YY[j][1].data)
		

	pickle.dump(loss_file, open("loss.log", "wb"))

	return model

def PGD(args, model, loader_train, loader_test, dtype):

	# Uncomment the following line to override pre-training
	# model = ConvNet()
	
	model = model.type(dtype)

	model.train()
		
	loss_f = nn.CrossEntropyLoss()

	SCHEDULE_EPOCHS = [20, 20]
	SCHEDULE_LRS = [0.01, 0.001]
	
	for (num_epochs, learning_rate) in zip(SCHEDULE_EPOCHS, SCHEDULE_LRS):

		optimizer = optim.Adam(model.parameters(), lr=learning_rate)
		
		for epoch in range(num_epochs):
			
			print('\nTraining epoch %d / %d with LR %.7f ...\n' % (epoch + 1, num_epochs, learning_rate))

			for i, (X_, y_) in enumerate(loader_train):

				X = V(X_.type(dtype), requires_grad=False)
				y = V(y_.type(dtype), requires_grad=False).long()

				preds = model(X)
				loss = hinge_loss(preds, y, dtype)
				
				if (i + 1) % args.print_every == 0:
					print('Batch %d done, loss = %.7f' % (i + 1, loss.item()))

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				for m in model.modules():

					if isinstance(m, (nn.Conv2d)) or isinstance(m, (nn.Linear)):
						m.weight.data = (m.weight.data > 0).float() * 2 - 1

			print('Batch %d done, loss = %.7f' % (i + 1, loss.item()))
			
			test(model, loader_test, dtype)

	return model


def test(model, loader_test, dtype):
	num_correct = 0
	num_samples = 0
	model.eval()
	for X_, y_ in loader_test:

		X = V(X_.type(dtype), requires_grad=False)
		y = V(y_.type(dtype), requires_grad=False).long()

		logits = model(X)
		_, preds = logits.max(1)

		num_correct += (preds == y).sum()
		num_samples += preds.size(0)

	accuracy = float(num_correct) / num_samples * 100
	print('\nAccuracy(X) = %.2f%%' % accuracy)

	num_correct = 0
	num_samples = 0
	modelQuantized = copy.deepcopy(model)

	for m in modelQuantized.modules():
		
		if isinstance(m, (nn.Conv2d)) or isinstance(m, (nn.Linear)):
			m.weight.data = (m.weight.data > 0).float() * 2 - 1
			# m.bias.data = (m.bias.data > 0).float() * 2 - 1



	for X_, y_ in loader_test:

		X = V(X_.type(dtype), requires_grad=False)
		y = V(y_.type(dtype), requires_grad=False).long()

		logits = modelQuantized(X)
		_, preds = logits.max(1)

		num_correct += (preds == y).sum()
		num_samples += preds.size(0)

	accuracy = float(num_correct) / num_samples * 100
	print('Accuracy(Y) = %.2f%%' % accuracy)

	model.train()

def normal_train(args, loader_train, loader_test, dtype):

	model = ConvNet()
	model = model.type(dtype)


	model.train()
		
	loss_f = nn.CrossEntropyLoss()

	SCHEDULE_EPOCHS = [20, 20]
	SCHEDULE_LRS = [0.01, 0.001]

	for (num_epochs, learning_rate) in zip(SCHEDULE_EPOCHS, SCHEDULE_LRS):

		optimizer = optim.Adam(model.parameters(), lr=learning_rate)
		
		for epoch in range(num_epochs):
			
			print('\nTraining epoch %d / %d with LR %.7f ...\n' % (epoch + 1, num_epochs, learning_rate))

			for i, (X_, y_) in enumerate(loader_train):

				X = V(X_.type(dtype), requires_grad=False)
				y = V(y_.type(dtype), requires_grad=False).long()

				preds = model(X)

				loss = hinge_loss(preds, y, dtype)
				
				if (i + 1) % args.print_every == 0:
					print('Batch %d done, loss = %.7f' % (i + 1, loss.item()))

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			print('Batch %d done, loss = %.7f' % (i + 1, loss.item()))
			
			test(model, loader_test, dtype)

	return model

def parse_arguments():

	parser = argparse.ArgumentParser()
	parser.add_argument('--data-dir', default='./dataset', type=str,
						help='path to dataset')
	parser.add_argument('--batch-size', default=512, type=int,
						help='size of each batch of cifar-10 training images')
	parser.add_argument('--print-every', default=50, type=int,
						help='number of iterations to wait before printing')

	return parser.parse_args()

if __name__ == '__main__':
	args = parse_arguments()
	main(args)

