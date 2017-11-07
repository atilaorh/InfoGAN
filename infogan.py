import torch 
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
import math

class Generator(nn.Module):

	def __init__(self, num_filters, num_classes, target_res):

		super(Generator,self).__init__()

		self.num_filters = num_filters
		self.num_classes = num_classes
		self.target_res = target_res

		self.layer1 = nn.Linear((128+num_classes),4*4*num_filters)

		self.upconv_blocks = []

		nf = num_filters

		for res in range(math.log2(target_res)-2):

			self.upconv_blocks.append(nn.Sequential(OrderedDict([

							('fsconv_1',nn.ConvTranspose2D(nf,nf/2,3,2,1,1,bias=False)),
							('relu_1',nn.LeakyReLU()),
							('batchnorm_1',nn.BatchNorm2D()),

							('fsconv_2',nn.ConvTranspose2D(nf/2,nf/4,3,1,1,1,bias=False)),
							('relu_2',nn.LeakyReLU()),
							('batchnorm_2',nn.BatchNorm2D())

						])))
			nf /= 4

		self.output = nn.Tanh()

	def forward(self,z):

		out = self.layer1(z)
		out = out.view(-1,self.num_filters,4,4)

		for res in range(math.log2(self.target_res)-2):
			out = self.upconv_blocks[res](out)

		out = self.output(out)

		return out


class Discriminator(nn.Module):

	def __init__(self, num_filters, num_classes, target_res):

		super(Discriminator,self).__init__()

		self.num_filters = num_filters
		self.num_classes = num_classes
		self.target_res = target_res

		self.layer1 = nn.Conv2D(3,num_filters,3,1)

		self.conv_blocks = []

		nf = num_filters

		for res in range(math.log2(target_res)):
			self.conv_blocks.append(nn.Sequential(OrderedDict([
								('conv_1',nn.Conv2D(nf,nf*2,3,1)),
								('relu_1',nn.LeakyReLU()),
								('batchnorm_1',nn.BatchNorm2D()),
								('conv_2',nn.Conv2D(nf*2,nf*4,3,2)),
								('relu_2',nn.LeakyReLU()),
								('batchnorm_2',nn.BatchNorm2D())
							])))
			nf *= 4

		self.realfake = nn.Linear(target_res*num_filters,1)
		self.recognition = nn.Sequential(OrderedDict([
							('logits',nn.Linear(target_res*num_filters,num_classes)),
							('logprobs',nn.LogSoftmax())
							]))

	def forward(self,x):

		x = self.layer1(x)

		for res in range(math.log2(self.target_res)-2):
			out = self.conv_blocks[res](out)

		realfake = self.realfake(out)
		representation = self.recognition(out)

		return realfake, recognition



