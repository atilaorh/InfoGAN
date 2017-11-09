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
		self.num_blocks = int(math.log2(target_res))-2

		self.layer1 = nn.Sequential(OrderedDict([
				('linear',nn.Linear(num_filters,4*4*num_filters)),
				('relu',nn.ReLU()),
				('batchnorm',nn.BatchNorm1d(4*4*num_filters)),
				]))

		self.upconv_blocks = []

		nf = int(num_filters)

		for res in range(self.num_blocks):

			self.upconv_blocks.append(nn.Sequential(OrderedDict([
					('fsconv_%d'%res,nn.ConvTranspose2d(nf,nf//2,3,2,1,1,bias=False)),
					('relu_%d'%res,nn.ReLU()),
					('batchnorm_%d'%res,nn.BatchNorm2d(nf//2)),
					])))
			nf = nf // 2

		self.output = nn.Sequential(OrderedDict([
				('conv',nn.Conv2d(nf,3,3,1,1)),
				('tanh',nn.Tanh())
			]))

	def forward(self,z):

		# print('Generator')
		# print(z.size())
		out = self.layer1(z)
		# print(out.size())
		out = out.view(-1,self.num_filters,4,4)
		# print(out.size())

		for res in range(self.num_blocks):
			out = self.upconv_blocks[res](out)
			# print(out.size())

		out = self.output(out)
		# print(out.size())

		return out


class Discriminator(nn.Module):

	def __init__(self, num_filters, num_classes, target_res):

		super(Discriminator,self).__init__()

		self.num_filters = 8 * num_filters // target_res
		self.num_classes = num_classes
		self.target_res = target_res
		self.num_blocks = int(math.log2(target_res)-3)
		self.layer1 = nn.Sequential(OrderedDict([
						('conv_1',nn.Conv2d(3,self.num_filters,3,2,1)),
						('relu_1',nn.LeakyReLU()),
						('batchnorm_1',nn.BatchNorm2d(self.num_filters)),
					]))

		self.conv_blocks = []

		nf = int(self.num_filters)

		for res in range(self.num_blocks):
			self.conv_blocks.append(nn.Sequential(OrderedDict([
								('conv_%d'%(res+2),nn.Conv2d(nf,nf*2,3,2,1)),
								('relu_%d'%(res+2),nn.LeakyReLU()),
								('batchnorm_%d'%(res+2),nn.BatchNorm2d(nf*2)),
							])))
			nf *= 2

		self.realfake = nn.Sequential(OrderedDict([
							('logit',nn.Linear(num_filters,1)),
							('prob',nn.Sigmoid())
							]))

		self.recognition = nn.Sequential(OrderedDict([
							('logits',nn.Linear(num_filters,num_classes)),
							('logprobs',nn.LogSoftmax())
							]))


	def forward(self,x,batch_size):

		# print('Discriminator')
		# print(x.size())
		out = self.layer1(x)
		# print(out.size())

		for res in range(self.num_blocks):
			out = self.conv_blocks[res](out)
			# print(out.size())
		# print(out.size())
		# out = out.view(batch_size,-1)
		out = torch.mean(out,-1,False)
		out = torch.mean(out,-1,False)
		# print(out.size())

		realfake = self.realfake(out)
		representation = self.recognition(out)

		return realfake, representation


