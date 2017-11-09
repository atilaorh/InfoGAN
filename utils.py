import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.utils as v_utils
from torchvision import transforms
from torch.autograd import Variable, grad
import torch
import numpy as np

def DataLoad(root,batch_size=64, download=False):

	transform = transforms.Compose([transforms.ToTensor(),
	                                transforms.Normalize((0.5, 0.5, 0.5),
	                                                     (0.5, 0.5, 0.5))])

	trainset = dset.CIFAR10(root=root, train=True,
	                                      download=download,
	                                      transform=transform)
	trainloader = DataLoader(trainset, batch_size=batch_size,
	                                          shuffle=True, num_workers=4)

	testset = dset.CIFAR10(root=root, train=False,
	                                     download=download,
	                                     transform=transform)
	testloader = DataLoader(testset, batch_size=batch_size,
	                                         shuffle=False, num_workers=4)

	return trainloader, testloader

def sample_z(batch_size=64,**kwargs):
	z = np.random.randn(batch_size,kwargs['num_filters']-kwargs['num_classes']).astype(np.float32)
	z = torch.from_numpy(z)
	return Variable(z)#.cuda()

def sample_c(batch_size=64,num_classes=10,label=1.):
	# Currently only supports classification
	code = np.zeros((batch_size,num_classes))
	sampleIdx = np.random.randint(0,num_classes,batch_size)
	# Label Smoothing
	code[range(batch_size),sampleIdx]=label
	code = code.astype(np.float32)
	code = torch.from_numpy(code)
	return Variable(code)#.cuda()


def GradientPenalty(D, real_data, fake_data, batch_size, target_res=32):
	# print "real_data: ", real_data.size(), fake_data.size()
	alpha = torch.rand(batch_size, 1)
	alpha = alpha.expand(batch_size, \
		real_data.nelement()//batch_size).contiguous().view(batch_size, 3, target_res, target_res)
	# alpha = alpha.cuda()

	interpolates = alpha * real_data + ((1 - alpha) * fake_data)

	# interpolates = interpolates.cuda()
	interpolates = Variable(interpolates, requires_grad=True)

	disc_interpolates,_ = D(interpolates,batch_size)

	gradients = grad(outputs=disc_interpolates, inputs=interpolates,
					grad_outputs=torch.ones(disc_interpolates.size()),
					create_graph=True, retain_graph=True, only_inputs=True)[0]

	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

	return gradient_penalty


