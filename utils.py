import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.utils as v_utils
from torchvision import transforms

def DataLoad(batch_size=64, download=True):

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

def sample_z(batch_size=64,dim=128):

	return torch.from_numpy(np.random.randn((batch_size,dim)))

def sample_c(batch_size=64,num_classes=10,label=1.):
	# Currently only supports classification
	code = np.zeros((batch_size,num_classes))
	sampleIdx = np.random.randint(0,num_classes,batch_size)
	# Label Smoothing
	code[range(batch_size),sampleIdx]=label

	return torch.from_numpy(code)


