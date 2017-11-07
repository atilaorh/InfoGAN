import torch
import torch.nn as nn
from torch.autograd import Variable

import argparse, os, glob, subprocess
import skimage as sk
import numpy as np

from .infogan import *
from .utils import *


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument("-r","--root", help="Root directory of dataset",type='str')
	parser.add_argument("-b","--batch_size",help="Batch size for Generator output",type='int',default=64)
	parser.add_argument("-v","--visdom",help="Start visdom visualization server",type='str',action="store_true")
	args = parse.parse_args()

	if args.root:
		root = args.root
	else:
		root = os.getcwd()

	if args.visdom:
		subprocess.call("python3 -m visdom.server",shell=True)

	train_data, test_data = DataLoad(batch_size=args.batch_size)

	net_config={'num_filters'=512, 'num_classes'=10, 'target_res'=32}
	G = Generator(**net_config).cuda()
	D = Discriminator(**net_config).cuda()

    # Qc_X is part of D
    D_criterion = torch.nn.BCELoss()
    D_criterion = torch.nn.BCELoss()
    D_optimizer = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

    G_criterion = torch.nn.BCELoss()
    G_optimizer = optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))

	for i, train_batch in enumerate(train_data):

		real_data, _ = Variable(train_batch).cuda()

		z = Variable(sample_z(args.batch_size)).cuda()
		c = Variable(sample_c(args.batch_size)).cuda()

		G_inputs = torch.cat((z,c),1)
		fake_data = G(G_inputs)

		D_inputs = torch.cat((real_data,fake_data),0)

		D_labels = np.zeros((2*args.batch_size))
		D_labels[:args.batch_size]=1.
		D_labels = Variable(torch.from_numpy(D_labels.astype(np.float32))).cuda()

		# Discriminator
		D_optimizer.zero_grad()
		D_out, Qc_X = D(D_inputs)

		# Variational Mutual Information
		Qc_X = Qc_X[batch_size:]
		MILB = torch.mean(torch.sum(Qc_X*c,1))
		D_loss = D_criterion(D_out,D_labels) - MILB










	

