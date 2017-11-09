import torch
import torch.nn as nn
from torch.autograd import Variable

import argparse, os, glob, subprocess, time
import numpy as np

from infogan import *
from utils import *


if __name__ == '__main__':

	if torch.cuda.is_available():
		print('CUDA GPU is available') 

	parser = argparse.ArgumentParser()

	parser.add_argument("-r","--root", help="Root directory of dataset",required=True,type=str)
	parser.add_argument("-b","--batch_size",help="Batch size for Generator output",required=True,type=int)
	parser.add_argument("-e","--num_epochs",help="Number of epochs",required=True,type=int)
	parser.add_argument("-o","--gan_obj",help="Objective function for GAN",required=True,type=str)
	parser.add_argument("-v","--visdom",help="Start visdom visualization server",action="store_true")
	args = parser.parse_args()

	assert args.gan_obj in ['wgan','wgan-gp','jsgan-ns']
	if args.root:
		root = args.root
	else:
		root = os.getcwd()

	if args.visdom:
		subprocess.call("python3 -m visdom.server",shell=True)

	print(args.batch_size)

	ALPHA = 0.99 # moving average decay rate
	N_DISC_STEPS = 4
	LAMBDA = 10

	train_data, test_data = DataLoad(root=args.root, batch_size=args.batch_size)
	print('Dataloader initialized.')

	net_config={'num_filters':256, 'num_classes':10, 'target_res':32}
	G = Generator(**net_config)
	print('Generator initialized.')
	G = G#.cuda()
	D = Discriminator(**net_config)
	print('Discriminator initialized.')
	D = D#.cuda()

	# Qc_X is part of D
	D_criterion = torch.nn.BCELoss()
	D_optimizer = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0., 0.9))

	G_criterion = torch.nn.BCELoss()
	G_optimizer = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0., 0.9))

	for epoch in range(args.num_epochs):
		tic = time.time()
		for i, train_batch in enumerate(train_data):

			real_data, _ = train_batch
			real_data = Variable(real_data)
			real_data = real_data#.cuda()

			z = sample_z(args.batch_size,**net_config)
			c = sample_c(args.batch_size)

			G_inputs = torch.cat([z,c],1)#.type(torch.FloatTensor)
			G_inputs = G_inputs#.cuda()
			fake_data = G(G_inputs)

			D_inputs = torch.cat([real_data,fake_data],0)

			D_labels = np.zeros((2*args.batch_size,1))
			D_labels[:args.batch_size,:]=1.
			D_labels = Variable(torch.from_numpy(D_labels.astype(np.float32)))
			D_labels = D_labels#.cuda()

			# Discriminator
			D_optimizer.zero_grad()
			D_out, Qc_X = D(D_inputs,batch_size=args.batch_size*2)

			# Variational Mutual Information
			Qc_X = Qc_X[args.batch_size:]
			MILB = torch.mean(torch.sum(Qc_X*c,1))
			GP = GradientPenalty(D,real_data.data,fake_data.data,args.batch_size)

			D_fake = D_out[args.batch_size:,:].mean()
			D_real = D_out[:args.batch_size,:].mean()

			if args.gan_obj == 'wgan':
				D_loss =  D_fake - D_real - MILB 
			elif args.gan_obj == 'wgan-gp':
				D_loss = D_fake - D_real + LAMBDA*GP - MILB 
			elif args.gan_obj == 'jsgan-ns':
				D_loss = D_criterion(D_out,D_labels)- MILB

			D_loss.backward(retain_graph=True)
			D_optimizer.step()

			# One Generator update per N_DISC_STEPS steps
			if i % N_DISC_STEPS == 0:
				G_optimizer.zero_grad()

				if args.gan_obj in ['wgan','wgan-gp']:
					G_loss = -D_fake - MILB

				elif args.gan_obj == 'jsgan-ns':
					G_loss = - MILB + G_criterion(D_out[args.batch_size:,:],D_labels[:args.batch_size,:])

				G_loss.backward()
				G_optimizer.step()

			if i == 0:
				D_loss_ma = D_loss.data[0]
				G_loss_ma = G_loss.data[0]
				MILB_ma   = MILB.data[0]
				GP_ma     = GP.data[0]
			else:
				D_loss_ma = (1-ALPHA)*D_loss_ma + ALPHA*D_loss.data[0]
				G_loss_ma = (1-ALPHA)*G_loss_ma + ALPHA*G_loss.data[0]
				MILB_ma   = (1-ALPHA)*MILB_ma + ALPHA*MILB.data[0]
				GP_ma     = (1-ALPHA)*GP_ma + ALPHA*GP.data[0]

			if i % 64 == 0:

				print('[Epoch %d] iteration %05d: G_loss=%.3f | D_loss=%.3f | MILB:%.3f | GP:%.3f' \
					%(epoch,i,G_loss_ma,D_loss_ma,MILB_ma, GP_ma))

		print('Epoch %d took %d seconds' %(epoch,int(time.time()-tic)))










	

