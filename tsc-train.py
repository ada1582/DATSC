#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 22:23:23 2020

@author: daniyalusmani1
"""


from __future__ import print_function

import argparse


epsilon = 1e-7

parser = argparse.ArgumentParser(description='PyTorch training for deep abstaining classifiers',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
#parser.add_argument('--net_type', default=None, type=str, help='model')
parser.add_argument('--dropout', default=0.2, type=float, help='dropout_rate')
parser.add_argument('--datadir',  type=str, required=True, help='data directory')


parser.add_argument('--dataset', default='mnist', type=str, help='dataset = [mnist/cifar10/cifar100/stl10-labeled/stl10-c/tin200/fashion]')

parser.add_argument('--train_x', default=None, type=str, help='train features. will default to the dataset default')
parser.add_argument('--train_y', default=None, type=str, help='train labels.  will default to the dataset default')
parser.add_argument('--test_x', default=None, type=str, help='test features. will default to the dataset default')
parser.add_argument('--test_y', default=None, type=str, help='test labels. will default to the dataset default')

parser.add_argument('--noise_percentage',default=0, type=float, help='noise percentage in dataset')

parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
parser.add_argument('--nesterov',dest='nesterov', action='store_true',default=False,help="Use Nesterov acceleration with SGD")
parser.add_argument('--batch_size',dest='batch_size', default=128, type=int, help='batch size for training')
parser.add_argument('--test_batch_size',dest='test_batch_size', default=1024, type=int, help='batch size for testing')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--epoch-dilation', dest='epdl',default=1.0, type=float, help='epoch time dilation factor. Stretches or shrinks the training iterations and learning rate schedule')


parser.add_argument('--learn_epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train  without abstaining')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('-cuda_device', dest='cuda_device',type=str,default='auto',
	help='GPU device id to use. If not specified will automatically try to use a free GPU')
parser.add_argument('-use-gpu',action='store_true',dest='use_gpu',default=False,help='Use GPU if available')

parser.add_argument('--data-parallel',action='store_true',dest='data_parallel',default=False,help='Do data parallel training')
parser.add_argument('--parallel-device-count', default=None, dest='parallel_device_count',type=int, help='number of GPUs to use for parallel training. If none specified, and parallel training is enabled,  will use all available GPUs')


#network arguments
parser.add_argument('--net_type', default=None, type=str, help='model')
parser.add_argument('--depth', default=16, type=int, help='depth of model')
parser.add_argument('--loss_fn',dest='loss_fn',type=str,default=None,
                        help="abstaining loss function.  If this switch isn't used, defaults to regular cross-entropy (non-abstaining) loss")
parser.add_argument('--output_path', default="./", type=str, help='output path')

parser.add_argument('--log_file', default=None, type=str, help='logfile name')

parser.add_argument('--save_val_scores',action='store_true',default=False,help='writes validation set softmax scores to file after each epoch')

parser.add_argument('--rand_labels', default=None, type=float, help='randomize a fraction of the labels. should be in [0,1]')

parser.add_argument('--save_epoch_model', type=int, default=None, metavar='N',
                    help='save model at specified epoch')
parser.add_argument('--expt_name', default="", type=str, help='experiment name')
parser.add_argument('--del_noisy_data', default=False, action='store_true', help='whether data with randomized labels should be removed')

parser.add_argument('--exclude_train_indices', default=None, type=str, help='numpy array containing indices of training data that should be removed')

parser.add_argument('--alpha_final', default=1.0, type=float, help='final value of alpha hyperparameter in the loss function if using linear ramp-up')

#parser.add_argument('--del_noisy_data', default=False, action='store_true', help='whether data with randomized labels should be removed')
parser.add_argument('--save_train_scores',action='store_true',default=False,help='writes train set softmax scores to file after each epoch')


parser.add_argument('--alpha_init_factor', default=64.0, type=float, help='alpha initiliazation factor')

parser.add_argument('--eval_model', type=str, default=None, help='evaluate model on data set. Output will be softmax scores on the train and test splits of the dataset')

parser.add_argument('--save_best_model',action='store_true',default=False,help='saves best performing model')

parser.add_argument('--no_overwrite',action='store_true',default=False,help='will not overwrite previous best models')

parser.add_argument('--label_noise_info', default=None, type=str, help='pickle file  containing indices and labels to use for simulating label noise')

parser.add_argument('--abst_rate', default=None, type=float, help='Pre-specified abstention rate; will attempt to dynamically tune abstention hyperparameter to stabilize abstention at this rate')

#for wide residual networks
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')


parser.add_argument('--k_p', default=0.1, type=float, help='PID proportional gain')
parser.add_argument('--k_i', default=0.1, type=float, help='PID integral gain')
parser.add_argument('--k_d', default=0.05, type=float, help='PID derivative gain')


args = parser.parse_args()


from utils import ucr_archive
from torch.utils.data import DataLoader


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.modules.loss import _Loss
import os
import sys
import time
import datetime
from torch.autograd import Variable
from utils import gpu_utils, datasets, label_noise
import pdb
import numpy as np
from networks import wide_resnet,lenet,vggnet, resnet, resnet2, cnn, resnet1d, lstm
from networks import config as cf

#import dac_loss_pid
#import dac_loss

from loss_functions import loss_fn_dict

try:
	import cPickle as cp
except ModuleNotFoundError: #no cPickle in python 3
	import pickle as cp


#do time compression or dilation
args.epochs = int(args.epochs*args.epdl)
args.learn_epochs = int(args.learn_epochs*args.epdl)

if not args.save_epoch_model is None:
	args.save_epoch_model = int(args.save_epoch_model*args.epdl)


if not args.log_file is None:
	sys.stdout = open(args.log_file,'w')
	sys.stderr = sys.stdout

torch.manual_seed(args.seed)


start_epoch, num_epochs = 1, args.epochs
batch_size = args.batch_size
best_acc = 0.

print('\n[Phase 1] : Data Preparation')
trainset, testset, num_classes, series_length = datasets.get_data(args)

sys.stdout.flush()
#abstain class id is the last class
abstain_class_id = num_classes
#simulate label noise if needed
#trainset = label_noise.label_noise(args, trainset, num_classes)
#set data loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

if args.save_train_scores:
	train_perf_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=2)

def getNetwork(args):
	if args.loss_fn is None:
		extra_class = 0
	else:
		extra_class = 1

	if (args.net_type == 'lenet'):
		net = lenet.LeNet(num_classes+extra_class)
		file_name = 'lenet'
		net.apply(lenet.conv_init)

	elif (args.net_type == 'vggnet'):
		#net = vggnet.VGG(args.depth, num_classes+extra_class, args.dropout)
		net = vggnet.VGG(args.depth, num_classes+extra_class)
		file_name = 'vgg-'+str(args.depth)
		net.apply(vggnet.conv_init)

	elif (args.net_type == 'resnet'):
		net = resnet.ResNet(args.depth, num_classes+extra_class)
		file_name = 'resnet-'+str(args.depth)
		net.apply(resnet.conv_init)

	elif (args.net_type == 'resnet2'):

		if args.dataset == 'mnist' or args.dataset == 'fashion':
			num_channels = 1
		else:
			num_channels = 3

		if args.depth == 34:
			net = resnet2.ResNet34(num_classes=num_classes+extra_class,num_input_channels=num_channels)
			file_name = 'resnet2-34'#+str(args.depth)

		elif args.depth == 18:
			#pdb.set_trace()
			net = resnet2.ResNet18(num_classes=num_classes+extra_class,num_input_channels=num_channels)
			file_name = 'resnet2-18'#+str(args.depth)

		else:
			print('Error : Resnet-2 Network depth should either be 18 or 34')
			sys.exit(0)

		net.apply(resnet2.conv_init)

	elif (args.net_type == 'wide-resnet'):
		net = wide_resnet.Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes+extra_class)
		file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
		net.apply(wide_resnet.conv_init)
	elif (args.net_type == 'tsc-resnet'):
		net = resnet1d.ResNet(series_length, num_classes+extra_class)
		file_name = 'tsc-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
	elif (args.net_type == 'tsc-lstm'):
        # the input dimension has dimension of 1
		net = lstm.TSCLSTM(1,series_length, args.depth, num_classes+extra_class)
		file_name = 'tsc-lstm-'+str(args.depth)+'x'+str(args.widen_factor)
	else:
	    print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet / ResNet 1d')
	    sys.exit(0)

	return net, file_name


print('\n[Phase 2] : Model setup')
print('| Building net')
if args.net_type is None:
    print("Using Default conv net")
    file_name = 'conv_net'
    if args.loss_fn is None: #no abstention. use the actual number of classes
        net = cnn.ConvNet(num_classes,args.dropout)
    else: #use extra class for abstention 
            net = cnn.ConvNet(num_classes+1,args.dropout)
else:
    print('| Building net type [' + args.net_type + ']...')
    net, file_name = getNetwork(args)
    #net.apply(conv_init)
    sys.stdout.flush()


#set up loss function and CUDA-fy if needed
if args.loss_fn is None:
	criterion = nn.CrossEntropyLoss()
	print('Using regular  (non-abstaining) loss function during training')
else:
	if args.loss_fn == 'dac_loss':
		if args.abst_rate is None:
			criterion = loss_fn_dict['dac_loss'](model=net, learn_epochs=args.learn_epochs, 
				total_epochs=args.epochs,  use_cuda=False, alpha_final=args.alpha_final, 
				alpha_init_factor=args.alpha_init_factor)
		else:
			pid_tunings = (args.k_p, args.k_i, args.k_d)
			criterion = loss_fn_dict['dac_loss_pid'](model=net, learn_epochs=args.learn_epochs,
				 total_epochs=args.epochs, use_cuda=False, cuda_device=None, abst_rate=args.abst_rate,
				 alpha_final=args.alpha_final,alpha_init_factor=args.alpha_init_factor, pid_tunings=pid_tunings)
	else:
		print("Unknown loss function")
		sys.exit(0)

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s

#pdb.set_trace()
def train(epoch):
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	abstain = 0

	if args.dataset == 'mnist':
	    if int(epoch/args.epdl) > 5 and  int(epoch/args.epdl) <= 20:
	    	args.lr = 0.01
	    if int(epoch/args.epdl) > 20 and int(epoch/args.epdl) <=50:
	    	args.lr = 0.001

    #optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=0.9, 
	    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, 
	    	nesterov=args.nesterov, weight_decay=5e-4)
	    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, args.lr))
 

	else: #cifar 10/100/stl-10/tin200/fashion
		optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, int(epoch/args.epdl)),
		 momentum=0.9, weight_decay=5e-4,nesterov=args.nesterov)
		print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, int(epoch/args.epdl))))

	#print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
	#pdb.set_trace()	
	for batch_idx, (inputs, targets) in enumerate(trainloader):
        #print(dir(inputs.cuda))
        #quit()
		optimizer.zero_grad()
		inputs, targets = Variable(inputs), Variable(targets)
		if (args.net_type == 'tsc-lstm'): # update input to match lstm
			inputs = inputs.view(-1, series_length, 1) # input dimensions for time series data set are 1
		outputs = net(inputs)               # Forward Propagation
		#pdb.set_trace()
		if args.loss_fn is None:
			loss = criterion(outputs, targets)
		else:
			loss = criterion(outputs, targets, epoch)  # Loss

		loss.backward()  # Backward Propagation
		optimizer.step() # Optimizer update

		train_loss += loss.data.item()
		_, predicted = torch.max(outputs.data, 1)
		this_batch_size =targets.size(0) 
		total += this_batch_size
		correct += predicted.eq(targets.data).cpu().sum().data.item()

		abstained_now = predicted.eq(abstain_class_id).sum().data.item()
		abstain += abstained_now

		if total-abstain != 0:
			#pdb.set_trace()
			abst_acc = 100.*correct/(float(total-abstain))
		else:
			abst_acc = 1.

		sys.stdout.write('\r')
		sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tAbstained %d Abstention rate %.4f Cumulative Abstention Rate: %.4f Loss: %.4f Acc@1: %.3f%% Acc@2: %.3f%%'
		        %(epoch, num_epochs, batch_idx+1,
		            (len(trainset)//batch_size)+1, abstain, float(abstained_now)/this_batch_size, float(abstain)/total, loss.data.item(), 100.*correct/float(total), abst_acc))

		
		sys.stdout.flush()

	#if args.loss_fn == 'dac_loss_pid':
	#	criterion.print_abst_stats(epoch)



def save_train_scores(epoch):
	#net.eval()
	global training_loss

	train_softmax_scores = []
	total = 0
	abstained = 0
	loss_acc = np.array([])
	for batch_idx, (inputs, targets) in enumerate(train_perf_loader):
		if (args.net_type == 'tsc-lstm'): # update input to match lstm
			inputs = inputs.view(-1, series_length, 1) # input dimensions for time series data set are 1
		inputs, targets = Variable(inputs), Variable(targets)
		outputs = net(inputs)               # Forward Propagation
		if args.loss_fn is None:
			loss = criterion(outputs, targets)
		else:
			loss = criterion(outputs, targets, epoch)  # Loss
		p_out = F.softmax(outputs,dim=1)
		#pdb.set_trace()
		total += p_out.size(0)
		_,predicted = torch.max(p_out.data,1)
		abstained += predicted.eq(abstain_class_id).sum().data.item()
		train_softmax_scores.append(p_out.data)
		loss_acc = np.append(loss_acc, loss.data.item())
	training_loss = np.append(training_loss, np.mean(loss_acc))

	train_scores = torch.cat(train_softmax_scores).cpu().numpy()
	print('Saving train softmax scores at  Epoch %d' %(epoch))
	#if args.log_file is None:
	# if args.expt_name is None:
	# 	fn = 'test'
	# else:
	# 	fn = args.expt_name 
	fn = args.expt_name if args.expt_name else 'test'
	np.save(args.output_path+fn+"/"+ fn +".train_scores.epoch_"+str(epoch), train_scores)
	print("\n##### Epoch %d Train Abstention Rate at end of epoch %.4f" 
			%(epoch, float(abstained)/total))


def test(epoch):
	global best_acc
	global validation_loss
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	abstain = 0
	loss_acc = np.array([])
	if args.save_val_scores:
		val_softmax_scores = []

	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(testloader):
			if (args.net_type == 'tsc-lstm'): # update input to match lstm
				inputs = inputs.view(-1, series_length, 1) # input dimensions for time series data set are 1
			inputs, targets = Variable(inputs), Variable(targets)
			outputs = net(inputs)
			if args.loss_fn is None:
				loss = criterion(outputs, targets)
			else:
				loss = criterion(outputs, targets, epoch)

			if args.save_val_scores:
				p_out = F.softmax(outputs,dim=1)
				val_softmax_scores.append(p_out.data)

			loss_acc = np.append(loss_acc, loss.data.item())
			test_loss += loss.data.item()
			_, predicted = torch.max(outputs.data, 1)
			# if epoch >= args.learn_epochs-1:
			# 	pdb.set_trace()
			total += targets.size(0)
			correct += predicted.eq(targets.data).cpu().sum().data.item()
			abstain += predicted.eq(abstain_class_id).sum().data.item()
		validation_loss = np.append(validation_loss, np.mean(loss_acc))
		if args.save_val_scores:
			val_scores = torch.cat(val_softmax_scores).cpu().numpy()

			print('Saving softmax scores at Validation Epoch %d' %(epoch))
			fn = args.expt_name if args.expt_name else 'test'
			#np.save(fn+".train_scores.epoch_"+str(epoch), train_scores)
			np.save(args.output_path+fn+"/"+ fn +".val_scores.epoch_"+str(epoch), val_scores)

		#pdb.set_trace()
		acc = 100.*correct/float(total)
		if total-abstain != 0:
			abst_acc = 100.*correct/(float(total-abstain))
		else:
			abst_acc = 100.

		print("\n| Validation Epoch #%d\t\t\tAbstained: %d Loss: %.4f Acc@1: %.2f%% Acc@2: %.2f%% " %(epoch, abstain, test_loss/(batch_idx+1), acc,abst_acc))

	    #return

	    # Save checkpoint when best model

		if acc > best_acc or epoch == args.save_epoch_model:# or (int(epoch/args.epdl) > 60 and int(epoch/args.epdl) <= 80):
			
			if args.save_best_model:
				print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
				state = {
				        'net':net,
				        'acc':acc,
				        'epoch':epoch,
				}
				if not os.path.isdir('checkpoint'):
				    os.mkdir('checkpoint')
				save_point = './checkpoint/'+args.dataset+os.sep
				if not os.path.isdir(save_point):
				    os.mkdir(save_point)
				#torch.save(state, save_point+file_name+'_rand_label_'+str(args.rand_labels)+'_epoch_'+str(epoch)+'_081318.t7')
				if args.expt_name == "":
					if not args.log_file is None:
						expt_name = os.path.basename(args.log_file).replace(".log","")
					else:
						expt_name = 'test' #assuming that if a log file has not been specified this is a test run.
				else:
					expt_name = args.expt_name
				if args.no_overwrite:
					torch.save(state, save_point+file_name+'_expt_name_'+str(expt_name)+'_epoch_'+str(epoch)+'.t7')
				else:
					torch.save(state, save_point+file_name+'_expt_name_'+str(expt_name)+'.t7')
		if acc > best_acc:
			best_acc = acc


def createDir(dirName):
    try:
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")
    
print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
sys.stdout.flush()

#print('| Optimizer = ' + str(optim_type))

training_loss = np.array([])
validation_loss = np.array([])

if args.expt_name:
    expDirName = args.output_path+args.expt_name
    lossValueDirName = args.output_path+args.expt_name + "/loss"
    createDir(expDirName)
    createDir(lossValueDirName)

elapsed_time = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()

    train(epoch)
    if args.save_train_scores:
    	save_train_scores(epoch)
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(get_hms(elapsed_time)))
    sys.stdout.flush()

print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = %.2f%%' %(best_acc))

print('\n[Phase 5] : Saving losses')			
fn = args.expt_name if args.expt_name else 'test'
np.save(args.output_path+fn+"/loss/"+ fn +".val-loss", validation_loss.flatten())
np.save(args.output_path+fn+"/loss/"+ fn +".train-loss", training_loss.flatten())


'''
from utils import ucr_archive
from torch.utils.data import DataLoader


if __name__ == '__main__':

    dataset = ucr_archive.UCRArchive('data/UCRArchive_2018', 'SmoothSubspace')
    dataloader = DataLoader(dataset, batch_size=120, shuffle=True, num_workers=2)
    for i, batch in enumerate(dataloader):
        #print(i, batch)
        print(batch[0].shape)
'''
        