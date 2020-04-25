# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 22:53:41 2020

@author: User
"""
import os
import time
import shutil
import argparse
import torch.utils.data
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet

import datetime


parser = argparse.ArgumentParser(description = 'ResNet101 with CIFAR10 PyTorch')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
#parser.add_argument('--resume', '-r', action = 'store_true', help = 'resume from checkpoint')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)

def main():
    global args, best_prec
    args = parser.parse_args()
    
    best_prec = 0
    
    #check if save directory exist or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    #check if gpu or cpu
    if torch.cuda.is_available():
        device = 'cuda'
        print('Use GPU')
    else:
        device = 'cpu'
        print('Use CPU')
        
    #load model ResNet101 to device
    model = resnet.ResNet101()
    model = model.to(device)
   
    #can add code to use multi GPU here
    print('Loaded model to device')
    
    #add code here to resume from a checkpoint
        
    #preparing CIFAR 10 dataset
#    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])
    
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
    
    #setting for for training dataset
    train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root = './data', train = True, transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                    ]), download = True),
            batch_size = args.batch_size, shuffle = True,
            num_workers = args.workers, pin_memory = True)
        
    #setting for validation dataset
    val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root = './data', train = False, transform = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                    ])),
        batch_size = 128, shuffle = False,
        num_workers = args.workers, pin_memory = True)
        
        
    #define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    #define optimizer used
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [100, 150], last_epoch = args.start_epoch - 1)
    
    
    
    #for validation process
    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    
    
    #print start time for taining
    starttime = datetime.datetime.now()
    print ('start time: ', str(starttime))
    
    #for training process
    for epoch in range(args.start_epoch, args.epochs):
        
        #for each epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()
        
        #after one epoch, evaluate using validation set
        prec = validate(val_loader, model, criterion)
        
        #save the best prec 
        is_best_prec = prec > best_prec
        best_prec = max(prec, best_prec)
        
        #save checkpoint
        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(), 
                             'best_prec': best_prec,},
         is_best_prec, filename = os.path.join(args.save_dir, 'checkpoint_resnet101.th'))
            
        save_checkpoint({'state_dict': model.state_dict(), 
                         'best_prec': best_prec,},
             is_best_prec, filename = os.path.join(args.save_dir, 'model_resnet101.th'))
        
    #print end time for taining
    endtime = datetime.datetime.now()
    print ('end time: ', str(endtime))
        
#define the training process for one epoch
def train(train_loader, model, criterion, optimizer, epoch):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    #put model ResNet101 in train mode
    model.train()
    
    end_time = time.time()
    for i, (input, target) in enumerate(train_loader):
        
        #measure loading time
        data_time.update(time.time() - end_time)
        
        #target = target.cuda()
        #input_var = input.cuda()
        target = target
        input_var = input
        target_var = target
        
        #compute output - need to load input to model ResNet101
        output = model(input_var) #load input to model
        
        #calculate loss - compute difference targeted output with actual output (output)
        loss = criterion(output, target_var)
        
        
        #compute gradient and perform SGD step
        optimizer.zero_grad() #reset gradient to zero first 
        loss.backward() #compute gradients in backward pass
        optimizer.step() #update values of net.params() with the computed gradients
        
        output = output.float()
        loss = loss.float()
        
        #measure network accuracy and record loss
        prec = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))
        
        #measure elapsed time
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        
        #print information for training of one epoch
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec_top1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          epoch, i, len(train_loader), batch_time = batch_time,
                          data_time = data_time, loss = losses, top1 = top1))
            
            

#validation process (after training)
def validate(val_loader, model, criterion):
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    #put model ResNet101 in evaluation (validation) mode
    model.eval()
    
    end_time = time.time()
    #do not generate gradient in evaluation mode
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            
#             target = target.cuda()
#             input_var = input.cuda()
#             target_var = target.cuda()
             
             target = target
             input_var = input
             target_var = target
             
             #compute output for input
             output = model(input_var)
             
             #calculate loss - compute difference targeted output with actual output (output)
             loss = criterion(output, target_var)
             
             #measure network accuracy and record loss
             prec = accuracy(output.data, target)[0]
             losses.update(loss.item(), input.size(0))
             top1.update(prec.item(), input.size(0))
             
             #measure elapse time
             batch_time.update(time.time() - end_time)
             end_time = time.time()
             
             #print information for training of one epoch
             if i % args.print_freq == 0:
                 print('Test: [{0}/{1}]\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                       'Prec_top1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time = batch_time,
                          loss = losses, top1 = top1))
                 
    print(' *Prec_top1 {top1.avg:.3f}'.format(top1 = top1))
    
    return top1.avg


#function to save the trained model.
def save_checkpoint(state, is_best, filename = 'checkpoint.pth.tar'):
    
    torch.save(state, filename)
    


#class to store the average and current value
class AverageMeter(object):
    def __init__(self):
        self.reset()
        print('reset')
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
        
    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count = self.count + n
        self.avg = self.sum / self.count
        


#computes precision for specified k
def accuracy(output, target, topk = (1, )):
     
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



if __name__ == '__main__':
    main()

    
        
    
        
    
    
    




    
    

    

    
        
    


