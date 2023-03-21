import numpy as np
import torch
import torch.nn as nn
import cv2 as cv
import matplotlib.pyplot as plt
from models.pointnet import PointNetDenseCls, feature_transform_regularizer
from dataLoaderETH import eventData
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader


def train(args,epoch,model,trainloader,optimizer):
    if args.arch=='EventNetSeg':
        train_eventnet(args,epoch,model,trainloader,optimizer)
    else:
        train_pointnet(args,epoch,model,trainloader,optimizer)




def train_pointnet(args,epoch,model,trainloader,optimizer):
    model.train()
    correct_all =0
    total = 0
    losses =0
    with tqdm(total=100, desc=f'Epoch:{epoch + 1}/{args.epoch2}',postfix=dict,mininterval=0.3) as pbar:
        for i, (event, seg,num_points) in enumerate(trainloader):
            event= event.transpose(2,1).cuda()
            seg = seg.transpose(2,1).cuda()
            pred, _, _ = model(event)
            pred = pred.view(-1, 2)
            seg = seg.view(-1, 1)[:, 0]

            loss = F.nll_loss(pred, seg)
            loss.backward()
            optimizer.step()

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(seg.data).cpu().sum()
            losses +=loss.item()
            total +=  args.batchpoints * args.batchsize
            correct_all += correct.item()
            pbar.set_postfix(**{'loss': losses/(i+1),'ac':correct_all/total,'lr':(optimizer.state_dict()['param_groups'][0]['lr'])})
            pbar.update(1)
     
def train_eventnet(args,epoch,model,trainloader,optimizer):
    model.train()
    correct_all =0
    total = 0
    losses =0
    with tqdm(total=100, desc=f'Epoch:{epoch + 1}/{args.epoch2}',postfix=dict,mininterval=0.3) as pbar:
        for i, (event, seg,num_points) in enumerate(trainloader):
            
            event= event.transpose(2,1).cuda()
            xyp = event[:,:3,:]
            delta_t = event[:,-1:,:]
            seg = seg.transpose(2,1).cuda()
            pred, _, _ = model(xyp,delta_t,None,None,None,None)
            pred = pred.view(-1, 2)
            # dataloader.plot(event[0].transpose(1,0).cpu(),seg[0].transpose(1,0).cpu())
            seg = seg.view(-1, 1)[:, 0]

            loss = F.nll_loss(pred, seg)
            loss.backward()
            optimizer.step()

            pred_choice = pred.data.max(1)[1]
            # dataloader.plot(event[0].transpose(1,0).cpu(),pred_choice.reshape(12,-1)[0].unsqueeze(1).cpu())
            correct = pred_choice.eq(seg.data).cpu().sum()


            losses +=loss.item()
            total += args.batchpoints * args.batchsize
            correct_all += correct.item()
            pbar.set_postfix(**{'loss': losses/(i+1),'ac':correct_all/total,'lr':(optimizer.state_dict()['param_groups'][0]['lr'])})
            pbar.update(1)

        
