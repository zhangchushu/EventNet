import numpy as np
import torch
import torch.nn as nn
import cv2 as cv
import matplotlib.pyplot as plt
from models.pointnet import PointNetDenseCls, feature_transform_regularizer
from models.eventnet import EventNetSeg
from dataLoaderETH import eventData
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader
from testing import test
from training import train
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=bool, default=True, help="use cuda or not")
parser.add_argument("--load_pth", type=bool, default=False, help="load pretrain model or not")
parser.add_argument("--arch",type=str,default='EventNetSeg2',choices=['EventNetSeg','PointNetDenseCls'])
parser.add_argument("--datapath",type=str,default='shapes_rotation', help="dataset path")
parser.add_argument("--batchsize",type=int,default=12, help="the training batchsize")
parser.add_argument("--batchpoints",type=int,default=8000,help='the number of points in one training bacth')
parser.add_argument("--epoch1",type=int,default=100,help='learning rate decays before the epoch')
parser.add_argument("--epoch2",type=int,default=500,help='the whole traning epoch')
parser.add_argument("--istrain",type=bool,default=False)
parser.add_argument("--istest",type=bool,default=True)

args = parser.parse_args()

if __name__ == '__main__':
    """----------------loading dataset---------------------------"""
    W, H, tau = 240, 180, 32
    dataPath = args.datapath
    dataLoader = eventData('cpu', dataPath, W, H,args.batchpoints, tau,args.batchsize)
    trainloader = DataLoader(dataLoader , batch_size=12, shuffle=False, num_workers=0, pin_memory=True)


    """----------------Pointnet or Eventnet-----------------------"""
    if args.arch=='EventNetSeg':
        model = EventNetSeg(k=2)    
    else:
        model = PointNetDenseCls(k=2, feature_transform=False)
    device = torch.device('cuda:0' if args.cuda else 'cpu')
    model.to(device)
    if args.load_pth:
        checkpoint = torch.load('eventnet_ckpt.pth', map_location = device)
        model.load_state_dict(checkpoint)


    """-----------------optimizer settings------------------------"""
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    
    
    

    if args.istrain:
        for epoch in range(args.epoch1):
            train(args,epoch,model,trainloader,optimizer)
            scheduler.step()
        for epoch in range(args.epoch1,args.epoch2):
            train(args,epoch,model,trainloader,optimizer)
    if args.istest:
        test(args,model,dataLoader)