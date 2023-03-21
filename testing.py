
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



def test(args,model,dataLoader):
    if args.arch=='EventNetSeg':
        test_eventnet(model,dataLoader)
    else:
        test_pointnet(model,dataLoader)


def test_eventnet(model,dataLoader):
    model.eval()
 
    correct_all =0
    total = 0
    #--------------generate the table---------------------
    x_LUT, fea_LUT = model.getLUT(128,128)
    event, seg = dataLoader.test_getallpoints()
    with tqdm(total=event.shape[0],postfix=dict,mininterval=0.3) as pbar:
        last_code=None
        last_t = None
        pred_list=[]
        for index,point in enumerate(event) :
            xyp = torch.tensor(point[:3]).float().unsqueeze(0).unsqueeze(2).cuda()
            x=  int(point[0])
            y = int(point[1])
            p  =int(point[2])
            delta_t = torch.tensor([point[-1]]).float().unsqueeze(0).unsqueeze(2).cuda()
            
            #--------------look up the table---------------------
            x1024 = x_LUT[p,y,x,:,:].unsqueeze(0)
            feat = fea_LUT[p,y,x,:,:].unsqueeze(0)
            
            
            pred, this_code ,this_t= model(xyp,delta_t, x1024, feat,last_code,last_t)
            pred = pred.view(-1, 2)
            pred_choice = pred.data.max(1)[1]
            seg_i = seg[:index+1, 0]
            pred_list.append(pred_choice[0].cpu())
            last_code =this_code
            last_t = this_t
            
            # dataLoader.plot(event[:index+1],seg[:index+1])
            # dataLoader.plot(event[:index+1],np.array(pred_list).reshape(-1,1))
            correct =np.array(np.array(pred_list)==(seg_i),dtype=np.int).sum()
            total += (index+1)
            correct_all += correct
            pbar.set_postfix(**{'ag': correct_all/total})
            pbar.update(1)
    print("AG:",(correct)/total)

def test_pointnet(model,dataLoader):
    model.eval()
 
    correct_all =0
    total = 0
    with tqdm(total=100,postfix=dict,mininterval=0.3) as pbar:
        for i in range(100):
            
            event, seg = dataLoader.test_getbatchpoints(i)
            total_num = event.shape[0]
        
            event=torch.tensor(event).float().unsqueeze(0).transpose(2,1).cuda()
            seg = torch.tensor(seg).unsqueeze(0).transpose(2,1).cuda()
            pred, trans, trans_feat = model(event)
            pred = pred.view(-1, 2)
            seg = seg.view(-1, 1)[:, 0]
            pred_choice = pred.data.max(1)[1]

            dataLoader.plot(event[0].transpose(1,0).cpu(),seg.unsqueeze(1).cpu())
            dataLoader.plot(event[0].transpose(1,0).cpu(),pred_choice.unsqueeze(1).cpu())
            correct = pred_choice.eq(seg.data).cpu().sum()
            total += total_num
            correct_all += correct.item()
            pbar.set_postfix(**{'ag': correct.item()/total_num})
            pbar.update(1)
    print("AG:",(correct_all)/total)