# dataLoaderETH.py
#
# Sample code for load and visualizing additional semantic-segmentation label
# for "shapes_translation" sequence in ETHTED.

# See readme.txt for detail.
#
# Y.Sekikawa 2018/03/29
from torch.utils.data import Dataset, DataLoader
import math 
import matplotlib.pyplot as pl
import torch
import numpy as np
import os
import random

dtype = torch.float32
itype = torch.long


class eventData(Dataset):
    def __init__(self, device, dataPath, W, H, N,tau,batchsize, bSort=False):
        # self.init0()
        self.tau = tau
        self.temporal_res = 1000
        self.W = W
        self.H = H
        self.N = N
        self.bSort = bSort
        self.device = device
        # self.labelType = ['gt', 'seg', 'imu']
        self.labelType = ['seg']
        self.len = batchsize * 100
        if os.path.exists(os.path.join(dataPath , 'event_cashe.npy')):
            print('Resuming from cashe')
            event = np.load(os.path.join(dataPath , 'event_cashe.npy'))
        else:
            print('Loading from txt')
            event = np.loadtxt(os.path.join(dataPath , 'events.txt'))
            np.save(os.path.join(dataPath ,'event_cashe.npy'), event)


        if  os.path.exists(os.path.join(dataPath , 'seg_cashe.npy')):
            print('Resuming from cashe')
            seg = np.load(os.path.join(dataPath , 'seg_cashe.npy'))
        else:
            print('Loading from txt')
            seg = np.loadtxt(os.path.join(dataPath , 'segmentation.txt')) if 'seg' in self.labelType else []
            np.save(os.path.join(dataPath ,'seg_cashe.npy'), seg)

        gt = np.loadtxt(os.path.join(dataPath , 'groundtruth.txt')) if 'gt' in self.labelType else []
        imu = np.loadtxt(os.path.join(dataPath ,'imu.txt')) if 'imu' in self.labelType else []
        self.event, self.gt, self.imu, self.seg = event, gt, imu, seg

    def getTimeRangeTrain(self):
        start_time = np.min(self.event[:, 0])
        end_time = 50.0 - self.tau/self.temporal_res
        return [start_time, end_time]

    def getTimeRangeTest(self):
        start_time = 50.0
        end_time = np.max(self.event[:, 0]) - self.tau/self.temporal_res
        return [start_time, end_time]

    def cropAt(self, t_s):
        t_e = t_s + self.tau/self.temporal_res
        i_s = np.nonzero(self.event[:, 0] > t_s)[0][0]  #np.nonzero get the first t_s index
        i_e = np.nonzero(self.event[:, 0] > t_e)[0][0]-1
        self.t_s = t_s
        w_start = math.floor( (self.W -128) *random.random())
        h_start =  math.floor((self.H-128) * random.random())
        w_end = w_start + 128
        h_end = h_start +128
        wp = (self.event[i_s:i_e, 1] >= float(w_start)) & (self.event[i_s:i_e, 1] < float(w_end))
        hp = (self.event[i_s:i_e, 2] >= float(h_start)) & (self.event[i_s:i_e, 2] < float(h_end))
       
        t = np.round(self.event[i_s:i_e][wp & hp, 0] * self.temporal_res)   #get the time sequence
        
        x = self.event[i_s:i_e][wp & hp, 1] - w_start
        y = self.event[i_s:i_e][wp & hp, 2] - h_start
        self.event_ = np.stack((x,y, self.event[i_s:i_e][wp & hp, 3], t-t[0]),  axis=1)
        self.seg_ = np.array(self.seg[i_s:i_e][wp & hp].reshape([-1,1])==2,dtype=np.int16)

        return torch.tensor(self.event_).type(dtype).to(self.device), \
               torch.tensor(self.seg_).type(itype).to(self.device)
                    #    torch.tensor(self.gt_).type(dtype).to(self.device), \
            #    torch.tensor(self.imu_).type(dtype).to(self.device), \

    def test_getbatchpoints(self, i):
        
        t_s = 50 + i*self.tau/self.temporal_res
        t_e = t_s + self.tau/self.temporal_res
        i_s = np.nonzero(self.event[:, 0] > t_s)[0][0]  #np.nonzero get the first t_s index
        i_e = np.nonzero(self.event[:, 0] > t_e)[0][0]-1
        self.t_s = t_s

        w_start = 56
        h_start = 26
        w_end = w_start + 128
        h_end = h_start +128
        wp = (self.event[i_s:i_e, 1] >= float(w_start)) & (self.event[i_s:i_e, 1] < float(w_end))
        hp = (self.event[i_s:i_e, 2] >= float(h_start)) & (self.event[i_s:i_e, 2] < float(h_end))
       
        t = np.round(self.event[i_s:i_e][wp & hp, 0] * self.temporal_res)   #get the time sequence

        x = self.event[i_s:i_e][wp & hp, 1] - w_start
        y = self.event[i_s:i_e][wp & hp, 2] - h_start
        self.event_ = np.stack((x,y, self.event[i_s:i_e][wp & hp, 3], t-t[0]),  axis=1)
        self.seg_ = np.array(self.seg[i_s:i_e][wp & hp].reshape([-1,1])==2,dtype=np.int16)

        return self.event_, self.seg_
    
    def test_getallpoints(self):
        i_s = np.nonzero(self.event[:, 0] > 50)[0][0]  

        w_start = 56
        h_start = 26
        w_end = w_start + 128
        h_end = h_start +128
        wp = (self.event[i_s:, 1] >= float(w_start)) & (self.event[i_s:, 1] < float(w_end))
        hp = (self.event[i_s:, 2] >= float(h_start)) & (self.event[i_s:, 2] < float(h_end))
       
        t = np.round(self.event[i_s:][wp & hp, 0] * self.temporal_res)   #get the time sequence

        x = self.event[i_s:][wp & hp, 1] - w_start
        y = self.event[i_s:][wp & hp, 2] - h_start
        self.event_ = np.stack((x,y, self.event[i_s:][wp & hp, 3], t-t[0]),  axis=1)
        self.seg_ = np.array(self.seg[i_s:][wp & hp].reshape([-1,1])==2,dtype=np.int16)

        return self.event_, self.seg_


    def __getitem__(self, index):
        [start_time, end_time] = self.getTimeRangeTrain()
        t_s = np.random.uniform(start_time, end_time, 1)
        t_e = t_s + self.tau/self.temporal_res
        i_s = np.nonzero(self.event[:, 0] > t_s)[0][0]  #np.nonzero get the first t_s index
        i_e = np.nonzero(self.event[:, 0] > t_e)[0][0]-1
        self.t_s = t_s
        w_start = math.floor( (self.W -128) *random.random())
        h_start =  math.floor((self.H-128) * random.random())
        w_end = w_start + 128
        h_end = h_start +128
        wp = (self.event[i_s:i_e, 1] >= float(w_start)) & (self.event[i_s:i_e, 1] < float(w_end))
        hp = (self.event[i_s:i_e, 2] >= float(h_start)) & (self.event[i_s:i_e, 2] < float(h_end))
       
        t = np.round(self.event[i_s:i_e][wp & hp, 0] * self.temporal_res)   #get the time sequence
        
        x = self.event[i_s:i_e][wp & hp, 1] - w_start
        y = self.event[i_s:i_e][wp & hp, 2] - h_start
        self.event_ = np.stack((x,y, self.event[i_s:i_e][wp & hp, 3], t-t[0]),  axis=1)
        self.seg_ = np.array(self.seg[i_s:i_e][wp & hp].reshape([-1,1])==2,dtype=np.int16)

        num_points = self.event_.shape[0]
        if num_points<=self.N:
            self.event_= np.vstack((self.event_,np.zeros((self.N-num_points,4))))
            self.seg_ =np.vstack((self.seg_,np.zeros((self.N-num_points,1))))
        else:
             n = np.linspace(0,num_points-1,self.N,dtype=int)
             self.event_= self.event_[n,:]
             self.seg_ = self.seg_[n,:]
        return torch.tensor(self.event_).type(dtype).to(self.device), \
               torch.tensor(self.seg_).type(itype).to(self.device),num_points
    

    
    def plot(self,event_,seg_):
        col_r = [1.0, 0.0, 0.0]
        col_g = [0.0, 1.0, 0.0]
        col_b = [0.0, 0.0, 1.0]
        col_w = [1.0, 1.0, 1.0]

        x = event_[:, 0]
        y = event_[:, 1]
        p = event_[:, 2]
        t = event_[:, 3]
        s = seg_[:, 0] 
        t = (t / self.tau)
        print(str(len(t) / self.tau) + 'KEPS')

        # Plot
        fig = pl.figure()
        if 1:
            ax = pl.subplot(1, 2, 1)
            ax.scatter(x[p == 1], y[p == 1],
                       color=np.multiply(t[p == 1].reshape([-1, 1]), np.array(col_g).reshape([1, 3])), s=1)
            ax.scatter(x[p == 0], y[p == 0],
                       color=np.multiply(t[p == 0].reshape([-1, 1]), np.array(col_r).reshape([1, 3])), s=1)
            pl.gca().invert_yaxis()
            pl.xlim(0, self.W)
            pl.ylim(0, self.H)
            ax.set_aspect('equal')
            ax.axis('off')
            pl.title('Porality, t:')

        if 'seg' in self.labelType:
            ax = pl.subplot(1, 2, 2)
            ax.scatter(x[s == 1], y[s == 1],
                       color=np.multiply(t[s == 1].reshape([-1, 1]), np.array(col_b).reshape([1, 3])), s=1)
            ax.scatter(x[s <= 0], y[s <= 0],
                       color=np.multiply(t[s <= 0].reshape([-1, 1]), np.array(col_w).reshape([1, 3])), s=1)
            
            pl.gca().invert_yaxis()
            pl.xlim(0, self.W)
            pl.ylim(0, self.H)
            ax.set_aspect('equal')
            ax.axis('off')
            pl.title('Segmentation, t:')

        # pl.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        # # fig.tight_layout()
        pl.show()
    def __len__(self):
        return self.len

if __name__ == '__main__':
    # dataPath = '/Users/sekikawayuusuke/Downloads/shapes_rotation/'
    dataPath = r'D:\myproject\eventnet\shapes_rotation'

    W, H, tau = 240, 180, 32.
    dataLoader = eventData('cpu', dataPath, 240, 180, tau)

    [start_time, end_time] = dataLoader.getTimeRangeTrain()
    
    for _ in range(100):
        event_,seg_ = dataLoader.test_getpoints(_)
        t = np.random.uniform(start_time, end_time, 1)
        event, seg = dataLoader.cropAt(t)
        # dataLoader.plot(event_,seg_)
