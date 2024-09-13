# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:17:35 2024

@author: aszor
"""

from abc import abstractmethod
import psutil
import argparse
import itertools
import datetime
import math
import numpy as np
import os
import sys
import time
import pandas as pd
import warnings
import torch
from viz_dataset import load_data
import matplotlib.pyplot as plt
from bentley_ottmann.planar import segments_intersect
from ground.base import get_context
import model_init
from train_stpp import cast
from scipy.ndimage import gaussian_filter


class MarkovModel:
    def __init__(self,cond_dist,seed=222,predname="markov"):
        self.dist = cond_dist
        self.rng = np.random.default_rng(seed)
        self.predname=predname
        self.currx=None
        self.curry=None
        self.currt=None

    def reset(self):
        self.currx=None
        self.curry=None
        self.currt=None

    def model_update(self,t,x,y):
        self.currx=x
        self.curry=y
        self.currt=t

    def get_predictors(self,t,steps,logit=True):
        #pass on to distribution object
        predarr = self.dist.get_predictors(self.currx,self.curry,t,steps)

        #pred = np.log(predarr)
        if logit:
           P =  predarr / sum(predarr)
           pred = np.log(P/(1-P))
        else:
           pred = predarr
           
        df = pd.DataFrame()
        df[self.predname] = pred
        return df
    

class ParametricModel:
    def __init__(self,uncond_dist,cols=['logdist','cardinality','dir_change','wall_dist','wall_dist_change','crossings','closest_point','dir_moment'],seed=333,maxtime_cross=13,maxtime_closest=17,maxtime_dir=2.5,col_t=False):
        self.allsteps = []
        self.allangs = []
        self.basedist = uncond_dist
        self.rng = np.random.default_rng(seed)
        self.currwd = None
        self.currang = None
        self.maxtime_cross = maxtime_cross
        self.maxtime_closest = maxtime_closest
        self.maxtime_dir = maxtime_dir
        self.cols = cols
        self.col_t = col_t
        
    def reset(self):
        self.allsteps = []
        self.allangs = []
            
    def model_update(self,t,x,y):
        self.allsteps.append([t,x,y])
        self.currwd = self.walldist(x,y)
        if len(self.allsteps)>1:
            dx = self.allsteps[-1][1]-self.allsteps[-2][1]
            dy = self.allsteps[-1][2]-self.allsteps[-2][2]
            self.currang = np.arctan2(dy,dx)
            self.allangs.append(self.currang)
        
    def walldist(self,x,y):
        return min([abs(x-self.basedist.extent[0]),abs(x-self.basedist.extent[1]),abs(y-self.basedist.extent[2]),abs(y-self.basedist.extent[3])])

    def closest_point(self,x0,y0,t):
        n = len(self.allsteps)
        dists = []
        for i in range(1,n):
            if len(dists)>0 and t-self.allsteps[n-i-1][0]>self.maxtime_closest:
                break
            x1 = self.allsteps[n-i-1][1]
            y1 = self.allsteps[n-i-1][2]
            dists.append(np.sqrt((x1-x0)**2+(y1-y0)**2))

        return min(dists)
    
    def crossings(self,x0,y0,t):
        x1 = self.allsteps[-1][1]
        y1 = self.allsteps[-1][2]
        context = get_context()
        Point, Segment = context.point_cls, context.segment_cls

        crossings = 0
        n = len(self.allsteps)
        for i in range(2,n):
            if t-self.allsteps[n-i][0]>self.maxtime_cross:
                break
            x2 = self.allsteps[n-i][1]
            x3 = self.allsteps[n-i-1][1]
            y2 = self.allsteps[n-i][2]
            y3 = self.allsteps[n-i-1][2]
            unit_segments = [Segment(Point(x0, y0), Point(x1, y1)), Segment(Point(x2, y2), Point(x3, y3))]
            if segments_intersect(unit_segments):
                crossings += 1
                
        return crossings
    
    def dir_moment(self,x0,y0,t):
        n = len(self.allsteps)
        xs = []
        ys = []
        dx = x0-self.allsteps[-1][1]
        dy = y0-self.allsteps[-1][2]

        for i in range(1,n):
            if len(xs)>0 and t-self.allsteps[n-i-1][0]>self.maxtime_dir:
                break
            xs.append(np.cos(self.allangs[n-i-1]))
            ys.append(np.sin(self.allangs[n-i-1]))
            
        totx = sum(xs)
        toty = sum(ys)
        
        return np.sqrt(totx**2 + toty**2)/len(xs)*np.cos(np.arctan2(dy,dx)-np.arctan2(toty,totx))
    
    def get_predictors(self,t,steps):
        predarr = []
        
        for [x,y] in steps:
           preds = {}
           
           dx = x-self.allsteps[-1][1]
           dy = y-self.allsteps[-1][2]
           dist = np.sqrt(dx**2+dy**2)
           if "logdist" in self.cols:
              preds["logdist"] = np.log(dist)
           if "cardinality" in self.cols:
              preds["cardinality"] = np.cos(4*np.arctan2(dy,dx))
           if "dir_change" in self.cols:
              preds["dir_change"] = np.cos(np.arctan2(dy,dx)-self.currang)
           if "wall_dist" in self.cols:
              wd = self.walldist(x,y)
              preds["wall_dist"] = wd
              if "wall_dist_change" in self.cols:
                 preds["wall_dist_change"] = wd - self.currwd
           if "crossings" in self.cols:
              name="crossings"
              if self.col_t:
                 name += "_"+str(self.maxtime_cross)    
              preds[name] = self.crossings(x,y,t)
           if "closest_point" in self.cols:
              name="closest_point"
              if self.col_t:
                 name += "_"+str(self.maxtime_closest)    
              preds[name] = self.closest_point(x,y,t)
           if "dir_moment" in self.cols:
              name="dir_moment"
              if self.col_t:
                 name += "_"+'{:.1f}'.format(self.maxtime_dir).replace('.','_')    
              preds[name] = self.dir_moment(x,y,t)
           predarr.append(preds)
           
        pred_df = pd.DataFrame(predarr)
        return pred_df
        
class NNModel:
    def __init__(self,model_id,maxhist=20,n_iter=20,predname="nn"):
        self.model = self.load_model(model_id)
        self.max_hist = maxhist
        self.predname = predname
        self.n_iter = n_iter
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reset()

    def load_model(model_id):
        x_dim = 2
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpt_dir = os.path.join('dl',model_id)
        checkpt_path = os.path.join(checkpt_dir,'model.pth')
        model_args = model_init.get_model_args(os.path.join(checkpt_dir,'args.txt'))

        nnmodel,optimizer,ema = model_init.model_init(model_args,x_dim,device)

        checkpt = torch.load(checkpt_path, device)
        nnmodel.load_state_dict(checkpt["state_dict"])

    def reset(self):
        self.event_times = cast(torch.empty([1,0]),self.device)
        self.spatial_locations = cast(torch.empty([1,0,2]),self.device)
            
    def model_update(self,t,x,y):
        self.event_times = torch.cat((self.event_times,cast(torch.tensor([[t]]),self.device)),1)
        self.spatial_locations = torch.cat((self.spatial_locations,cast(torch.tensor([[[x,y]]]),self.device)),1)

    
    def get_predictors(self,t,steps,logit=True):

        loglik_arr = []
        
        i = self.event_times.size()[1]
        with torch.no_grad():
            if i>self.max_hist:
               t0 = self.event_times[0,i-self.max_hist]
               likfun = self.model.spatial_conditional_logprob_fn(t-t0, self.event_times[0,(i-self.max_hist):]-t0, self.spatial_locations[0,(i-self.max_hist):,:],0,None)              
            else:
               likfun = self.model.spatial_conditional_logprob_fn(t, self.event_times[0,:], self.spatial_locations[0,:,:],0,None)
        for [x,y] in steps:
            S = np.stack([x*np.ones(self.n_iter), y*np.ones(self.n_iter)], axis=1)
            S = torch.tensor(S).to(self.device).float()

            with torch.no_grad():
               loglik = likfun(S).detach().cpu().numpy()
            loglik_arr.append(np.mean(loglik))
        
        P = np.exp(np.array(loglik_arr)) #remove log
        if logit:
           P = P / sum(P) #normalize to sum of 1
           pred = np.log(P/(1-P)) #logit
        else:
           pred = P
           
        return pd.DataFrame(pred,columns=[self.predname])



#Super class for all histogram-based data representations.
#Sub-classes must have a get_distribution() method which takes any extra init() arguments
#Optional methods include get_candidates(x,y,n) for sampling of n possible steps at time t
#And get_predictors(x,y,t,steps) for Markovian step selection
class Distribution:
    def __init__(self,dataset,extent,seed=111,nhist_x=50,nhist_y=50,nhist_dist=50,nhist_ang=25,**kwargs):
        self.rng = np.random.default_rng(seed)
        self.extent = extent
        self.cellsize_x = (extent[1]-extent[0])/nhist_x
        self.cellsize_y = (extent[3]-extent[2])/nhist_y
        self.x_edges = np.linspace(extent[0],extent[1],nhist_x+1)
        self.y_edges = np.linspace(extent[2],extent[3],nhist_y+1)
        self.dx_edges = np.linspace(-(extent[1]-extent[0]),(extent[1]-extent[0]),2*nhist_x+1)
        self.dy_edges = np.linspace(-(extent[3]-extent[2]),(extent[3]-extent[2]),2*nhist_y+1)
        meshx,meshy = np.meshgrid(self.x_edges[1:],self.y_edges[1:])
        self.mx = meshx.flatten()
        self.my = meshy.flatten()
        meshdx,meshdy = np.meshgrid(self.dx_edges[1:],self.dy_edges[1:])
        self.mdx = meshdx.flatten()
        self.mdy = meshdy.flatten()
        self.maxdist = 0.5*np.sqrt((self.extent[1]-self.extent[0])**2+(self.extent[3]-self.extent[2])**2)
        self.dist_edges = np.arange(0,nhist_dist+1)*self.maxdist/nhist_dist
        self.ang_edges = np.arange(0,nhist_ang+1)*2*np.pi/nhist_ang - np.pi
        self.distribs = self.get_distribution(dataset,**kwargs)
        self.cumdistribs = [np.cumsum(x.T.flatten()) for x in self.distribs]

    @abstractmethod    
    def get_distribution(dataset,**kwargs):
        #returns a list of distributions
        pass

class Conditional_XY(Distribution):
    #P(x(t+1),y(t+1)|x(t),y(t))
    def __init__(self,dataset,extent,seed=111,**kwargs):
        super().__init(dataset,extent,**kwargs)
        
    def get_distribution(self,dataset,sigma=4):
        
        h = np.zeros([len(self.x_edges)-1,len(self.y_edges)-1,len(self.x_edges)-1,len(self.y_edges)-1])
        
        for data in dataset:
            for t in range(1,data.shape[0]):
               px = np.argmax(self.x_edges>data[t-1,1])
               py = np.argmax(self.y_edges>data[t-1,2])
               cx = np.argmax(self.x_edges>data[t,1])
               cy = np.argmax(self.y_edges>data[t,2])
               if px>0 and py>0 and cx>0 and cy>0:
                  h[cx-1,cy-1,px-1,py-1] += 1
        
        #smooth
        h_smooth = gaussian_filter(h,sigma)
        
        #normalize
        for i in range(len(self.x_edges)-1):
            for j in range(len(self.y_edges)-1):
                h_smooth[:,:,i,j] = h_smooth[:,:,i,j] / sum(h_smooth[:,:,i,j].flatten())
                
        return [h_smooth]
    
class Conditional_dXdY(Distribution):
    #P(dx(t),dy(t)|x(t),y(t))
    def __init__(self,dataset,extent,**kwargs):
        super().__init__(dataset,extent,**kwargs)
        
    def get_distribution(self,dataset,sigma=5):        
        h = np.zeros([len(self.dx_edges)-1,len(self.dy_edges)-1,len(self.x_edges)-1,len(self.y_edges)-1])
        
        for data in dataset:
            for t in range(1,data.shape[0]):
               px = np.argmax(self.x_edges>data[t-1,1])
               py = np.argmax(self.y_edges>data[t-1,2])
               dx = np.argmax(self.dx_edges>(data[t,1]-data[t-1,1]))
               dy = np.argmax(self.dy_edges>(data[t,2]-data[t-1,2]))
               if px>0 and py>0 and dx>0 and dy>0:
                  h[dx-1,dy-1,px-1,py-1] += 1
        
        #smooth
        h_smooth = gaussian_filter(h,sigma)
        
        #normalize
        for i in range(len(self.x_edges)-1):
            for j in range(len(self.y_edges)-1):
                h_smooth[:,:,i,j] = h_smooth[:,:,i,j] / sum(h_smooth[:,:,i,j].flatten())
                
        return [h_smooth]
    
    def get_predictors(self,currx,curry,t,steps,minp=1e-10):
        predarr = []
        for [x,y] in steps:
           #preds = {}
           dx = x-currx
           dy = y-curry
           px = np.argmax(self.x_edges>currx)
           py = np.argmax(self.y_edges>curry)
           ix = np.argmax(self.dx_edges>dx)
           iy = np.argmax(self.dy_edges>dy)
           
           if px>0 and py>0 and ix>0 and iy>0:
              p = self.distribs[0][ix-1,iy-1,px-1,py-1] + minp
           else:
              p = minp
           
           predarr.append(p)
           
        return np.array(predarr)
              
           
class Unconditional_XY(Distribution):
    #P(x(t),y(t))
    def __init__(self,dataset,extent,seed=333,**kwargs):
        super().__init__(dataset,extent,**kwargs)
                
    def get_distribution(self,dataset,sigma=5):        
        h = np.zeros([len(self.x_edges)-1,len(self.y_edges)-1])
        
        for data in dataset:
            for t in range(1,data.shape[0]):
               px = np.argmax(self.x_edges>data[t-1,1])
               py = np.argmax(self.y_edges>data[t-1,2])
               if px>0 and py>0:
                  h[px-1,py-1] += 1
        
        #smooth
        h_smooth = gaussian_filter(h,sigma)
        
        #normalize                
        return [h_smooth/sum(h_smooth.flatten())]

    def get_candidates(self,currx,curry,n):
        steps = []
        for i in range(n):
           
            r1 = self.rng.random()
            ind = np.argmax(self.cumdistribs[0]>r1)
            xbin = self.mx[ind]
            ybin = self.my[ind]
            xind = np.argmax(self.x_edges==xbin)
            yind = np.argmax(self.y_edges==ybin)
 
            r21 = self.rng.random()
            r22 = self.rng.random()
            x = r21*self.x_edges[xind-1] + (1-r21)*self.x_edges[xind]
            y = r22*self.y_edges[yind-1] + (1-r22)*self.y_edges[yind]
                                         
            steps.append([x,y])
        return steps

        
class Unconditional_dXdY(Distribution):
    #P(dx(t),dy(t))
    def __init__(self,dataset,extent,**kwargs):
        super().__init__(dataset,extent,**kwargs)
        
    def get_distribution(self,dataset,sigma=5):
        
        h = np.zeros([len(self.dx_edges)-1,len(self.dy_edges)-1])
        
        for data in dataset:
            for t in range(1,data.shape[0]):
               dx = np.argmax(self.dx_edges>(data[t,1]-data[t-1,1]))
               dy = np.argmax(self.dy_edges>(data[t,2]-data[t-1,2]))
               if dx>0 and dy>0:
                  h[dx-1,dy-1] += 1
        
        #smooth
        h_smooth = gaussian_filter(h,sigma)
        
        #normalize
        h_smooth = h_smooth / sum(h_smooth.flatten())
                
        return [h_smooth]
    
    def get_candidates(self,currx,curry,n):
        steps = []
        for i in range(n):
           
           while True:
               r1 = self.rng.random()
               ind = np.argmax(self.cumdistribs[0]>r1)
               dxbin = self.mdx[ind]
               dybin = self.mdy[ind]
               dxind = np.argmax(self.dx_edges==dxbin)
               dyind = np.argmax(self.dy_edges==dybin)
               #if dxind==0:
               #    print(r1,ind,dxbin,dybin)
    
               r21 = self.rng.random()
               r22 = self.rng.random()
               x = currx + r21*self.dx_edges[dxind-1] + (1-r21)*self.dx_edges[dxind]
               y = curry + r22*self.dy_edges[dyind-1] + (1-r22)*self.dy_edges[dyind]
                              
               if x>self.extent[0] and x<self.extent[1] and y>self.extent[2] and y<self.extent[3]:
                   break
           
           steps.append([x,y])
        return steps

        
class Unconditional_ThetaR_Indep(Distribution):
    #P(theta)*P(r)
    def __init__(self,dataset,extent,**kwargs):
        super().__init__(dataset,extent,**kwargs)
        
    def get_distribution(self,dataset):
        
        h_dists = np.zeros(len(self.dist_edges)-1)
        h_angs = np.zeros(len(self.ang_edges)-1)
        
        for data in dataset:
            spatial_locations = data[:,1:3]
            diff_data = np.diff(spatial_locations,axis=0)
            dists_data = np.sqrt(diff_data[:,0]**2+diff_data[:,1]**2)
            angs_data = np.arctan2(diff_data[:,1],diff_data[:,0])
            
            h,_ = np.histogram(dists_data,bins=self.dist_edges)
            h_dists += h
            h,_ = np.histogram(angs_data,bins=self.ang_edges)
            h_angs += h
            #hack to enforce symmetry
            #h,_ = np.histogram(-angs_data,bins=self.ang_edges)
            #h_angs += h
            
        return [h_dists/np.sum(h_dists),h_angs/np.sum(h_angs)]
    
    def get_candidates(self,currx,curry,n):
        steps = []
        for i in range(n):
           
           while True:
               r1 = self.rng.random()
               r11 = self.rng.random()
               ind1 = np.argmax(self.cumdistribs[0]>r1)
               dist = r11*self.dist_edges[ind1] + (1-r11)*self.dist_edges[ind1+1]
    
               r2 = self.rng.random()
               r22 = self.rng.random()
               ind2 = np.argmax(self.cumdistribs[1]>r2)
               ang = r22*self.ang_edges[ind2] + (1-r22)*self.ang_edges[ind2+1]
               
               x = currx + dist*np.cos(ang)
               y = curry + dist*np.sin(ang)
               
               if x>self.extent[0] and x<self.extent[1] and y>self.extent[2] and y<self.extent[3]:
                   break
           
           steps.append([x,y])
        return steps

        
def step_select_run(data,distobj,models,n,pred_start,prevdata=None):
    
    assert pred_start>0


    df = pd.DataFrame()
    
    ii = 0
    
    datalen = data.shape[0]-1
    for i in range(datalen):
        sys.stdout.write("\r%d%%" % round(100*i/datalen))
        sys.stdout.flush()
        
        [t1,x1,y1] = data[i+1,:]
        [t0,x0,y0] = data[i,:]

        for m in models:
            m.model_update(t0,x0,y0)
        if i<pred_start:
            continue

        if prevdata is None:
           steps = distobj.get_candidates(x0,y0,n)
        else:
           steps = []
           for j in range(n):
              steps.append([prevdata['x'][(n+1)*ii+j],prevdata['y'][(n+1)*ii+j]])
        steps.append([x1,y1])
        target = [0 for j in range(n)]
        target.append(1)

        newdf = pd.DataFrame(np.array(steps),columns=['x','y'])
        newdf['t']=t1
        for m in models:
            preds_df = m.get_predictors(t1,steps)            
            newdf = pd.concat([newdf,preds_df],axis=1)
        newdf['target'] = target
        df = pd.concat([df,newdf])
        ii += 1
    
    return df

    
def markov_test(distobj,n=100):
    
    out = [[0,0]]
    traj = out
    for i in range(n):
        out = distobj.get_candidates(out[0][0],out[0][1],1)
        traj.append(out[0])

    tarr = np.array(traj)
    plt.plot(tarr[:,0],tarr[:,1])       
    plt.show()
     
    return tarr
 

def import_preds(new_df,old_df):
    if len(new_df)==len(old_df) and sum(abs(new_df["x"]-old_df["x"]))<1e-6*len(new_df):
        for col in new_df.columns:
           old_df[col] = new_df[col]
        print("\nSuccessfully combined with existing file.")
        return old_df
    else:
        print("\nData frame importing failed! Candidate steps do not agree.")
        return new_df    

def add_ID_cols(pred_df,dataset,testID):
    
    if dataset == "waldo":
       pred_df["Season"] = testID[:6]
       nums = testID[6:].split('-S')
       pred_df["Participant"] = int(nums[0])
       if len(nums)>1:
          pred_df["Sequence"] = int(nums[1])
    else:
       raise ValueError("Unknown dataset in ID parser")
    #other datasets can be added below
          
    
    return pred_df


def main(testdata,models,n_candidates,pred_start=1,save_file=None,existing_file=None,data_ids=None):
    #iterates through data and collects predictors and sequence IDs (if applicable)
    #data_ids should be a tuple containing dataset identifier and a list of strings with participant info for each
    
    if existing_file is not None:
       total_df = pd.read_csv(existing_file)
       curr_row = 0
       
    warnings.filterwarnings('ignore')

    if data_ids is not None:
        assert len(data_ids[1])==len(testdata)
        
    #run new/changed models if needed
    if len(models)>0:
        print(f"Running {len(models)} new or modified models.")
        new_df = pd.DataFrame()

        for i,data in enumerate(testdata):
           print('\nSequence',i+1,'of',len(testdata))
           if existing_file is not None:
                n_rows = (n_candidates+1)*(len(data)-pred_start-1)
                prevdata = total_df.iloc[curr_row:(curr_row+n_rows),:]
                prevdata = prevdata.reset_index(drop=True)
                curr_row += n_rows
           else:
                prevdata = None   
           pred_df = step_select_run(data,basedist,models,n_candidates,pred_start,prevdata=prevdata) 
           #pass to ID parser
           if data_ids is not None:
              pred_df = add_ID_cols(pred_df,data_ids[0],data_ids[1][i])
           new_df = pd.concat([new_df,pred_df],axis=0)
           
           for m in models:
              m.reset()
           
        
        new_df = new_df.reset_index(drop=True)
        if existing_file is not None:
            total_df = import_preds(new_df,total_df)
        else:
            total_df = new_df
            
        if save_file is not None and existing_file is not save_file:
            total_df.to_csv(save_file,index=False)
            print("\nSuccessfully saved.")
        else:
            print("\nNot saving: new filename needed!")
            
    return total_df

def time_constant_opt():
    models = []
    models.append(ParametricModel(basedist,cols=['logdist','cardinality','dir_change','wall_dist','wall_dist_change']))
    for t in range(1,21):
       models.append(ParametricModel(basedist,cols=["crossings"],maxtime_cross=t,col_t=True))
    for t in range(1,21):
       models.append(ParametricModel(basedist,cols=["closest_point"],maxtime_closest=t,col_t=True))
    for t in np.arange(0.5,10.1,0.5):
       models.append(ParametricModel(basedist,cols=["dir_moment"],maxtime_dir=t,col_t=True))
    models.append(MarkovModel(markovdist,predname="markov_dxdy"))

    return models

def compare_synthetic(pred_df,basedist,models,n_candidates):
    
    data_df = pd.read_csv('./generator_data/generator_markov_bound_sr20_sig0.4_0.csv',header=1)
    data = np.array(data_df)
    
    pred_df2 = step_select_run(data,basedist,models,n_candidates) 
    
    inds = pred_df.target==1
    plt.hist(pred_df.crossings[inds],bins=range(0,10),density=True,alpha=0.7)
    inds = pred_df2.target==1
    plt.hist(pred_df2.crossings[inds],bins=range(0,10),density=True,alpha=0.7)
    plt.legend(['data','markovian generator'])
    plt.xlabel('number of self crossings')
    plt.ylabel('frequency')
    plt.show()

if __name__ == '__main__':
    
    
    """
    At minimum, the following need to be specified:
    - traindata and testdata: list of numpy arrays with dim (N,3)
      columns: (t,x,y)
      If the data comes straight from basic_measures.py, use:
          data_npz = np.load(npz_path)
          data = [data_npz[x] for x in data_npz.files]
      Train and test can be the same if overfitting is not an issue.
    - extent: tuple (xmin,xmax,ymin,ymax)
      These are the screen boundaries in data units.
    - n_candidates: set to zero if you only want predictors for real data points
    - data_ids: set to None if participant info not needed in dataset
      otherwise, this is a tuple of (dataset name string, list of ID strings)
      and a parser should be added to the function add_ID_cols
    """
    
    ### SELECT DATASET BELOW    
    data = 'waldo_test'
    
    ### SAVED NN MODEL TO USE (only used if NNModel is in model list)
    nnmodel_id = 'shorter-beta2'
       
    data_ids = None 
    
    if data=="waldo_all":
        #all data, full sequences
        traindata = load_data('waldo','all')
        testIDs = traindata.IDs
        testdata = traindata    
        data_ids = ('waldo',testIDs)

    elif data=="waldo_test":
        #train distributions on NN training data, step selection on NN test+val data
        traindata = load_data('waldo_shorter','train')
        testdata1 = load_data('waldo_shorter','test')
        testdata2 = load_data('waldo_shorter','val')
        testdata = testdata1 + testdata2
        testIDs = testdata1.IDs + testdata2.IDs
        data_ids = ('waldo',testIDs)
    else:
        raise ValueError("invalid data type")
    
    if "waldo" in data:
        stds = traindata.S_std.numpy().flatten()
        means = traindata.S_mean.numpy().flatten()
        extent = ((0-means[0])/stds[0],(1000-means[0])/stds[0],(0-means[1])/stds[1],(800-means[1])/stds[1])
        traindata = [data.detach().numpy() for data in traindata]
        testdata = [data.detach().numpy() for data in testdata]

 

    basedist = Unconditional_XY(traindata,extent)
    markovdist = Conditional_dXdY(traindata,extent,seed=444)
    
 

    #Import previously run steps and predictors
    #Models in list will be added if they dont exist or overwrite older columns if they do
    #Set to None to start from scratch
    existing_file = None #'stepselect_predictors_nn60_new_posttimeopt.csv'
    
    save_file = None #'stepselect_output.csv'
    
    models = []    

    ### LIST OF PREDICTIVE MODELS SPECIFIED BELOW
    
    #models = time_const_opt()   
    models.append(ParametricModel(basedist))
    models.append(MarkovModel(markovdist,predname="markov_dxdy"))
    #models.append(NNModel(nnmodel_id,n_iter=60,maxhist=1,predname="nn_nohist"))
    #models.append(NNModel(nnmodel_id,n_iter=60))

    n_candidates = 4 #num false steps per real step

    
    pred_df = main(testdata,models,n_candidates,save_file=save_file,existing_file=existing_file,data_ids=data_ids)        
        



