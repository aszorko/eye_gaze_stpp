# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:52:57 2024

@author: alexansz
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os


def crosscorr_nan(x, y, tend=1000):
    #nan-friendly 2-time cross correlation
    tt = len(x)
    out = np.zeros(tend)
    for i in range(tend):
        x1 = x[0:(tt-tend)]
        x1 = (x1 - np.nanmean(x1)) / np.nanstd(x1)
        y1 = y[i:(tt-tend+i)]
        y1 = (y1 - np.nanmean(y1)) / np.nanstd(y1)
        out[i] = np.nansum(x1*y1)/tt
        
    return out

#def analyse(file):

def ecdf(xarr):
    newarr = xarr[xarr>0]
    n = len(newarr)
    y = np.array([(i+1)/(n+1) for i in range(n)])
    
    return sorted(newarr),y



def preprocess(x,y,category,boundaries,debug=False):

        maxspeed = 12 #20 #pixels per ms
        minx = boundaries[0]
        miny = boundaries[2]
        maxx = boundaries[1]
        maxy = boundaries[3]
        minsegment = 20 #shortest non-nan segment (ms)

        ###SMOOTHING

        #initial x,y smooth (mostly to help remove spurious 1-3 frame segments)
        #wind = 5 #3
        #x = np.convolve(x, np.ones(wind), 'same') / wind
        #y = np.convolve(y, np.ones(wind), 'same') / wind
 
                
        #get raw speeds
        vx = np.diff(x)
        vy = np.diff(y)
        origspeed = np.sqrt(vx**2+vy**2)
        

        #remove blinks and out of bounds x,y
        inds = np.argwhere((x <= minx) | (y <= miny) | (x > maxx) | (y > maxy))
        x[inds[:,0]] = np.nan
        y[inds[:,0]] = np.nan

        x[category==0] = np.nan
        y[category==0] = np.nan


        #remove sudden jumps
        inds = np.argwhere((origspeed>maxspeed) | (np.isnan(origspeed)==True))

        x[inds[:,0]] = np.nan
        x[inds[:,0]+1] = np.nan
        y[inds[:,0]] = np.nan
        y[inds[:,0]+1] = np.nan
        origspeed[inds] = np.nan


        #remove short segments
        segchange = np.diff(np.isfinite(x))
        seginds = np.argwhere(segchange==1)
        for segstart in seginds[:,0]:
            seglen = np.argmax(segchange[segstart:]==-1)
            if seglen>0 and seglen < minsegment:
                x[segstart+1:segstart+seglen] = np.nan
                y[segstart+1:segstart+seglen] = np.nan

        #interpolate over gaps
        xnew = np.array(pd.DataFrame(x).interpolate())
        vx = np.diff(xnew,axis=0)
        ynew = np.array(pd.DataFrame(y).interpolate())
        vy = np.diff(ynew,axis=0)

        speed = np.sqrt(vx**2+vy**2).flatten()
        
        startvalid = np.argmax(np.isfinite(speed))
        endvalid = len(speed) - 1 - np.argmax(np.isfinite(speed[-1::-1]))
        
        #speed smoothing
        wind = 9
        #smspeed = np.convolve(speed, np.ones(wind), 'same') / wind
        smspeed_sub = signal.savgol_filter(speed[startvalid:endvalid],wind,3)
        
        smspeed = 1*speed
        smspeed[startvalid:endvalid] = smspeed_sub

        #remove originally invalid times
        smspeed[inds] = np.nan
        smspeed[np.isnan(origspeed)] = np.nan
        smspeed[smspeed<0] = 0
        #category[inds] = -1
        
        x[inds[:,0]] = np.nan
        x[inds[:,0]+1] = np.nan
        y[inds[:,0]] = np.nan
        y[inds[:,0]+1] = np.nan

        #remove newly invalid times
        inds = np.argwhere(smspeed>maxspeed)
        smspeed[inds] = np.nan
        x[inds[:,0]] = np.nan
        x[inds[:,0]+1] = np.nan
        y[inds[:,0]] = np.nan
        y[inds[:,0]+1] = np.nan

        if debug:
           frames = [30600,31000,35000] #chosen for participant 49
           length = 100
           for startframe in frames:
              plt.plot(origspeed[startframe:startframe+length])
              plt.plot(smspeed[startframe:startframe+length])
              plt.show()

        """
        #remove blinks and out of bounds x,y
        inds = np.argwhere((x <= minx) | (y <= miny) | (x > maxx) | (y > maxy))
        x[inds[:,0]] = np.nan
        y[inds[:,0]] = np.nan

        x[category==0] = np.nan
        y[category==0] = np.nan
        """
        
        return x,y,smspeed,xnew,ynew

    
def get_saccades_auto(category):
    startinds = []
    endinds= []
    diffcat = np.diff(category)
    inds = np.argwhere((diffcat>0) & (category[1:]==2))
    for i,ind in enumerate(inds[:,0]):
        startinds.append(ind+1)
        end = np.nanargmax(category[(ind+1):]<2)
        if i==len(inds)-1 and end==0:
            endinds.append(len(category)-1)
        else:
            endinds.append(ind+end+1)
        
    return startinds,endinds


def get_saccades_manual(x,y,speed):
    
    #initial peak finding
    minpeakheight = 2 #3 #5 #pixels/ms
    minpeakdist = 15 #20 #ms

    #saccade start/end finding
    peakbase = 1.2 #0.8 #0.5 #1 #pixels/ms
    accelbase = 0.05 #0.3 #pixels/ms/ms
    minfixduration = 10 #ms

    #filtering
    minsaccade = 0 #20 #pixels
    maxduration = 100
    minavgspeed = 0 #1.0 #pixels/ms


    startinds = []
    endinds = []
    accel = np.diff(speed)
    
    peaks = signal.find_peaks(speed,height=minpeakheight,distance=minpeakdist)
    
    
    joins = 0
    discards = 0
    
    for peak in peaks[0]:
        
        if len(endinds)>0 and endinds[-1] > peak:
            #skip
            continue
        
        ind = np.argmax((speed[peak:0:-1]<peakbase) & (accel[peak:0:-1]<accelbase))
        if ind>0:
           startind = peak-ind+1
        else:
           startind = 0
        ind = np.argmax(speed[peak:]<peakbase)
        if ind>0:
           endind = peak+ind-1
        else:
           endind = len(speed)
        
        startinds.append(startind)
        endinds.append(endind)

    print("Candidate saccades:",len(startinds))

        
    #go through again and filter
    finalstartinds = []
    finalendinds = []
    startind = startinds[0]
    for i in range(len(startinds)):
        #update endind, skip if there is a short fixation after
        endind = endinds[i]
        if i<len(startinds)-1 and startinds[i+1]-endind < minfixduration:
           joins += 1
           continue
        dist = np.sqrt((x[startind]-x[endind])**2+(y[startind]-y[endind])**2)
        if dist > minsaccade and endind-startind<maxduration and np.nanmean(speed[startind:endind])>minavgspeed:
           finalstartinds.append(startind)
           finalendinds.append(endind)
        else:
           discards += 1
        #update startind
        if i<len(startinds)-1:
           startind = startinds[i+1]
        
        
    print("Saccade joins:",joins)
    print("Saccade discards:",discards)
    
    return finalstartinds,finalendinds
    
def get_saccade_stats(trialdict,x,y,speed,category,method,plot=False):
    #custom classification

    #grouped means
    maxt = 80
    durbins = np.arange(20,maxt+1,20)
    speedsum = np.zeros([maxt+1,len(durbins)-1])
    speedcount = np.zeros([maxt+1,len(durbins)-1])    
    accelsum = np.zeros([maxt+1,len(durbins)-1])
    accelcount = np.zeros([maxt+1,len(durbins)-1])    

    #2d histograms
    maxt_mat = 40
    accelbins = np.arange(-1.5,1.51,0.1)
    speedbins = np.arange(0,12,0.5)
    accelmat = np.zeros([len(accelbins)-1,maxt_mat])
    speedmat = np.zeros([len(speedbins)-1,maxt_mat])

    #plt.hist(np.log(np.diff(peaks[0])),20)
    #plt.show()
    

    #plt.plot(x,y)
    
    
    newpeaks = []
    peakheights = []
    relpeaks = []
    accelpeaks = []
    accelmins = []
    alldists = []
    allsacdur = []
    allfixdur = []
    allmeanspeed = []
    allaccel = []
    sactimes = []
    fixtimes = []
    xfix = 1*x
    yfix = 1*y
    speedfix = 1*speed
    xsac = x*np.nan
    ysac = y*np.nan
    speedsac = speed*np.nan
    accel = np.diff(speed)

    if method=='manual':
        startinds,endinds = get_saccades_manual(x,y,speed)
    elif method=='auto':
        startinds,endinds = get_saccades_auto(category)
    else:
        raise(ValueError,'invalid method')
    fixtarr = np.array(startinds)
    stpp_data = np.array([fixtarr/1000,x[fixtarr],y[fixtarr]])
    stpp_mask = np.isfinite(stpp_data[1,:])
    
    xfix[category<1] = np.nan    
    yfix[category<1] = np.nan    
    speedfix[category[:-1]<1] = np.nan

    acceltime = 8 #frames

    for i,startind in enumerate(startinds):
        if startind+acceltime < len(accel):
           subaccel = accel[startind:startind+acceltime]
           if sum(np.isfinite(subaccel))>0:
              allaccel.append(np.nanmean(subaccel))
        endind = endinds[i]
        dist = np.sqrt((x[startind]-x[endind])**2+(y[startind]-y[endind])**2)
        try:
           peak = np.nanargmax(speed[startind:endind])
        except:
           continue
        try:
           accelpeaks.append(np.nanargmax(accel[startind:startind+peak]))        
        except:
           accelpeaks.append(np.nan)
        try:
           accelmins.append(peak+np.nanargmax(-accel[startind+peak:endind]))        
        except:
           accelmins.append(np.nan)
        relpeaks.append(peak)
        newpeaks.append(peak+startind)
        peakheights.append(speed[peak+startind])
        allsacdur.append(endind-startind+1)
        sactimes.append(startind)
        alldists.append(dist)
        allmeanspeed.append(np.nanmean(speed[startind:endind]))
        if i>0:
           allfixdur.append(startind-endinds[i-1])
           fixtimes.append(endinds[i-1]+1)
        dbin = np.argmax(durbins>allsacdur[-1])
        if dbin > 0:
           #build grouped speed vs time during saccade
           subspeed = speed[startind:startind+durbins[dbin-1]]
           validinds = np.argwhere(np.isnan(subspeed)==False)
           speedsum[validinds[:,0],dbin-1] += subspeed[validinds[:,0]] 
           speedcount[validinds[:,0],dbin-1] += 1
           #build grouped accel vs time during saccade
           subaccel = accel[startind:startind+durbins[dbin-1]]
           validinds = np.argwhere(np.isnan(subaccel)==False)
           accelsum[validinds[:,0],dbin-1] += subaccel[validinds[:,0]] 
           accelcount[validinds[:,0],dbin-1] += 1
        
        xsac[startind:endind] = x[startind:endind]
        ysac[startind:endind] = y[startind:endind]
        speedsac[startind:endind] = speed[startind:endind]
        xfix[startind:endind] = np.nan
        yfix[startind:endind] = np.nan
        speedfix[startind:endind] = np.nan
        for j in range(startind,min(startind+maxt_mat,endind)):
            if j<len(accel):
               aind = np.argmax(accelbins>accel[j])
               if aind > 0:
                  accelmat[aind-1,j-startind] += 1
            if j<len(speed):
               speedind = np.argmax(speedbins>speed[j])
               if speedind > 0:
                  speedmat[speedind-1,j-startind] += 1
    
    #vxsac = np.diff(xsac)
    #vysac = np.diff(ysac)
    #speedsac = np.sqrt(vxsac**2+vysac**2)

    #vxfix = np.diff(xfix)
    #vyfix = np.diff(yfix)
    #speedfix = np.sqrt(vxfix**2+vyfix**2)

    minfix = 60
    dur,durcdf = ecdf(np.array(allfixdur)-minfix)
    wbx = np.log(dur)
    wby = np.log(-np.log(1-durcdf))
    md = sm.OLS(wby,np.array([np.ones([len(wbx)]),wbx]).T)
    dur_mdf = md.fit()
    #print(dur_mdf.summary())
    #print(dur_mdf.params)
    weibull_k = dur_mdf.params[1]
    weibull_l = np.exp(-dur_mdf.params[0]/weibull_k)

    
    if plot:
        plt.subplot(1,2,1)
        plt.plot(xfix,yfix)
        plt.subplot(1,2,2)
        plt.plot(xsac,ysac)
        plt.title(file)
        plt.show()

        dur,durcdf = ecdf(np.array(allsacdur))
        plt.scatter(np.log(dur),np.log(-np.log(1-durcdf)))
        plt.xlabel('saccade duration (log)')
        plt.ylabel('Weibull estimation function')
        plt.show()
        
        plt.hist(np.log(allsacdur),20)
        plt.xlabel('saccade duration (log)')
        plt.show()

        plt.hist(np.log(alldists),20)
        plt.xlabel('saccade distance (log)')
        plt.show()        

        plt.hist(relpeaks,20)
        plt.xlabel('time of speed peak ')
        plt.show()

        plt.hist(accelpeaks,20)
        plt.xlabel('time of acceleration peak ')
        plt.show()

        plt.hist(accelmins,20)
        plt.xlabel('time of deceleration peak')
        plt.show()


        plt.scatter(wbx,wby)
        plt.ylim([np.nanmin(wby)-0.1,np.nanmax(wby)+0.1])
        xvec = np.array([np.nanmin(wbx)-0.1,np.nanmax(wbx)+0.1])
        yvec = xvec*dur_mdf.params[1] + dur_mdf.params[0]
        plt.plot(xvec,yvec)
        plt.xlabel('fixation post-minimum (log)')
        plt.ylabel('Weibull estimation function')
        plt.show()
        
        plt.hist(np.log(allfixdur),20)
        plt.xlabel('fixation duration (log)')
        plt.show()

        plt.hist(allmeanspeed,20)
        plt.xlabel('mean speed')
        plt.show()

        plt.hist(allaccel,20,range=(-1,1))
        plt.xlabel('starting acceleration')
        plt.show()

        
        #plt.hist(np.diff(np.array(newpeaks)),20)
        #plt.xlabel('time between speed peaks')
        #plt.show()

        if method=='manual':
           plt.hist(speed[category[:-1]==1],40,range=(0,4),alpha=0.7)
           plt.hist(speedfix,40,range=(0,4),alpha=0.7)
           plt.show()
        
           plt.hist(speed[category[:-1]==2],40,range=(0,20),alpha=0.7)
           plt.hist(speedsac,40,range=(0,20),alpha=0.7)
           plt.show()
        
        plt.scatter(alldists,peakheights)
        plt.xlabel('saccade distance')
        plt.ylabel('peak speed')
        plt.show()

        plt.scatter(allsacdur,peakheights)
        plt.xlabel('saccade duration (ms)')
        plt.ylabel('peak speed')
        plt.show()

        plt.scatter(allsacdur,alldists)
        plt.xlabel('saccade duration (ms)')
        plt.ylabel('saccade distance')
        plt.show()

        plt.scatter(allsacdur,allmeanspeed)
        plt.xlabel('saccade duration (ms)')
        plt.ylabel('mean speed')
        plt.show()
        
        plt.scatter(allsacdur,relpeaks)
        plt.xlabel('saccade duration (ms)')
        plt.ylabel('time of speed peak (ms)')
        plt.show()

        plt.scatter(sactimes,np.log(allsacdur))
        plt.xlabel('time in trial(ms)')
        plt.ylabel('saccade duration (log ms)')
        plt.show()

        plt.scatter(fixtimes,np.log(allfixdur))
        plt.xlabel('time in trial (ms)')
        plt.ylabel('fixation duration (log ms)')
        plt.show()

        plt.scatter(accelpeaks,np.array(accelmins)-np.array(accelpeaks))
        plt.xlabel('time of acceleration peak (ms)')
        plt.ylabel('accel-decel peak time (ms)')
        plt.show()


        plt.plot(speedsum/speedcount)
        plt.xlabel('time during saccade (ms)')
        plt.ylabel('speed')
        plt.show()

        for i in range(maxt_mat):
            speedmat[:,i] = speedmat[:,i] / sum(speedmat[:,i])
        plt.imshow(speedmat,aspect='auto',origin='lower',extent=(0,maxt_mat,speedbins[0],speedbins[-1]),interpolation='gaussian',filterrad=2.5)
        plt.xlabel('time during saccade (ms)')
        plt.ylabel('speed (px/ms)')
        plt.show()

        plt.plot(accelsum/accelcount)
        plt.xlabel('time during saccade (ms)')
        plt.ylabel('acceleration')
        plt.show()

        for i in range(maxt_mat):
            accelmat[:,i] = accelmat[:,i] / sum(accelmat[:,i])
        plt.imshow(accelmat,aspect='auto',origin='lower',extent=(0,maxt_mat,accelbins[0],accelbins[-1]),interpolation='gaussian',filterrad=4.0)
        plt.xlabel('time during saccade (ms)')
        plt.ylabel('acceleration (px.ms^-2)')
        plt.plot([0,maxt_mat],[0,0],'white')
        plt.show()

    trialdict["Mean speed"]=np.nanmean(speed)*1000
    trialdict["Median speed"]=np.nanmedian(speed)*1000
    trialdict["Saccades"]=len(alldists)
    trialdict["Median saccade distance"]=np.nanmedian(alldists)
    trialdict["Median saccade duration"]=np.nanmedian(allsacdur)/1000
    trialdict["Median fixation duration"]=np.nanmedian(allfixdur)/1000
    trialdict["Weibull fixation k"]=weibull_k
    trialdict["Weibull fixation lambda"]=weibull_l/1000   
    #trialdict["Median time between saccades"]=np.nanmedian(np.diff(np.array(newpeaks)))/1000

    #print(speedsum)
    #print(count)

    return trialdict,stpp_data[:,stpp_mask]

def spatial_stats(x,y,smspeed,plot=False):
    gridsize = 10
    gridx = np.arange(0,1001,gridsize)
    gridy = np.arange(0,801,gridsize)
    visitsites = np.zeros([len(gridx)-1,len(gridy)-1])
    
    visitsvst = []
    visitsum = 0
    for t in range(len(x)):
        if np.isfinite(x[t]):
            xloc = np.argmax(gridx>x[t])
            yloc = np.argmax(gridy>y[t])
            if xloc>0 and yloc>0:
                if visitsites[xloc-1,yloc-1]==0:
                   visitsites[xloc-1,yloc-1] = 1
                   visitsum += 1
                visitsvst.append(visitsum)
    
    nsites = (len(gridx)-1)*(len(gridy)-1)
    tt = np.arange(len(visitsvst))
    poissonspeed = np.nanmean(smspeed)/gridsize/nsites
    poissonsites = 1-np.exp(-tt*poissonspeed)
    empsites = np.array(visitsvst)/nsites

    md = sm.OLS(np.log(1-empsites),np.array(tt).T)
    sites_mdf = md.fit()
    #print(sites_mdf.summary())
    #print(sites_mdf.params)
    lamb = sites_mdf.params[0]
    fitsites = 1-np.exp(tt*lamb)

    if plot:
       t=np.arange(0,len(empsites),1)/1000            
       plt.plot(t,100*empsites)
       #plt.plot(np.log(t+1),np.log(100*poissonsites),'-.')
       plt.plot(t,100*fitsites,'--')
       plt.xlabel('valid time steps')
       plt.ylabel('% visited sites')
       plt.show()
    
    return -1000*lamb,sites_mdf.rsquared

if __name__=="__main__":

    
    files = os.listdir()
    #files = ['data_participant_4_trial_4.txt']

    stpp_out = {}
    
    newdf = pd.DataFrame()

    fps = 1000
    minlength = 15

    for file in files:
        if file[0] == '.' or ('data' not in file and 'Ruj' not in file):
            continue
        
        
        data = pd.read_csv(file,sep=' ',header=None)
        
        df = data.transpose()
        #time (ms),x,y,pupil,sm_x(?),sm_y(?),sm_pupil(?),0,cat 1, cat 2
        
        
        #left eye x=1,y=2,pup=3,cat=8; right eye x=4,y=5,pup=6,cat=9
        x = np.array(df[4])
        y = np.array(df[5])
        trialcat = np.array(df[9])

        if 'Ruj' in file:
            seglength = 45
            num_segments = 9
            minx = -50
            miny = 0
            maxx = 1950
            maxy = 1100
        else:
            seglength=30
            num_segments = 6
            minx = 0
            miny = 0
            maxx = 1000
            maxy = 800            

        trialx,trialy,trialspeed,_,_ = preprocess(x,y,trialcat,(minx,maxx,miny,maxy),debug=False)
        
            

        #autocorr = crosscorr_nan(speed,speed,tend=500)
        
        #plt.plot(autocorr)
        #plt.show()
        
        for ii in range(num_segments):
            if fps*(ii*seglength+minlength)>len(trialx):
                continue
            
            #inds = range(int(len(trialx)*ii/num_segments),int(len(trialx)*(ii+1)/num_segments))
            inds = range(ii*fps*seglength,min(len(trialx),(ii+1)*fps*seglength))
            newx = trialx[inds]
            newy = trialy[inds]
            smspeed = trialspeed[inds[:-1]]
            category = trialcat[inds]
            
            trialdict = {}
            
            trialdict["Filename"]=file
            
            
            if 'Winter' in file:
               trialdict["Season"] = "Winter"
            elif 'data' in file:
               trialdict["Season"] = "Summer"
            else:
               trialdict["Season"] = "NA"
                
            if "Ruj" in file:
               trialdict["Participant"] = str(100+int(file[9:].split('.')[0]))
            else:
               trialdict["Participant"] = file.split('_')[-3]
               
            trialdict["Segment"] = ii+1
            
            trialdict["Length"]=len(newx)/1000
            trialdict["NaNs"]=sum(np.isnan(newx)/len(newx))
            trialdict["X_std"]=np.nanstd(newx)
            trialdict["Y_std"]=np.nanstd(newy)
            
            lamb,pois_rsq = spatial_stats(newx,newy,smspeed,plot=False)
            
            
            method = 'auto'
            trialdict,stpp_data = get_saccade_stats(trialdict,newx,newy,smspeed,category,method,plot=False)
            
            dpos = np.diff(stpp_data[1:3,:].T,axis=0)
            stpp_data_dists = np.sqrt(np.sum(dpos**2,axis=1))
            stpp_data_angs = np.arctan2(dpos[:,1],dpos[:,0])
            stpp_data_dt = np.diff(stpp_data[0,:])
            
            #waldo data
            seqname = trialdict["Season"]+trialdict["Participant"]+'-S'+str(trialdict["Segment"])
            #alternative below should work for any data:
            #seqname = trialdict["Filename"]+'-S'+str(trialdict["Segment"])    
            
            stpp_out[seqname] = stpp_data.T
    
            trialdict["Poisson discovery rate"] = lamb
            trialdict["Poisson discovery fit"] = pois_rsq
            
            for key in trialdict:
               print(key,trialdict[key])
            
            
            newdf = pd.concat([newdf,pd.DataFrame(trialdict,index=[0])])
        
    newdf.to_csv('basicmeasures.csv')
    np.savez('fixations_auto',**stpp_out)
    

    """
    (a,b,c,d) = np.unique(newdf["Participant"],return_index=True, return_inverse=True, return_counts=True)
    suminds = (newdf["Season"]=="Summer") & (d[c]==2*num_segments)
    sumdf = newdf[suminds].sort_values("Participant")
    wintinds = (newdf["Season"]=="Winter") & (d[c]==2*num_segments)
    wintdf = newdf[wintinds].sort_values("Participant")
    
    if len(sumdf)>num_segments: 
        dtypes = wintdf.dtypes
        for i,measure in enumerate(wintdf.columns):
            if dtypes[i]=='float64':
                plt.scatter(sumdf[measure],wintdf[measure])
                plt.xlabel('summer')
                plt.ylabel('winter')
                plt.title(measure)
                plt.show()
    """