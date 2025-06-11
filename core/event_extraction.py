"""
RBA_theta searches for variations in a dataset within and above a given Threshold. 
"""            
import pandas as pd
import numpy as np
from itertools import compress
from core.helpers import lam


def significant_events(data, threshold, use_mcmc=False, mcmc_threshold=None, window_minutes=10, freq_secs=60):
    """
    Detects significant up/down ramp events using sliding windows and log-transformed data.
    """
    if use_mcmc and mcmc_threshold is not None:
        threshold = mcmc_threshold

    eps = 1e-6
    points_per_win = int((window_minutes * 60) / freq_secs)
    n = len(data)
    all_events = []

    for start in range(0, n, points_per_win):
        end = min(start + points_per_win, n)
        if end - start < 2:
            continue
        window = data[start:end]
        log_window = np.log(window + eps)

        prev = log_window[0]
        delta = []
        for item in log_window[1:]:
            delta.append(item - prev)
            prev = item

        events = []
        idx_start = 0
        sign = lambda x: (1, -1)[x < 0]
        next_delta = np.diff(log_window)
        next_delta = np.append(next_delta, 0)

        for i in range(len(log_window) - 1):
            if sign(next_delta[i]) == sign(next_delta[i + 1]):
                continue
            else:
                new_event = [idx_start, i + 1, log_window[idx_start], log_window[i + 1]]
                events.append(new_event)
                idx_start = i + 1

        # Filter by amplitude
        filtered = [ev for ev in events if abs(ev[3] - ev[2]) >= threshold]

        # Further filter by duration and slope
        min_duration = 3
        min_slope = 0.05
        final_events = []
        for ev in filtered:
            dur = ev[1] - ev[0]
            slope = abs(ev[3] - ev[2]) / max(1, dur)
            if dur >= min_duration and slope >= min_slope:
                final_events.append(ev)

        for ev in final_events:
            t1 = ev[0] + start
            t2 = ev[1] + start
            all_events.append(pd.Series([
                t1,
                t2,
                t2 - t1,
                ev[2],
                ev[3],
                ev[3] - ev[2],
                (ev[2] + ev[3]) / 2,
                (np.arctan2((ev[3] - ev[2]) * 100, (t2 - t1))) * 180 / np.pi
            ], index=['t1', 't2', '∆t_m', 'w_m(t1)', 'w_m(t2)', '∆w_m', 'σ_m', 'θ_m']))

    return pd.DataFrame(all_events)

def stationary_events(data, threshold, use_mcmc=False, mcmc_threshold=None, window_minutes=10, freq_secs=60):
    """
    Detects stationary segments using sliding windows and log-transformed data.
    """
    if use_mcmc and mcmc_threshold is not None:
        threshold = mcmc_threshold

    eps = 1e-6
    points_per_win = int((window_minutes * 60) / freq_secs)
    n = len(data)
    all_events = []

    for start in range(0, n, points_per_win):
        end = min(start + points_per_win, n)
        if end - start < 2:
            continue
        window = data[start:end]
        log_window = np.log(window + eps)
        limit = threshold / 2
        idx = 0
        length = len(log_window)

        while idx < length:
            allowance = length - idx
            forward = 0
            for i in range(1, allowance):
                if log_window[idx] - limit <= log_window[idx + i] <= log_window[idx] + limit:
                    forward += 1
                else:
                    break
            if forward >= 5:
                segment = log_window[idx:(idx + forward + 1)]
                if max(segment) - min(segment) < threshold:
                    t1 = idx + start
                    t2 = idx + forward + start
                    all_events.append(pd.Series([
                        t1,
                        t2,
                        t2 - t1,
                        sum(segment) / len(segment)
                    ], index=['t1', 't2', '∆t_s', 'σ_s']))
            idx += forward + 1

    return pd.DataFrame(all_events)


def rainflow(data, threshold, flm=0, l_ult=1e16, uc_mult=0.5):
    """ Rainflow counting of a signal's turning points with Goodman correction

        Args:
            array_ext (numpy.ndarray): array of turning points

        Keyword Args:
            flm (float): fixed-load mean [opt, default=0]
            l_ult (float): ultimate load [opt, default=1e16]
            uc_mult (float): partial-load scaling [opt, default=0.5]

        Returns:
            array_out (numpy.ndarray): (5 x n_cycle) array of rainflow values:
                                        1) load range
                                        2) range mean
                                        3) Goodman-adjusted range
                                        4) cycle count
                                        5) Goodman-adjusted range with flm = 0

    """
    prev = data[0]
    delta = []
    for item in data[1:]:
        delta.append(item - prev)
        prev = item

    start = 0
    maximum = [data[start], start]
    sign = lambda x: (1, -1)[x < 0]

    for i in range(len(delta) - 1):
        if sign(delta[i]) == sign(delta[i+1]):
            continue
        else:
            new_event = [data[i + 1], i + 1]
            maximum.append(new_event)
            start = i + 1

    maximum.append([data[len(data) - 1], len(data) - 1])
    
    
    #rainflow with Power range and time range of cycles
    
    l = len(maximum)                        # total size of input array
    array_out = np.zeros((l - 1, 8))        # initialize output array
    pr = 0                                  # index of input array
    po = 0                                  # index of output array
    j = -1                                  # index of temporary array "a"
    a = np.empty((len(maximum),2))          # temporary array for algorithm
    # loop through each turning point stored in input array
    for i in range(l):
        
        j += 1                  # increment "a" counter
        a[j] = maximum[pr]      # put turning point into temporary array
        pr += 1                 # increment input array pointer

        
        while ((j >= 2) & (np.fabs( a[j-1,0] - a[j-2,0] ) <= np.fabs( a[j,0] - a[j-1,0]) ) ):
            lrange = np.fabs( a[j-1,0] - a[j-2,0] )
            
            # partial range
            if j == 2:
                mean = (a[0, 0] + a[1, 0]) / 2.
                Pstart = a[0, 0]
                Pend = a[1, 0]
                tstart = a[0, 1]
                tend = a[1, 1]
                a[0] = a[1]
                a[1] = a[2]
                j = 1
                if (lrange > 0):
                    array_out[po, 0] = Pstart
                    array_out[po, 1] = Pend
                    array_out[po, 2] = tstart
                    array_out[po, 3] = tend
                    array_out[po, 4] = Pend-Pstart
                    array_out[po, 5] = tend-tstart
                    array_out[po, 6] = mean
                    array_out[po, 7] = uc_mult
                    po += 1
            # full range
            else:
                mean = (a[j-1, 0] + a[j-2, 0]) / 2.
                Pstart = a[j-2, 0]
                Pend = a[j-1, 0]
                tstart = a[j-2, 1]
                tend = a[j-1, 1]
                a[j - 2] = a[j]
                j = j - 2
                if (lrange > 0):
                    array_out[po, 0] = Pstart
                    array_out[po, 1] = Pend
                    array_out[po, 2] = tstart
                    array_out[po, 3] = tend
                    array_out[po, 4] = Pend-Pstart
                    array_out[po, 5] = tend-tstart
                    array_out[po, 6] = mean
                    array_out[po, 7] = 1.00
                    po += 1

    # partial range
    for i in range(j):
        lrange = np.fabs(a[i, 0] - a[i + 1, 0])
        mean = (a[i, 0] + a[i + 1, 0]) / 2.
        Pstart = a[i, 0]
        Pend = a[i + 1, 0]
        tstart = a[i, 1]
        tend = a[i + 1, 1]

        if (lrange > 0):
            array_out[po, 0] = Pstart
            array_out[po, 1] = Pend
            array_out[po, 2] = tstart
            array_out[po, 3] = tend
            array_out[po, 4] = Pend-Pstart
            array_out[po, 5] = tend-tstart
            array_out[po, 6] = mean
            array_out[po, 7] = uc_mult
            po += 1  
    
    # get rid of unused entries
    mask = np.ones(len(array_out), dtype=bool)
    
    for i in range(len(array_out)):
        if array_out[i, 7] == 0.0:
            mask[i]=False
    array_out = array_out[mask]
    

    angles=[0]
    
    for i in range(len(array_out)):
        n=array_out[i,4]/array_out[i,5]
        new=np.arctan(n)
        angles=np.vstack((angles,new))
    angles=angles[1:]
    angles=np.degrees(angles)
    array_out=np.c_[array_out,angles]

    """ Not in use
    angles=[]
    
    for i in range(len(array_out)):
        new=array_out[i,4]*10/array_out[i,5]
        angles.append(new)
        #else:
          #  new=math.atan2(lg[i,0],lg[i,1])
          #  angles.append(new)
    angles=np.arctan(angles)
    angles=np.degrees(angles)
    array_out=np.c_[array_out,angles]"""
    def pers(binn,array):
        freq=np.zeros((len(binn),1))
        persi=np.zeros((len(array),1))
        for k in range(len(binn)-1):
            for i in range(len(array)):
                if array[i]==binn[k]or binn[k] < array[i]< binn[k+1]:
                    freq[k]+=1 #how many times this(between the range of particular bin) occured

        for k in range(len(binn)-1):
            for i in range(len(array)):
                if array[i]==binn[k]or binn[k] < array[i]< binn[k+1]:
                    persi[i]=freq[k]
        return persi

    bins = int(1 // threshold)
    amp_bins=np.linspace(-1,1,bins)
    a=array_out[:,4]
    amp_pers=pers(amp_bins,a)
    
    angle_bins=np.linspace(-90,90,bins)
    angle_pers=pers(angle_bins,angles)
    
    max_time=max(array_out[:,5])
    time_bins=np.linspace(1,max_time,bins)
    b=array_out[:,5]
    time_pers=pers(time_bins,b)
    
    
    unique_amp=np.unique(array_out[:,4:],axis=0)
    
    #count of cycles with amplitude
    cycles=np.zeros((len(unique_amp),1))
    for k in range(len(unique_amp)):
        for i in range(len(array_out)):
            if array_out[i,4]==unique_amp[k,0]:
                cycles[k]+=array_out[i,7]
    
    cyclecount=np.zeros([len(array_out),1])
    for k in range(len(unique_amp)):
        for i in range(len(array_out)):
            if array_out[i,4]==unique_amp[k,0]:
                cyclecount[i]=cycles[k]
    
    #persistance=np.hstack((amp_pers,time_pers,angle_pers,cyclecount))
    array_out=np.append(array_out,amp_pers,axis=1) 
    array_out=np.append(array_out,time_pers,axis=1) 
    array_out=np.append(array_out,angle_pers,axis=1) 
    array_out=np.append(array_out,cyclecount,axis=1) 
    Array_out=pd.DataFrame(array_out,columns=['w1', 'w2', 't1', 't2', '∆w_r', '∆t_r', 'σ_r', 'cycle', 'θ_r', 'amp_freq', 'time_freq', 'angle_freq', 'cyclecount'])
    return Array_out