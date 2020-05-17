#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:52:08 2018

@author: deniz  --- changed by Miriam Henning 

Changed: Add White noise option diagonal 
"""

import h5py #use this  for hdf5 v7.
import os
from scipy import interpolate
import scipy.io as sio
import numpy as np
import tkinter
import tkinter.filedialog as filedialog
from scipy.signal import butter, filtfilt
import re
import warnings



def get_stim(stimpath):
    """Returns whitenoise stimulus values from the MATLAB file.

    Parameters
    ----------
    stimpath : str
        The file adress of the MATLAB document. *(i.e. 'C:\file\stimulus.mat')*

    Returns
    -------
    3D array (mxnxj)
        Array of whitenoise stimulus values where -1, 0 and 1 mean black grey
        and white. The axes m indexes the frames and n shows the bar number.
        j axis mostly has length of 1 due to MATLAB rendering of arrays.
    """

    stim = h5py.File(stimpath)
    #stim file has group names 'stimulus' and 'stimulusMetadata'
    #to reach group keys run list(stim.keys())
    stim = stim['stimulus'][()]  ##stim is np array with frames
    return stim

def get_fileloc(title, initialdir):
    """ Obtains the data path of recordings by using a graphical user
    interface (GUI).

    Parameters
    ----------
    title : str
        The title of the GUI window.
    initaldir : str
        The directory GUI initiates. Default is my pathway for the file with
        pData.

    Returns
    -------
    fileloc
        The path to the file chosen by the user through GUI.

    """
    root = tkinter.Tk()
    root.withdraw()
    fileloc = filedialog.askopenfilename(initialdir=initialdir,title=title)
    return fileloc

def get_dirloc(title, initialdir):
    """ Obtains a directory path by using a graphical user
    interface (GUI).

    Parameters
    ----------
    title : str
        The title of the GUI window.
    initaldir : str
        The directory GUI initiates. Default is my pathway for the file with
        pData.

    Returns
    -------
    dirloc
        The path to the directory chosen by the user through GUI.

    """
    root = tkinter.Tk()
    root.withdraw()
    fileloc = filedialog.askdirectory(initialdir=initialdir,title=title)
    return fileloc

def get_parent(path, order=1):
    """Obtains the parent paths of any order.

    Parameters
    ----------
    path : str
        Pathway of the file or folder to obtain the parent path.
    order : int
        Order of the parent directory. For example, order of two yields the
        parent of the parent of the path.

    Returns
    -------
    parentpath : str
        ..

    Example
    -------
    >>> get_parent('parent2/parent1/file', 1)
    parent1
    >>> get_parent('parent2/parent1/file', 2)
    parent2

    """
    for i in range(order):
        parentpath = os.path.dirname(os.path.normpath(path))
        path = parentpath
    return parentpath

def mattopython(pdatapath, mergedROI=False, manualSelectLayer = False):
    """Extracts the data from the MATLAB pData file to Python arrays.

    .. note::
        This function requires the data_file.mat file located in the same
        directory as pData file to extract aligned image information which
        is not always saved in the pData.

    Parameters
    ----------
    pdatapath : `str`
        The file adress of the pData file.

    mergedROI : `boolean` (optional)
        When `True`, it extracts the data of merged ROI from Cluster Analysis
        Code (refer to Miriam Henning). Default is False

    Returns
    -------
        A `tuple` with the data called ch1amax, ch1alast, imagetimes,
        imagesamplerate, stimframes, imframenos, stimtimes, stimtype, dsignal,
        masks, stimframes100hz, ClusterAnalysis

    ch1amax : `array`
        Maximum projection values of the first 2700 image frames.
    ch1alast : `array`
        Maximum projection values of the last 2700 image frames.
    imagetimes : `array`

    """
    # Derive parent directory as imagedir
    imagedir = get_parent(pdatapath)

    # Using datafile in parent to extract imaging parameters
    datmat = h5py.File(imagedir + os.sep + 'data_file_SIMA_only_m.mat','r')   # changed to get SIMA data file (mh, 05.03.19)
    # Extract maximum projection of aligned images from first 2700 images
    ch1a = datmat['out']['ch1']                                               # changed to ch1 instead of ch1a, because name was changed in SIMA alignm. (mh, 05.03.19)
    ch1amax = np.amax(ch1a[:2700],axis=0)
    ch1amax = np.transpose(ch1amax)
    # Extract maximum projection of aligned images from last 2700 images
    ch1alast = np.amax(ch1a[-2700:],axis=0)
    ch1alast = np.transpose(ch1alast)
    imagetimes = datmat['out']['xml']['relativeFrameTimes'][0]

    imagesamplerate =  datmat['out']['xml']['framerate'][0]

    # Loading pData parameters
    # All zero indexings are due to the MATLAB file design

    # Different versions of mat files have different file hierarchy
    try:   # If this data loading works, version is lower than 7.3
        a=sio.loadmat(pdatapath)['strct'][0]
        matlab_v7_3=False
    except:     # Otherwise version higher than 7.3
        a= h5py.File(pdatapath,'r')['strct']
        matlab_v7_3=True

    stimframes = a['fstimpos2']
    imframenos = a['frame_nums']
    stimtimes = a['stimTimes']
    stimtype = a['stim_type']

    # Version specific data extraction
    if not matlab_v7_3:
        stimframes= stimframes[0]
        imframenos = imframenos[0]
        stimtimes = stimtimes[0]
        stimtype = stimtype[0]
    elif matlab_v7_3:
        stimframes = np.transpose(stimframes.value)
        imframenos = np.transpose(imframenos.value)
        stimtimes = np.transpose(stimtimes.value)
        stimtype = [''.join(chr(c) for c in stimtype.value)]

    stimframes = stimframes[:,0]
    imframenos=imframenos[:,0][1:]  #[1:] removes first frame
    imframenos = imframenos[imframenos<=len(stimframes)]
    #above indexing removes excess image frames recorded in stimulus recorder
    #i.e. you image 150,stimoutput goes till 152
    stimtimes = stimtimes[:,0][1:len(imframenos)+1] #remove first time value

    # Finding whether the pData is coming from cluster analysis or not by
    # by checking group names like 'Layer' exist in the MATLAB structure
    try:
        groupkeys = a.dtype.names
    except:
        groupkeys = list(a.keys())

    if 'Layer' or 'ClusterInfo' in groupkeys:
        ClusterAnalysis = True
        try :
            a['ClusterInfo']
        except ValueError:
            ClusterAnalysis = False
            
    elif manualSelectLayer:
        ClusterAnalysis = True
    else:
        ClusterAnalysis = False
    if ClusterAnalysis:
        if 'ClusterInfo' in groupkeys and not manualSelectLayer:
            # Miriam now groups cluster data in ClusterInfo group
            a = a['ClusterInfo']
        if manualSelectLayer:
            a = a['ClusterInfo_ManuallySelect']
        if mergedROI:
            # Miriam now merges clusters which might arise from the same cell.
            # Refer to Miriam for details
            layer = a['Layer_m']
            dsignal = a['avSignal1_CA_m']
            masks = a['masks_CA_m']
            t4t5 = a['T4_T5_m']
        else:
            layer = a['Layer']
            dsignal = a['avSignal1_CA'] # I changed it to avSignal, because the bg-subtracted trace is sometimes really noisy and we dont need to do this here! (mh, 11.02.19)
            masks = a['masks_CA']
            t4t5 = a['T4_T5']
    else:
        dsignal = a['avSignal1']
        masks = a['masks']

    # Data rearrangement for mat files of version earlier than 7.3
    if not matlab_v7_3:
        masks = masks[0][0]
        masks = np.array(list(masks),dtype=np.uint8) #object to array conversion
        dsignal = dsignal[0]
        if ClusterAnalysis:
            layer = layer[0]
            t4t5 = t4t5[0]

    # Data rearrangement for mat files of version later than 7.3
    elif matlab_v7_3:
        masks = masks.value
        masklist = []
        for i in range(masks.shape[0]):
            masklist.append(np.transpose(a[masks[i,0]]))
        masks = np.array(masklist,dtype=np.uint0)
        dsignal = np.transpose(dsignal.value)
        if ClusterAnalysis:
            layer = np.transpose(layer.value)
            t4t5 = np.transpose(t4t5.value)

    # Finding stimulus_output file,loading 100Hz timing of stimulus change
    files = os.listdir(imagedir)
    for file in files:
        if file.startswith('_stim'):
            stimoutdir = os.path.join(imagedir,file)
    stimframes100hz = np.genfromtxt(stimoutdir,
                                    skip_header=1)[:,6][1:1+len(imframenos)]
    stimframes100hz = np.array(stimframes100hz,dtype='float32')


    if 't4t5' in locals():
        return (ch1amax, ch1alast, imagetimes, imagesamplerate, stimframes,
            imframenos, stimtimes, stimtype, dsignal, masks, stimframes100hz,
            ClusterAnalysis, t4t5, layer)
    else:
        return (ch1amax, ch1alast, imagetimes, imagesamplerate, stimframes,
            imframenos, stimtimes, stimtype, dsignal, masks, stimframes100hz,
            ClusterAnalysis)

def interpolate_data(stimtimes, stimframes100hz, dsignal, imagetimes, freq):
    """Interpolates the stimulus frame numbers (*stimframes100hz*), signal
    traces (*dsignal*) by using the
    stimulus time (*stimtimes*)  and the image time stamps (*imagetimes*)
    recorded. Interpolation is done to a frequency (*freq*) defined by the
    user.
    recorded in

    Parameters
    ----------
    stimtimes : 1D array
        Stimulus time stamps obtained from stimulus_output file (with the
        rate of ~100Hz)
    stimframes100hz : 1D array
        Stimulus frame numbers through recording (with the rate of ~100Hz)
    dsignal : mxn 2D array
        Fluorescence responses of each ROI. Axis m is the number of ROIs while
        n is the time points of microscope recording with lower rate (10-15Hz)
    imagetimes : 1D array
        The time stamps of the image frames with the microscope recording rate
    freq : int
        The desired frequency to interpolate

    Returns
    -------
    newstimtimes : 1D array
        Stimulus time stamps with the rate of *freq*
    dsignal : mxn 2D array
        Fluorescence responses of each ROI with the rate of *freq*
    imagetimes : 1D array
        The time stamps of the image frames with the rate of *freq*
    """
    # Interpolation of stimulus frames and responses to freq

    # Creating time vectors of original 100 Hz(x) and freq Hz sampled(xi)
    # x = vector with 100Hz rate, xi = vector with user input rate (freq)
    x = np.linspace(0,len(stimtimes),len(stimtimes))
    xi = np.linspace(0,len(stimtimes),
                     np.round((np.max(stimtimes)-np.min(stimtimes))*freq)+1)

    # Get interpolated stimulus times for 20Hz
    # stimtimes and x has same rate (100Hz)
    # and newstimtimes is interpolated output of xi vector
    newstimtimes = np.interp(xi, x, stimtimes)
    newstimtimes =  np.array(newstimtimes,dtype='float32')

    # Get interpolated stimulus frame numbers for 20Hz
    # Below stimframes is a continuous function with stimtimes as x and
    # stimframes100Hz as y values
    stimframes = interpolate.interp1d(stimtimes,stimframes100hz,kind='nearest')
    # Below interpolated stimulus times are given as x values to the stimtimes
    # function to find interpolated stimulus frames (y value)
    stimframes = stimframes(newstimtimes)
    stimframes = stimframes.astype('int')

    #Get interpolated responses for 20Hz
    dsignal1 = np.empty(shape=(dsignal.shape[0],
                               len(newstimtimes)),dtype=dsignal.dtype)
    for i in range(dsignal.shape[0]):
        dsignal1[i]=np.interp(newstimtimes, imagetimes, dsignal[i])
    dsignal = dsignal1

    return (newstimtimes, dsignal, stimframes)

def calculate_df(dsignal,grayinterleave):
    """Calculates the """
    df = np.zeros(shape=dsignal.shape, dtype=dsignal.dtype)
    for i in range(len(dsignal)):
        F = dsignal[i][grayinterleave]
        # Find mean fluorescence of a mask in grey stim
        F = np.mean(F)
        df[i] = (dsignal[i]-F)/np.abs(F)
    return df

def find_flyid(pdatapath):
    # Finding fly ID as flyno, date and imageno to save the variables
    flyno = re.search('fly\d+', pdatapath, re.I).group(0)
    date = re.search('\d{6}', pdatapath, re.I).group(0)
    imageno = re.search('image\d+', pdatapath, re.I).group(0)
    filename = '_'.join([date, flyno, imageno]) # Output file name
    return filename

def saveprocesseddata(pdatapath, outfolder, stimpath,
                      pick = False, filt = False, test=False):
    # Get stimulus file
    stim = get_stim(stimpath)
    #-----------------Obtaining data variables from pData file-------------
    # Convert the MATLAB information to python arrays
    converteddata = mattopython(pdatapath)

    ch1amax, ch1alast, imagetimes, imagesamplerate, stimframes,\
    imframenos, stimtimes, stimtype, dsignal, masks, stimframes100hz,\
    ClusterAnalysis = converteddata[0:12]
    if len(converteddata)>12:
        t4t5, layer = converteddata[-2:]

    #--- Interpolation of stimulus time, frames and ROI responses to 20Hz------
    freq = 20 # The update rate of stimulus frames is 20Hz
    newstimtimes, dsignal, stimframes = interpolate_data(stimtimes,
                                                         stimframes100hz,
                                                         dsignal,
                                                         imagetimes,
                                                         freq)
    # OPTIMIZATION: Discarding data points which might be noise
    # If you are not interested in discarding data,
    # give 0 or False value to pick arg.
    if pick:
        # The criterion for noise detection is chosen as deviation
        # from the mean of global (moving average) and local
        # (time window mean) responses.

        # Globally discarding data
        # Find moving average trace with a window consisting of 501 time points
        cherryboolean = np.zeros(shape=dsignal.shape)
        weights = np.ones(501)/501  # the window has lenght of 501
        discardind = []
        # Loop in the dsignal from each ROI
        # to find discarded point and their idx
        for i in range(dsignal.shape[0]):
            # Convolve the trace with the window
            rolled=np.convolve(dsignal[i],weights,'valid')
            # Find boolean array of points exceeding 3std(exceeding = 0, other = 1)
            bools = (rolled < np.mean(rolled)
            +3*np.std(rolled))&(rolled > np.mean(rolled)
            -3*np.std(rolled)).astype(int)
            # Due to convolution nature, the points at beginning and end are cut
            # off. The size of cuts are (windowsize-1)/2 in this case 250 with
            # 501 window size. We add these points back below to avoid index error
            cherryboolean[i] = np.concatenate((np.ones(250),bools,np.ones(250)))

            # Find the start and end of the discarded points
            ibooleanpad = np.concatenate(([1],cherryboolean[i],[1]))
            changes = np.abs(np.diff(ibooleanpad))
            changesind = np.where(changes==1)[0].reshape(-1,2)

            # Padding the discarded intervals with --------------------------------------------
            paddingind = np.concatenate(((changesind+[-2,1]).reshape(changesind.size),
                                         (changesind+[-1,0]).reshape(changesind.size)))
            paddingind[paddingind<0]=0
            paddingind[paddingind>=dsignal.shape[1]]=dsignal.shape[1]-1
            discardind.append(paddingind)

        # MA is used as acronym of Moving Average
        discardindMA = np.copy(discardind)
        cherrybooleanMA = np.copy(cherryboolean.astype(np.int8))

        # Locally discarding data on top of the global discard

        # The window size is 3 minutes
        winsize = 20*3*60
        bins = int(dsignal.shape[1]/winsize) + 2

        for i in range(dsignal.shape[0]):
            iboolean = np.empty(shape=dsignal.shape[1])
            for ii in range(1,bins):
                if ii == bins-1:
                    dstd = 3*np.std(dsignal[i][winsize*(ii-1):])
                    dmean = np.mean(dsignal[i][winsize*(ii-1):])
                    extremevalues = ((dsignal[i][winsize*(ii-1):]>(dmean+dstd))\
                                     |(dsignal[i][winsize*(ii-1):]<(dmean-dstd))\
                                     ).astype(int)
                    iboolean[winsize*(ii-1):] = extremevalues
                else:
                    dstd = 3*np.std(dsignal[i][winsize*(ii-1):ii*winsize])
                    dmean = np.mean(dsignal[i][winsize*(ii-1):ii*winsize])
                    extremevalues = ((dsignal[i][winsize*(ii-1):ii*winsize]>\
                                      (dmean+dstd))|(dsignal[i][winsize*(ii-1):ii*winsize]\
                                      <(dmean-dstd))).astype(int)
                    iboolean[winsize*(ii-1):ii*winsize] = extremevalues
            # Find the start and end of the discarded points
            ibooleanpad = np.concatenate(([0],iboolean,[0]))
            changes = np.abs(np.diff(ibooleanpad))
            changesind = np.where(changes==1)[0].reshape(-1,2)
            paddingind = np.concatenate(((changesind+[-2,1]).reshape(changesind.size),
                                         (changesind+[-1,0]).reshape(changesind.size)))
            paddingind[paddingind<0]=0
            paddingind[paddingind>=dsignal.shape[1]]=dsignal.shape[1]-1
            discardind[i] = np.append(discardind[i],paddingind)
            np.put(iboolean,paddingind,1)
            iboolean = np.logical_not(iboolean)
            cherryboolean[i]=np.multiply(iboolean,cherryboolean[i])

        discardind = discardind
        cherryboolean = cherryboolean.astype(np.int8)


    # OPTIMIZATION: Applying filter to the signal traces

    if filt:
        # Ignore the warning of badly fitting polynomials
        # The fit gets better when one puts x-mean(x) in polyfit but
        # I saw that it introduces dramatic trend when there is a big deviation
        # in the trace, so I ignored the warning for that
        warnings.simplefilter('ignore', np.RankWarning)

        # Fitting 4th order polynomial to detrend (Polynomial filtering)
        polytrace = np.zeros(shape=dsignal.shape)
        for i in range(dsignal.shape[0]):
            fit = np.polyval(np.polyfit(newstimtimes, dsignal[i], 4),
                             newstimtimes)
            polytrace[i]=dsignal[i]-fit+np.mean(dsignal[i])

        # Butterworth filtering
        buttertrace = np.zeros(shape=dsignal.shape,
                            dtype=dsignal.dtype)
        # butternomean array had butterworth filtered array without mean addition
        # The filtering removes the DC component, hence mean.
        butternomean=np.zeros(shape=dsignal.shape,dtype=dsignal.dtype)
        for i in range(dsignal.shape[0]):
            b, a = butter(4, 0.1/(imagesamplerate/2), btype='highpass')
            butternomean[i] = filtfilt(b,a,dsignal[i])
            buttertrace[i] = butternomean[i]+np.mean(dsignal[i])

    # Calculate df/F

    # Find indices where grey(0) stim is on
    grayinterleave = np.where(stimframes==0)
    df = calculate_df(dsignal,grayinterleave)

    if filt:
        butterdf = calculate_df(buttertrace, grayinterleave)
        butternodf = calculate_df(butternomean, grayinterleave)
        polydf = calculate_df(polytrace, grayinterleave)

    #Save variables

    # Compress data from 64 bit to 32
    stimtimes = np.array(stimtimes,dtype='float32')
    ch1amax = np.array(ch1amax,dtype='float32')
    imagetimes = np.array(imagetimes,dtype='float32')
    dsignal = np.array(dsignal, dtype = 'float32')

    keystosave = ['df', 'dsignal', 'stimframes',
                  'newstimtimes', 'masks', 'ch1amax',
                  'stimtype', 'imagesamplerate', 'ClusterAnalysis']
    if pick:
        keystosave.extend(['cherryboolean', 'discardindMA',
                           'cherrybooleanMA', 'discardind'])
    if filt:
        keystosave.extend(['butterdf', 'polydf', 'butternodf',
                           'polytrace', 'butternomean', 'buttertrace'])
    if ClusterAnalysis:
        keystosave.extend(['layer','t4t5'])
    datadict = {}
    # Loop in key names to save the local variables
    for key in keystosave:
        val = locals()[key]
        datadict.update({key:val})

    # Finding fly ID as flyno, date and imageno to save the variables

    filename = find_flyid(pdatapath) # Output file name
    # For the differently named pdata files from
    # different analysis, analysistype is extracted
    # i.e. Fly2_pData_CA.mat where pData_CA is analysistype
    analysistype = re.search('pdata.*\.mat', pdatapath, re.I).group(0)
    analysistype = analysistype[:-4] #exclude extension
    if len(analysistype) > 5:
        filename += analysistype[5:]
    newfolder = '1-Analyzed_Data'
    dest = os.path.join(outfolder,newfolder)
    if newfolder not in os.listdir(outfolder):
        os.mkdir(dest)
    outpath = os.path.join(dest, filename)
    if test:
        return (df, dsignal, stimframes, newstimtimes, masks, ch1amax,
 stimtype, imagesamplerate, ClusterAnalysis, layer, t4t5)
    else:
        np.savez(outpath, **datadict)

def save_rf(outfolder, snippet, analyzeportion, stimfile,
            mode='o', dataloc = ' ', pick=False,
            mergedROI = False, test = False, manualSelectLayer = False):
    # Getting data and stimulus

    # If no dataloc arg is given, the function initiates GUI for getting
    # location of the whitenoise data file
    if '.mat' not in dataloc and '.npz' not in dataloc:
        title='Please select npz/mat whitenoise data'
        dataloc = get_fileloc(title, dataloc)

    # Obtaining data from MATLAB file
    if '.mat' in dataloc:
        converteddata = mattopython(dataloc, mergedROI, manualSelectLayer)

        ch1amax, ch1alast, imagetimes, imagesamplerate, stimframes,\
        imframenos, stimtimes, stimtype, dsignal, masks, stimframes100hz,\
        ClusterAnalysis = converteddata[0:12]
        if len(converteddata)>12:
            t4t5, layer = converteddata[-2:]

        #Interpolation of stimulus time, frames and ROI responses to 20Hz
        freq = 20 # The update rate of stimulus frames is 20Hz
        newstimtimes, dsignal, stimframes = interpolate_data(stimtimes,
                                                         stimframes100hz,
                                                         dsignal,
                                                         imagetimes,
                                                         freq)
        grayinterleave = np.where(stimframes==0)
        df = calculate_df(dsignal,grayinterleave)

    # Obtaining data from npz Python files of the folder 1-Analyzed_Data
    elif '.npz' in dataloc:
        with np.load(dataloc) as data:
            #       ClusterAnalysis = data['ClusterAnalysis']
#            if not ClusterAnalysis:
#                title = 'The file does not have cluster analysis\
#                      data, please select again.'
#                dataloc = get_fileloc(title, sourcefolder)
            # Checking if the data is from cluster analysis
            # maybe implement stim type???------------------------------------------------
            if 'layer' in data.keys():
                ClusterAnalysis = True
                layer = data['layer']

            stimframes, stimtype = data['stimframes'], data['stimtype'][0]

            if mode == 'b':
                df = data['butterdf']
            elif mode == 'bnm':
                df = data['butternodf']
            elif mode == 'p':
                df = data['polydf']
            elif mode == 'o':
                df = data['df']
            else:
                raise ValueError('Invalid mode arguement: %s'%mode)

            if pick:
                cherryboolean = data['cherryboolean'].astype(bool)
                cherrybooleanMA = data['cherrybooleanMA'].astype(bool)
    else:
        raise ValueError('The file should have either .npz or .mat extensions\
                         \n%s'%dataloc)

    # Getting stimulus information

    stim = h5py.File(stimfile)
    #This file has group names 'stimulus' and 'stimulusMetadata'
    #list(stim.keys())
    stim = stim['stimulus'][()]  # stim is np array with frames

    # Finding where the stim frames is more preceeding frame length
    booleans = stimframes>=snippet
    # From stimulus name(stimtype), extracting data of relevant layers
    if ClusterAnalysis:
        if '23_El' in str(stimtype):
            # Elevation whitenoise will elicit reponse in 1st and 2nd layers
            layerno1,layerno2 = 1, 2
        elif '23_Az' in str(stimtype):
            # Azimuth whitenoise will elicit reponse in 3rd and 4th layers
            layerno1,layerno2 = 3, 4
        elif 'Dg_leftU' in str(stimtype):
            layerno1,layerno2 = 1, 2
        elif 'Dg_rightU' in str(stimtype):
            layerno1,layerno2 = 3, 4  
        else:
            raise ValueError('The stimulus not understood %s'%stimtype)
        # Find indexes with appropriate layer identity
        roi_ind = np.where((layer == layerno1)|(layer==layerno2))[0]
    else:
        # If not cluster analysis, get every drawn ROI
        roi_ind = np.arange(df.shape[0])

    # Extracting portion of data to be used

    # Padding stimulus frames by 0 (grey interleave) values
    padframes = np.concatenate(([0],stimframes,[0]))
    # Finding the points where frames shift between grey to frame numbers
    difs = np.diff(padframes>0)
    # Finding the indices of epoch start and end
    # rows = epoch no, columns = start and end
    epochind=np.where(difs==1)[0].reshape(-1,2)

    # If only one epoch is used for RF(STA) calculation
    if type(analyzeportion) is int:
        pyind = analyzeportion-1 #turn epoch no to python indx (i.e. 1 to 0)
        analyzelimit = epochind[pyind,:]    #where to splice
        # Splicing data
        stimframesused = stimframes[analyzelimit[0]:analyzelimit[1]]
        dfused = df[:,analyzelimit[0]:analyzelimit[1]]
    # If more epochs are used
    elif type(analyzeportion) is list:
        analyzeportion.sort()
        pyinds = [i-1 for i in analyzeportion]
        epochused = epochind[pyinds,:]
        # Splicing data
        stimframesused = stimframes[epochind[0,0]:epochind[0,1]]
        dfused = df[:,epochind[0,0]:epochind[0,1]]
        # Concatenating data from different epochs
        for i in range(1,len(analyzeportion)):
            stimframesused = np.concatenate((stimframesused,
                            stimframes[epochind[i,0]:epochind[i,1]]))
            dfused = np.concatenate((dfused,
                            df[:, epochind[i,0]:epochind[i,1]]), axis = 1)
    else:
        raise ValueError('Invalid analyzeportion arguement: %s'%
                         str(analyzeportion))
    # Used stimulus is sliced by using the frame information
    stimused = stim[np.min(stimframesused)-1:np.max(stimframesused),:,:]
    # Get unique frame numbers from trials
    uniquestim = np.unique(stimframesused)
    # Find frame nos grater than preceeding time window as a boolean matrix
    boolean2 = uniquestim >= snippet

    # Efficiently calculate RFs(stas) with mean subtracted stimulus
    # and mean substracted df

    #centering df (mean substraction)
    
    centereddf = (dfused[roi_ind].T-np.mean(dfused[roi_ind],axis=1)).T
    #centereddf = dfused[roi_ind] # no centering (mh, 11.02.19)

    #centering stimulus (mean substraction)
    stimused = stimused - np.mean(stimused,axis=0)
    #stimused = stimused - 0.5 #shift around 0 (mh, 11.02.19) 

    stas = np.empty(shape=(len(roi_ind),snippet,stimused.shape[1]))
    avg = np.zeros(shape=(len(roi_ind),snippet,stimused.shape[1]))
    # For loop iterates through different stimulus frame numbers
    # Finds the data where the specific frame is shown and calculates
    # sta
    for ii in range(snippet-1,len(uniquestim)):
        # Calculate means of responses to specific frame with ii index
        responsechunk = centereddf[:,np.where(stimframesused==uniquestim[ii])[0]]
        responsemean = np.mean(responsechunk,axis=1)
        # Create a tiled matrix for fast calculation
        response = np.tile(responsemean,(stimused.shape[1],snippet,1)).T
        # Find the stimulus values in the window and get tiled matrix
        stimsnip = stimused[uniquestim[ii]-snippet:uniquestim[ii],:,0]
        stimsnip = np.tile(stimsnip,(len(response),1,1))
        # Fast calculation is actually is a multiplication
        avg += np.multiply(response,stimsnip)
    stas = avg/np.sum(boolean2)     # Average with the number of additions

    # Calculating RF info of the data with discarded points
    # (see saveprocesseddata)
    if pick:

        # Calculate sta for globally and locally discarded data
        stapick = np.zeros(shape=(len(roi_ind),snippet,stim.shape[1]))
        stapickMA = np.zeros(shape=(len(roi_ind),snippet,stim.shape[1]))
        for i in range(len(roi_ind)):
            # There may be error since dfused and cherrybooleans will not match
            # in analyzeportions less than 1
            dd = dfused[roi_ind[i]]
            centereddf =  dd-np.mean(dd[cherryboolean[roi_ind[i]]])
            avgpick = np.zeros(shape=(1,snippet,stim.shape[1]))
            #(np.where(np.diff(boolean2)==1)[0]+1)[0]
            for ii in range(snippet-1,len(uniquestim)):
                responseind = np.where((stimframesused==uniquestim[ii])&
                                       (cherryboolean[roi_ind[i]]>0))
                if responseind[0].size>0:
                    responsechunk = centereddf[responseind]
                    responsemean = np.mean(responsechunk)
                    stimsnip = stimused[uniquestim[ii]-snippet:uniquestim[ii],:,:]
                    stimsnip = stimsnip.reshape(1,snippet,stim.shape[1])
                    avgpick += np.multiply(responsemean,stimsnip)
            stapick[i,:,:] = avgpick/np.sum(boolean2)

        # Calculate sta for only globally discarded data
        for i in range(len(roi_ind)):
            # There may be error since dfused and cherrybooleans will not match
            # in analyzeportions less than 1
            dd = dfused[roi_ind[i]]
            centereddf =  dd-np.mean(dd[cherrybooleanMA[roi_ind[i]]])
            avgpick = np.zeros(shape=(1,snippet,stim.shape[1]))
            for ii in range(snippet-1,len(uniquestim)):
                responseind = np.where((stimframesused==uniquestim[ii])&
                                       (cherrybooleanMA[roi_ind[i]]>0))
                if responseind[0].size>0:
                    responsechunk = centereddf[responseind]
                    responsemean = np.mean(responsechunk)
                    stimsnip = stimused[uniquestim[ii]-snippet:uniquestim[ii],:,:]
                    stimsnip = stimsnip.reshape(1,snippet,stim.shape[1])
                    avgpick += np.multiply(responsemean,stimsnip)
            stapickMA[i,:,:] = avgpick/np.sum(boolean2)


    # Calculating predictions
    # At the end there will be two dictionaries, pred and corrcoefs
    # The keys of the dictionaries are organized as 'pred1' for 1st epoch etc.
    offset = snippet-1

    # Trial length of each epoch
    trialslength = epochind[:,1] - epochind[:,0]
    pred = {}
    df_chosen = df[roi_ind,:]
    # Looping in each epoch trial to get predictions
    for i in range(epochind.shape[0]):
        # Calculating stimulus average
        # I calculate it for each epoch since the last epoch may have
        # less data points due to rounded frame number given in two photon
        start = epochind[i,0] # index of epoch start
        end = epochind[i,1] #index of epoch end
        fstart = stimframes[start]-1 #stimulus frame in the epoch start
        fend = stimframes[end-1] #stimulus frame in the epoch end
        stimavg = stim[fstart:fend,:,:]
        stimavg = stimavg - np.mean(stimavg,axis=0)

        key = 'pred' + str(i+1) # keys of prediction i.e. 'pred1' for 1st epoch
        prediction = np.zeros(shape=(len(roi_ind),trialslength[i]-offset),
                              dtype='float32')
        # Loop to multiply sta with stimulus windows
        # TODO: get rid of that for loop below
        # The stimpreceed should be all preceeding windows in each column/row
        # of the matrix so that with a simple multiplication you get the whole
        # predictions
        # (The method is similar to usage of repmat function in MATLAB)
        # you can check these functions:
        # numpy.lib.stride_tricks.as_strided,
        # scipy.linalg.hankel or np.einsum
        for j in range(start+offset, end):
            stimpreceed = stimavg[stimframes[j]-snippet:stimframes[j],:,0]
            prediction[:,j-start-offset] = np.sum(stimpreceed*stas, axis = (1, 2))
        pred[key] = prediction

    # Correlation coefficient calculation
    corrcoefs = np.empty(shape=(epochind.shape[0],len(roi_ind)),dtype='float32')
    for i in range(epochind.shape[0]):
        start = epochind[i,0] # index of epoch start
        end = epochind[i,1] #index of epoch end
        predno = 'pred' + str(i+1)  # key
        # find correlation matrix
        corrcoefmat = np.corrcoef(pred[predno],df_chosen[:,start+offset:end])
        # Get diagonal of the matrix
        corrcoef = np.diagonal(corrcoefmat,offset=df_chosen.shape[0])
        corrcoefs[i,:] = corrcoef

    # Save stas (RFs) with prediction and corrcoefs

    keystosave = ['stas',\
                  'corrcoefs','epochind','roi_ind','pred']
    if pick:
        keystosave.extend(['stapick','stapickMA'])
    datadict = {}
    for key in keystosave:
        val = locals()[key]
        datadict.update({key:val})
    datadict.update(**pred)

    flyid = find_flyid(dataloc)
    masterdir = "2-Analyzed_RF" # The name of the folder to save rf data

    if type(analyzeportion) is list:
        temp = str(analyzeportion).strip('[]') # Remove brackets of analyzeportion
        temp = temp.replace(" ", "") # Remove spaces
        analyzeportion = temp.replace(",", "_") # Change commas to underscores

    outfile = flyid +\
            '_RFdata_%s_epoch'%(str(analyzeportion))

    if mode != 'o': # Only when the filtered data is used, write it in filename
        outfile = "_".join(outfile, mode)
    if pick:
        outfile = "_".join(outfile, "picked")
    if masterdir not in os.listdir(outfolder): # Generate folder
        os.mkdir(os.path.join(outfolder,masterdir))
    masterdir = os.path.join(outfolder,masterdir)
    if test:
        return (stas, corrcoefs, epochind, roi_ind , pred)
    else:
        #np.savez(masterdir+os.sep+outfile+"_noMsub", **datadict) # Save (no mean subtratcted, mh 11.02.19)
        np.savez(masterdir+os.sep+outfile, **datadict) # Save (no mean subtratcted, mh 11.02.19)

