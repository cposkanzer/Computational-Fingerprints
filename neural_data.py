import sys, os, time
import numpy as np
from scipy import stats

def data_agg(sub):
    #Establish path to data
    ROI_dir = '/gsfs0/data/poskanzc/MVPN/data/' + sub + '/face_ROIs/' 
    cb_map = np.load('/gsfs0/data/poskanzc/FingerPrint_Project/masks/cbidx.npy')
    NULL = True
    for run in range(1,9):
        FFA_data = np.load(ROI_dir + 'run_' + str(run) + '_ROIs/FFA_denoised_functional.npy')               
        GM_data_run = np.load(ROI_dir + 'run_' + str(run) + '_ROIs/GM_denoised_functional.npy')
        
        #delete ffa voxels without change
        sds = np.std(FFA_data,0)
        FFA = np.delete(FFA_data, np.nonzero(sds==0)[0],1)

        #delete cerebellum from GM
        cb_idx = np.nonzero(cb_map==0)[0]
        GM = np.delete(GM_data_run,cb_idx,1)

        ROI_data_run = np.concatenate([FFA], 1)       
        avg_roi = np.mean(ROI_data_run,1) #data is shape time x voxels
        ROI_data_norm = stats.zscore(avg_roi)
        GM_data_norm = stats.zscore(GM,0)
        if NULL:
            ROI_data = ROI_data_norm
            GM_data = GM_data_norm
            NULL = False
        else: 
            ROI_data = np.concatenate([ROI_data, ROI_data_norm], 0) 
            GM_data = np.concatenate([GM_data, GM_data_norm], 0)
    print("ROI_shape:", np.shape(ROI_data))
    print("GM_shape:", np.shape(GM_data))        

    return ROI_data, GM_data
# number of GM voxels in mask = 49087


