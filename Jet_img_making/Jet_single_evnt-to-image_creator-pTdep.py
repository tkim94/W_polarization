#!/usr/bin/env python

import sys
import math
import os
from array import array
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.colors import LogNorm

try:
    input = raw_input
except:
    pass

if len(sys.argv) < 2:
    print("Usage: python Jet_image_creator.py (run_location)")
    sys.exit(1)

#Example :  python examples/Jet_image_creator.py /Volumes/Tae_Research-Backup/pp2wjet_run/ 10000


#Load Saved Results
jet_result = pd.read_csv(sys.argv[1]+"jet_result.csv")

# Select Random X number of events to create jet images ( Working on... )
import random
from matplotlib.colors import LogNorm

print jet_result

emptydf = pd.DataFrame()
evnt_max = max(jet_result['Event'])        # Maximum events
jet_img = pd.DataFrame()
temp_img = pd.DataFrame()
jet_img_fnl = []

for i in range(evnt_max):
    
    temp_img = jet_result[jet_result['Event']==(i+1)]
    
    # pT/E normalization
    #temp_img['pT/Etot'] = temp_img['Constituent pT']/Etot
    
    # E/Etot normalization
    #Etot = temp_img['jet E'].sum()
    #temp_img['E/Etot'] = temp_img['jet E']/Etot
    
    # E/Emax normalization
    Emax = temp_img['jet E'].max()
    temp_img['E/Emax'] = temp_img['jet E']/Emax
        
    jet_img = jet_img.append(temp_img)
    temp_img = emptydf
    
    xdata = jet_img['Qx']
    ydata = jet_img['Qy']
    #zdata = jet_img['pT/Etot']
    #zdata = jet_img['E/Etot']
    zdata = jet_img['E/Emax']
    #print jet_img.shape
    h,xedges,yedges,image = plt.hist2d(xdata,ydata,bins=[20,20],range=[[-1,1],[-1,1]],weights=zdata,cmap='rainbow')
    jet_img_fnl.append(h)
    #plt.colorbar()
    #plt.show()
    #print(h)
    jet_img = emptydf


# Regain ndarray format    
jet_img_arry = np.array(jet_img_fnl)

# Save Results
#np.save(sys.argv[1]+"jet_img_Etot-Norm-"+str(evnt_in_img)+"-W-events_"+sys.argv[4]+".npy", jet_img_arry)
#np.save(sys.argv[1]+"jet_img_Emax-Norm-single-W-events.npy", jet_img_arry)
# Way to load : np.load(".npy file")




#input("Press Enter to continue...")

