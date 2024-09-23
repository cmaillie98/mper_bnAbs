#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os, numpy as np, re
from prody import *
from collections import defaultdict
from prody import * 
import matplotlib.pyplot as plt
from numpy.linalg import svd
import scipy.stats
import pandas as pd
from itertools import combinations
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import binom_test
import time
from scipy import stats
from matplotlib.lines import Line2D                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
from collections import Counter
import csv 
plt.rcParams['font.sans-serif'] = "Arial"


# In[2]:


def calcRMS(x, axis=None):
    return np.sqrt(np.nanmean(x**2, axis=axis))


def calcRunningAvg(data, running_avg_list, running_stdev_list, frame_size): 
    for i in range(len(data)): 
        if i<(len(data)-frame_size): 
            frame_avg = sum(data[i:i+frame_size])/frame_size 
            running_avg_list.append(frame_avg) 
            frame_stdev = np.std(np.array(data[i:i+frame_size])) 
            running_stdev_list.append(frame_stdev) 
    running_avg_list = np.array(running_avg_list) 
    running_stdev_list = np.array(running_stdev_list) 
    return  


# In[3]:



def plot_phos_correlation(avg_rmsd, stdev_rmsd, plot_phos_correlation, prefix): 

    avg_rmsd = np.array(avg_rmsd)
    stdev_rmsd = np.array(stdev_rmsd)
    po4_occupancy = np.array(po4_occupancy)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 7), gridspec_kw={'height_ratios': [0.25, 6]})
    plt.subplots_adjust(hspace=0.1)
    avg_rmsd = avg_rmsd[0:5023]
    stdev_rmsd = stdev_rmsd[0:5023]

    #x = np.arange(0, len(avg_po4_xyz_RMSD_4e10_n15)*.0002, .0002)
    x = np.arange(0, len(avg_rmsd),1)

    ax1.fill_between(x, -1, 0, where=np.array(po4_occupancy[0:5023])==1, 
                     alpha=1, color='#53C767') 
    ax1.fill_between(x, -1, 0, where=np.array(po4_occupancy[0:5023])==0, 
                     alpha=1, color='#696969') 

    ax1.axhline(y=-1, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
    ax1.axhline(y=0, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
    ax1.axvline(x=5023, color='#000000', linestyle='-', ymin=0, linewidth=1)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    #ax1.set_ylabel(r"${\rm PO_4}$ Bound", fontsize=40, rotation='horizontal', 
    #               va="center", labelpad=120)
    ax1.set_xlim(-15, 5030)
    ax1.set_ylim(-1, 0)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    ax2.plot(x, avg_rmsd, lw=1, color='black')#00122E
    ax2.fill_between(x, avg_rmsd-stdev_rmsd, 
                     avg_rmsd+stdev_rmsd, 
                     color='black', alpha=0.25)
    ax2.set_xticks([0, 30, 1030, 2030, 3030, 4030, 5030]) 
    ax2.set_xticklabels([])#'', '0', '0.2', '0.4', '0.6', '0.8', '1.0' 
    ax2.set_yticks([0, 5, 10])
    ax2.set_yticklabels([])
    ax2.tick_params(axis='x', labelsize=50)
    ax2.tick_params(axis='y', labelsize=50)#, fontsize=16
    ax2.set_ylim(-.5, 12)
    ax2.set_xlim(-15, 5030)
    #ax2.set_xlabel(r"Time ($\rm \mu s$)", fontsize=60)
    #ax2.set_ylabel(r"Loop BB RMSD ($\rm \AA$)", fontsize=60) #, wrap=True / to X-Ray 
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    
    fig_name = prefix + '_po4_rmsd.png' 
    plt.savefig(fig_name, transparent=True, bbox_inches="tight")

    plt.show()
    return "Made figure: ", prefix


# In[4]:



def plot_phos_correlation_v2(avg_rmsd, stdev_rmsd, avg_rmsd_repl, stdev_rmsd_repl, po4_occupancy, prefix): 

    avg_rmsd = np.array(avg_rmsd)
    stdev_rmsd = np.array(stdev_rmsd)
    avg_rmsd_repl = np.array(avg_rmsd_repl)
    stdev_rmsd_repl = np.array(stdev_rmsd_repl)
    
    po4_occupancy = np.array(po4_occupancy)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 7), gridspec_kw={'height_ratios': [0.25, 6]})
    plt.subplots_adjust(hspace=0.1)
    avg_rmsd = avg_rmsd[0:5023]
    stdev_rmsd = stdev_rmsd[0:5023]
    avg_rmsd_repl = avg_rmsd_repl[0:5023]
    stdev_rmsd_repl = stdev_rmsd_repl[0:5023]
    #x = np.arange(0, len(avg_po4_xyz_RMSD_4e10_n15)*.0002, .0002)
    x = np.arange(0, len(avg_rmsd),1)

    ax1.fill_between(x, -1, 0, where=np.array(po4_occupancy[0:5023])==1, 
                     alpha=1, color='#53C767') 
    ax1.fill_between(x, -1, 0, where=np.array(po4_occupancy[0:5023])==0, 
                     alpha=1, color='#696969') 

    ax1.axhline(y=-1, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
    ax1.axhline(y=0, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
    ax1.axvline(x=5023, color='#000000', linestyle='-', ymin=0, linewidth=1)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    #ax1.set_ylabel(r"${\rm PO_4}$ Bound", fontsize=40, rotation='horizontal', 
    #               va="center", labelpad=120)
    ax1.set_xlim(-15, 5030)
    ax1.set_ylim(-1, 0)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    ax2.plot(x, avg_rmsd, lw=1, color='black')#00122E
    ax2.fill_between(x, avg_rmsd-stdev_rmsd, 
                     avg_rmsd+stdev_rmsd, 
                     color='black', alpha=0.25)
    
    ax2.plot(x, avg_rmsd_repl, lw=1, color='grey')#00122E
    ax2.fill_between(x, avg_rmsd_repl-stdev_rmsd_repl, 
                     avg_rmsd_repl+stdev_rmsd_repl, 
                     color='grey', alpha=0.25)
    ax2.set_xticks([0, 30, 1030, 2030, 3030, 4030, 5030]) 
    ax2.set_xticklabels(['', '0', '0.2', '0.4', '0.6', '0.8', '1.0' ])
    ax2.tick_params(axis='x', labelsize=50)
    ax2.tick_params(axis='y', labelsize=50)#, fontsize=16
    ax2.set_ylim(-.5, 2)
    ax2.set_xlim(-15, 5030)
    ax2.set_xlabel(r"Time ($\rm \mu s$)", fontsize=60)
    ax2.set_ylabel(r"Loop BB RMSD ($\rm \AA$)", fontsize=60) #, wrap=True / to X-Ray 
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    
    fig_name = prefix + '_po4_rmsd.png' 
    plt.savefig(fig_name, transparent=True, bbox_inches="tight")

    plt.show()
    return "Made figure: ", prefix


# In[7]:


#caclulate distances of loop resdiues to phosphate for crystal strucutre 
crys_5t85 = parsePDB("/Users/cmaillie/Dropbox (Scripps Research)/manuscript/pdbs/5t85.pdb")
crys_phos = crys_5t85.select( "resid 301 name P" )
crys_phos_xyz = crys_phos.getCoords() 





xray_phos_selection =  crys_5t85.select( "resid 301 name P" )
#remove gly for furute clacluations to superimpose 
l_groove_frags = crys_5t85.select(  "resnum 66 67 68 69 70 26 27 28 29 30 31  and chain L")
cdrl1_frag_xray = crys_5t85.select(  "resnum 26 27 28 29 30 31  and chain L")

#full res loop 4 is acutally the combination of all loop fragments invovlved in 108e binding site, kept variable name the same for ease of writing code 
l_groove_xray_loop_ca = l_groove_frags.select("name CA")
print(l_groove_xray_loop_ca.getResnames())


# In[ ]:





# In[60]:


wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/"
headgroup_correlation_10e8(wd+"10e8_ppm/final_analysis_input.pdb", 
                      wd+"10e8_ppm/final_analysis_traj.dcd",
                      26855, 26836, '10e8_ppm_')


# In[72]:


wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/"
headgroup_correlation_10e8(wd+"10e8_p15/final_analysis_input.pdb", 
                      wd+"10e8_p15/final_analysis_traj.dcd",
                    24895, 24876, '10e8_p15_')


# In[76]:


wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/"
headgroup_correlation_10e8(wd+"10e8_p15/final_analysis_input.pdb", 
                      wd+"10e8_p15/final_analysis_traj.dcd",
                    28647, 28628, '10e8_p15_replace')


# In[70]:


wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/"
headgroup_correlation_10e8(wd+"10e8_n15/final_analysis_input.pdb", 
                      wd+"10e8_n15/final_analysis_traj.dcd",
                    28787, 28768,'10e8_n15_')


# In[74]:


wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/"
headgroup_correlation_10e8(wd+"10e8_ppm_rep/final_analysis_input.pdb", 
                      wd+"10e8_ppm_rep/final_analysis_traj.dcd",
                    24253, 24234,'10e8_ppm_rep_')


# In[75]:


wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/"
headgroup_correlation_10e8(wd+"10e8_ppm_rep/final_analysis_input.pdb", 
                      wd+"10e8_ppm_rep/final_analysis_traj.dcd",
                    25057, 25038,'10e8_ppm_rep_replace_')


# In[14]:


#read in relevant csv data files for plotting
wd = '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts_v2/' 

po4_RMSD_10e8_ppm=[]
avg_po4_RMSD_10e8_ppm=[]
stdev_po4_RMSD_10e8_ppm = []    

nitro_RMSD_10e8_ppm=[]
avg_nitro_RMSD_10e8_ppm=[]
stdev_nitro_RMSD_10e8_ppm = [] 

headgroup_occupancy_10e8_ppm = [] 

avg_arg2po4_10e8_ppm=[]
stdev_arg2po4_10e8_ppm=[]

with open(wd+'10e8_ppm_po4_xyz_RMSD_avg.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        po4_RMSD_10e8_ppm.append(float(i[0]))

with open(wd+'10e8_ppm_po4_xyz_RMSD_avg.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        avg_po4_RMSD_10e8_ppm.append(float(i[0]))
        
with open(wd+'10e8_ppm_po4_xyz_RMSD_stdev.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        stdev_po4_RMSD_10e8_ppm.append(float(i[0]))
        


with open(wd+'10e8_ppm_nitro_xyz_RMSD.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        nitro_RMSD_10e8_ppm.append(float(i[0]))        
        
with open(wd+'10e8_ppm_nitro_xyz_RMSD_avg.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        avg_nitro_RMSD_10e8_ppm.append(float(i[0]))
        
with open(wd+'10e8_ppm_nitro_xyz_RMSD_stdev.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        stdev_nitro_RMSD_10e8_ppm.append(float(i[0]))
        

with open(wd+'10e8_ppm_headgroup_occupancy.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        headgroup_occupancy_10e8_ppm.append(float(i[0]))
        


with open(wd+'10e8_ppm_arg2po4_dist_avg.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        avg_arg2po4_10e8_ppm.append(float(i[0]))
with open(wd+'10e8_ppm_arg2po4_dist_stdev.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        stdev_arg2po4_10e8_ppm.append(float(i[0]))  
        
# plt.plot(nitro_RMSD_10e8_ppm)
plot_phos_correlation(avg_nitro_RMSD_10e8_ppm,
                      stdev_nitro_RMSD_10e8_ppm, 
                      headgroup_occupancy_10e8_ppm,
                      "10e8_ppm_headgroup") 


# In[16]:


#read in relevant csv data files for plotting
wd = '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts_v2/' 

po4_RMSD_10e8_p15=[]
avg_po4_RMSD_10e8_p15=[]
stdev_po4_RMSD_10e8_p15 = []    

nitro_RMSD_10e8_p15=[]
avg_nitro_RMSD_10e8_p15=[]
stdev_nitro_RMSD_10e8_p15 = [] 

headgroup_occupancy_10e8_p15 = [] 

avg_arg2po4_10e8_p15=[]
stdev_arg2po4_10e8_p15=[]

with open(wd+'10e8_p15_po4_xyz_RMSD_avg.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        po4_RMSD_10e8_p15.append(float(i[0]))

with open(wd+'10e8_p15_po4_xyz_RMSD_avg.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        avg_po4_RMSD_10e8_p15.append(float(i[0]))
        
with open(wd+'10e8_p15_po4_xyz_RMSD_stdev.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        stdev_po4_RMSD_10e8_p15.append(float(i[0]))
        


with open(wd+'10e8_p15_nitro_xyz_RMSD.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        nitro_RMSD_10e8_p15.append(float(i[0]))        
        
with open(wd+'10e8_p15_nitro_xyz_RMSD_avg.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        avg_nitro_RMSD_10e8_p15.append(float(i[0]))
        
with open(wd+'10e8_p15_nitro_xyz_RMSD_stdev.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        stdev_nitro_RMSD_10e8_p15.append(float(i[0]))
        

with open(wd+'10e8_p15_headgroup_occupancy.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        headgroup_occupancy_10e8_p15.append(float(i[0]))
        

avg_nitro_RMSD_10e8_p15_replace = [] 
stdev_nitro_RMSD_10e8_p15_replace = [] 
with open(wd+'10e8_p15_replacenitro_xyz_RMSD_avg.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        avg_nitro_RMSD_10e8_p15_replace.append(float(i[0]))
        
with open(wd+'10e8_p15_replacenitro_xyz_RMSD_stdev.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        stdev_nitro_RMSD_10e8_p15_replace.append(float(i[0]))
        
        
        
with open(wd+'10e8_p15_headgroup_occupancy.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        headgroup_occupancy_10e8_p15.append(float(i[0]))
  
        
# plt.plot(nitro_RMSD_10e8_p15)

plot_phos_correlation(avg_nitro_RMSD_10e8_p15,
                      stdev_nitro_RMSD_10e8_p15,
                      headgroup_occupancy_10e8_p15,
                      "10e8_p15_headgroup")   
# plot_phos_correlation_v2(avg_nitro_RMSD_10e8_p15,
#                       stdev_nitro_RMSD_10e8_p15,
#                          avg_nitro_RMSD_10e8_p15_replace,
#                       stdev_nitro_RMSD_10e8_p15_replace,
#                       headgroup_occupancy_10e8_p15,
#                       "10e8_p15_headgroup") 


# In[17]:


#read in relevant csv data files for plotting
wd = '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts_v2/' 

po4_RMSD_10e8_n15=[]
avg_po4_RMSD_10e8_n15=[]
stdev_po4_RMSD_10e8_n15 = []    

nitro_RMSD_10e8_n15=[]
avg_nitro_RMSD_10e8_n15=[]
stdev_nitro_RMSD_10e8_n15 = [] 

headgroup_occupancy_10e8_n15 = [] 

avg_arg2po4_10e8_n15=[]
stdev_arg2po4_10e8_n15=[]


with open(wd+'10e8_n15_po4_xyz_RMSD_avg.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        avg_po4_RMSD_10e8_n15.append(float(i[0]))
        
with open(wd+'10e8_n15_po4_xyz_RMSD_stdev.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        stdev_po4_RMSD_10e8_n15.append(float(i[0]))
        

     
        
with open(wd+'10e8_n15_nitro_xyz_RMSD_avg.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        avg_nitro_RMSD_10e8_n15.append(float(i[0]))
        
with open(wd+'10e8_n15_nitro_xyz_RMSD_stdev.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        stdev_nitro_RMSD_10e8_n15.append(float(i[0]))
        

with open(wd+'10e8_n15_headgroup_occupancy.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        headgroup_occupancy_10e8_n15.append(float(i[0]))
  
        
# plt.plot(nitro_RMSD_10e8_n15)
plot_phos_correlation(avg_nitro_RMSD_10e8_n15,
                      stdev_nitro_RMSD_10e8_n15, 
                      headgroup_occupancy_10e8_n15,
                      "10e8_n15_headgroup") 


# In[18]:


#read in relevant csv data files for plotting
wd = '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts_v2/' 

po4_RMSD_10e8_ppm_rep=[]
avg_po4_RMSD_10e8_ppm_rep=[]
stdev_po4_RMSD_10e8_ppm_rep = []    

nitro_RMSD_10e8_ppm_rep=[]
avg_nitro_RMSD_10e8_ppm_rep=[]
stdev_nitro_RMSD_10e8_ppm_rep = [] 

headgroup_occupancy_10e8_ppm_rep = [] 

avg_arg2po4_10e8_ppm_rep=[]
stdev_arg2po4_10e8_ppm_rep=[]


with open(wd+'10e8_ppm_rep_po4_xyz_RMSD_avg.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        avg_po4_RMSD_10e8_ppm_rep.append(float(i[0]))
        
with open(wd+'10e8_ppm_rep_po4_xyz_RMSD_stdev.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        stdev_po4_RMSD_10e8_ppm_rep.append(float(i[0]))
        

     
        
with open(wd+'10e8_ppm_rep_nitro_xyz_RMSD_avg.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        avg_nitro_RMSD_10e8_ppm_rep.append(float(i[0]))
        
with open(wd+'10e8_ppm_rep_nitro_xyz_RMSD_stdev.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        stdev_nitro_RMSD_10e8_ppm_rep.append(float(i[0]))

avg_nitro_RMSD_10e8_ppm_rep_replace = [] 
stdev_nitro_RMSD_10e8_ppm_rep_replace = [] 
with open(wd+'10e8_ppm_rep_replace_nitro_xyz_RMSD_avg.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        avg_nitro_RMSD_10e8_ppm_rep_replace.append(float(i[0]))
        
with open(wd+'10e8_ppm_rep_replace_nitro_xyz_RMSD_stdev.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        stdev_nitro_RMSD_10e8_ppm_rep_replace.append(float(i[0]))
        
        
        
with open(wd+'10e8_ppm_rep_headgroup_occupancy.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        headgroup_occupancy_10e8_ppm_rep.append(float(i[0]))
  
        
# plt.plot(nitro_RMSD_10e8_ppm_rep)
plot_phos_correlation(avg_nitro_RMSD_10e8_ppm_rep,
                      stdev_nitro_RMSD_10e8_ppm_rep,
                      headgroup_occupancy_10e8_ppm_rep,
                      "10e8_ppm_rep_headgroup")  

# plot_phos_correlation_v2(avg_nitro_RMSD_10e8_ppm_rep,
#                       stdev_nitro_RMSD_10e8_ppm_rep,
#                          avg_nitro_RMSD_10e8_ppm_rep_replace,
#                       stdev_nitro_RMSD_10e8_ppm_rep_replace,
#                       headgroup_occupancy_10e8_ppm_rep,
#                       "10e8_ppm_rep_headgroup") 


# In[82]:


print(headgroup_occupancy_10e8_ppm.count(0)/len(headgroup_occupancy_10e8_ppm_rep)*100)
print(headgroup_occupancy_10e8_ppm.count(1)/len(headgroup_occupancy_10e8_ppm_rep)*100)
print('\n')
print(headgroup_occupancy_10e8_p15.count(0)/len(headgroup_occupancy_10e8_ppm_rep)*100)
print(headgroup_occupancy_10e8_p15.count(1)/len(headgroup_occupancy_10e8_ppm_rep)*100)
print('\n')
print(headgroup_occupancy_10e8_n15.count(0)/len(headgroup_occupancy_10e8_ppm_rep)*100)
print(headgroup_occupancy_10e8_n15.count(1)/len(headgroup_occupancy_10e8_ppm_rep)*100)
print('\n')
print(headgroup_occupancy_10e8_ppm_rep.count(0)/len(headgroup_occupancy_10e8_ppm_rep)*100)
print(headgroup_occupancy_10e8_ppm_rep.count(1)/len(headgroup_occupancy_10e8_ppm_rep)*100)


# In[ ]:





# In[103]:


nitro_RMSD_10e8_ppm=[]
with open(wd+'10e8_ppm_nitro_xyz_RMSD.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        nitro_RMSD_10e8_ppm.append(float(i[0]))
nitro_occupancy_10e8_ppm = [] 
with open(wd+'10e8_ppm_headgroup_occupancy.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        nitro_occupancy_10e8_ppm.append(float(i[0]))
        
nitro_RMSD_10e8_p15=[]
with open(wd+'10e8_p15_nitro_xyz_RMSD.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        nitro_RMSD_10e8_p15.append(float(i[0]))
nitro_occupancy_10e8_p15 = [] 
with open(wd+'10e8_p15_headgroup_occupancy.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        nitro_occupancy_10e8_p15.append(float(i[0]))
nitro_RMSD_10e8_n15=[]
with open(wd+'10e8_n15_nitro_xyz_RMSD.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        nitro_RMSD_10e8_n15.append(float(i[0]))
nitro_occupancy_10e8_n15 = [] 
with open(wd+'10e8_n15_headgroup_occupancy.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        nitro_occupancy_10e8_n15.append(float(i[0]))
nitro_RMSD_10e8_ppm_rep=[]
with open(wd+'10e8_ppm_rep_nitro_xyz_RMSD.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        nitro_RMSD_10e8_ppm_rep.append(float(i[0]))
nitro_occupancy_10e8_ppm_rep = [] 
with open(wd+'10e8_ppm_rep_headgroup_occupancy.csv', newline='') as f:
    reader = csv.reader(f)
    for i in reader:
        nitro_occupancy_10e8_ppm_rep.append(float(i[0]))
        


# In[ ]:





# In[104]:


bound_nitro_RMSD2Xray =[]
for i in range(len(nitro_RMSD_10e8_ppm)):
    if headgroup_occupancy_10e8_ppm[i]==1:
        bound_nitro_RMSD2Xray.append(nitro_RMSD_10e8_ppm[i])
for i in range(len(nitro_RMSD_10e8_p15)):
    if headgroup_occupancy_10e8_p15[i]==1:
        bound_nitro_RMSD2Xray.append(nitro_RMSD_10e8_p15[i])
for i in range(len(nitro_RMSD_10e8_n15)):
    if headgroup_occupancy_10e8_n15[i]==1:
        bound_nitro_RMSD2Xray.append(nitro_RMSD_10e8_n15[i])
for i in range(len(nitro_RMSD_10e8_ppm_rep)):
    if headgroup_occupancy_10e8_ppm_rep[i]==1:
        bound_nitro_RMSD2Xray.append(nitro_RMSD_10e8_ppm_rep[i])


# In[106]:




fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(3, 5))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.3, hspace=0.5)
violin_parts = axs.violinplot(bound_nitro_RMSD2Xray , widths=1, showextrema=True, showmedians=True)
axs.set_title('', fontsize=16)
#axs.set_ylabel(r'RMSD ($\rm \AA$)', fontsize=44)
#axs.set_xlabel(, fontsize=20)
axs.set_xticks([1])
axs.set_xticklabels([])
#axs.set_xticklabels([r"${\rm PO_4}$ Bound"], fontsize=44)
axs.tick_params(axis='y', labelsize= 40)
#axs.set_ylim([0, 10])
for partname in ('cbars','cmins','cmaxes','cmedians'):
    vp = violin_parts[partname]
    vp.set_edgecolor('black')
    vp.set_linewidth(1)
#colors = ['#049DBF','#C682D9',  '#F24130', '#72A603', '#F2BD1D', '#F27405', "#4321B0", "#DB749E"]

violin_parts['bodies'][0].set_facecolor('#53C767')
#violin_parts['bodies'][1].set_facecolor('#696969')
for vp in violin_parts['bodies']:
    #vp.set_facecolor('grey')
    vp.set_edgecolor('black')
    vp.set_linewidth(1)
    vp.set_alpha(1)
    
plt.savefig("10e8_bound_nitro_RMSD2Xray.png", transparent=True, bbox_inches="tight")


# In[109]:


print(np.mean(bound_nitro_RMSD2Xray))
print(np.std(bound_nitro_RMSD2Xray))


# In[133]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_ppm/'
pdb_fp = wd+'final_analysis_input.pdb'
dcd_fp = wd+'final_analysis_traj.dcd'

input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()


#l_groove_frags = crys_5t85.select(  "resnum 66 67 68 69 70 26 27 28 29 30 31  and chain L")
cdrl1_frag_xray = crys_5t85.select(  "resnum 26 27 28 29 30 31  and chain L")
cdrl1_frag_xray_bb = cdrl1_frag_xray.select('name CA CB N')
#full res loop 4 is acutally the combination of all loop fragments invovlved in 108e binding site, kept variable name the same for ease of writing code 
l_groove_xray_loop_ca = l_groove_frags.select("name CA")

loop_aln_loop_rmsd_running_ppm=[]
fr_aln_loop_rmsd_running_ppm=[]
for i, frame in enumerate(dcd):
    #select protein 
    frame_fab = frame.getAtoms().select("protein not resname TIP3")
    #select loop & loop+flanking residues for each frame 
    #manually confirmed same # of atoms in crystal selections - required for later superposition 
    full_res_frame_loop_ext = frame_fab.select(  "resnum 257 258 259 260 261 262" )
    #use backbone atoms for alignemnt & RMSD calculations 
    bb_frame_loop_ext = full_res_frame_loop_ext.select("name CA CB N ")
    superpose(bb_frame_loop_ext, cdrl1_frag_xray_bb) 
    loop_RMSD = calcRMSD(bb_frame_loop_ext, cdrl1_frag_xray_bb)
    loop_aln_loop_rmsd_running_ppm.append(loop_RMSD)

    
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_p15/'
pdb_fp = wd+'final_analysis_input.pdb'
dcd_fp = wd+'final_analysis_traj.dcd'

input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

loop_aln_loop_rmsd_running_p15=[]
fr_aln_loop_rmsd_running_p15=[]
for i, frame in enumerate(dcd):
    #select protein 
    frame_fab = frame.getAtoms().select("protein not resname TIP3")

    #select loop & loop+flanking residues for each frame 
    #manually confirmed same # of atoms in crystal selections - required for later superposition 
    full_res_frame_loop_ext = frame_fab.select(  "resnum 257 258 259 260 261 262" )
    #use backbone atoms for alignemnt & RMSD calculations 
    bb_frame_loop_ext = full_res_frame_loop_ext.select("name CA CB N ")
    superpose(bb_frame_loop_ext, cdrl1_frag_xray_bb) 
    loop_RMSD = calcRMSD(bb_frame_loop_ext, cdrl1_frag_xray_bb)
    loop_aln_loop_rmsd_running_p15.append(loop_RMSD)

    
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_n15/'
pdb_fp = wd+'final_analysis_input.pdb'
dcd_fp = wd+'final_analysis_traj.dcd'

input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

loop_aln_loop_rmsd_running_n15=[]
fr_aln_loop_rmsd_running_n15=[]
for i, frame in enumerate(dcd):
    #select protein 
    frame_fab = frame.getAtoms().select("protein not resname TIP3")

    #select loop & loop+flanking residues for each frame 
    #manually confirmed same # of atoms in crystal selections - required for later superposition 
    full_res_frame_loop_ext = frame_fab.select(  "resnum 257 258 259 260 261 262" )
    #use backbone atoms for alignemnt & RMSD calculations 
    bb_frame_loop_ext = full_res_frame_loop_ext.select("name CA CB N ")
    superpose(bb_frame_loop_ext, cdrl1_frag_xray_bb) 
    loop_RMSD = calcRMSD(bb_frame_loop_ext, cdrl1_frag_xray_bb)
    loop_aln_loop_rmsd_running_n15.append(loop_RMSD)

wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_ppm_rep/'
pdb_fp = wd+'final_analysis_input.pdb'
dcd_fp = wd+'final_analysis_traj.dcd'

input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

loop_aln_loop_rmsd_running_ppm_rep=[]
fr_aln_loop_rmsd_running_ppm_rep=[]
for i, frame in enumerate(dcd):
    #select protein 
    frame_fab = frame.getAtoms().select("protein not resname TIP3")

    #select loop & loop+flanking residues for each frame 
    #manually confirmed same # of atoms in crystal selections - required for later superposition 
    full_res_frame_loop_ext = frame_fab.select(  "resnum 257 258 259 260 261 262" )
    #use backbone atoms for alignemnt & RMSD calculations 
    bb_frame_loop_ext = full_res_frame_loop_ext.select("name CA CB N ")
    superpose(bb_frame_loop_ext, cdrl1_frag_xray_bb) 
    loop_RMSD = calcRMSD(bb_frame_loop_ext, cdrl1_frag_xray_bb)
    loop_aln_loop_rmsd_running_ppm_rep.append(loop_RMSD)


# In[134]:


#go thorugh replicate by replicate & plot - then aggregate time 

#lists : loop bb RMSD & headgroup occupancy 
bound_loop_rmsd = []
unbound_loop_rmsd = [] 
for i in range(len(loop_aln_loop_rmsd_running_ppm_rep)):
    #bound
    if headgroup_occupancy_10e8_ppm[i] == 1: 
        bound_loop_rmsd.append(loop_aln_loop_rmsd_running_ppm[i])
    #unbound
    else: 
        unbound_loop_rmsd.append(loop_aln_loop_rmsd_running_ppm[i])

for i in range(len(loop_aln_loop_rmsd_running_ppm_rep)):
    #bound
    if headgroup_occupancy_10e8_p15[i] == 1: 
        bound_loop_rmsd.append(loop_aln_loop_rmsd_running_p15[i])
    #unbound
    else: 
        unbound_loop_rmsd.append(loop_aln_loop_rmsd_running_p15[i])
        
for i in range(len(loop_aln_loop_rmsd_running_ppm_rep)):
    #bound
    if headgroup_occupancy_10e8_n15[i] == 1: 
        bound_loop_rmsd.append(loop_aln_loop_rmsd_running_n15[i])
    #unbound
    else: 
        unbound_loop_rmsd.append(loop_aln_loop_rmsd_running_n15[i])
for i in range(len(loop_aln_loop_rmsd_running_ppm_rep)):
    #bound
    if headgroup_occupancy_10e8_ppm_rep[i] == 1: 
        bound_loop_rmsd.append(loop_aln_loop_rmsd_running_ppm_rep[i])
    #unbound
    else: 
        unbound_loop_rmsd.append(loop_aln_loop_rmsd_running_ppm_rep[i])


# In[135]:


fr_bb_RMSD_bound_unbound = [bound_loop_rmsd, unbound_loop_rmsd ]

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.3, hspace=0.5)
violin_parts = axs.violinplot(fr_bb_RMSD_bound_unbound , widths=0.4, showextrema=True, showmedians=True)
axs.set_title('', fontsize=16)
axs.set_ylabel(r'RMSD ($\rm \AA$)', fontsize=44)
#axs.set_xlabel(, fontsize=20)
axs.set_xticks([1, 2])
#axs.set_xticklabels([r"${\rm PO_4}$ Bound", r"${\rm PO_4}$ Unbound"], fontsize=44)
axs.set_xticklabels([])
axs.tick_params(axis='y', labelsize= 40)
axs.set_yticks([0, 1, 2])
#axs.set_xticklabels(, .5, 1.0, 1.5])
for partname in ('cbars','cmins','cmaxes','cmedians'):
    vp = violin_parts[partname]
    vp.set_edgecolor('black')
    vp.set_linewidth(1)
#colors = ['#049DBF','#C682D9',  '#F24130', '#72A603', '#F2BD1D', '#F27405', "#4321B0", "#DB749E"]

violin_parts['bodies'][0].set_facecolor('#53C767')
violin_parts['bodies'][1].set_facecolor('#696969')


for vp in violin_parts['bodies']:
    #vp.set_facecolor('grey')
    vp.set_edgecolor('black')
    vp.set_linewidth(1)
    vp.set_alpha(1)
    
plt.savefig("10e8_loop_RMSD_MD2Xray.png", transparent=True, bbox_inches="tight")


# In[128]:



wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_ppm/'
pdb_fp = wd+'final_analysis_input.pdb'
dcd_fp = wd+'final_analysis_traj.dcd'

input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

ref_pdb  = parsePDB(pdb_fp) 
cdrl1_frag_ref = ref_pdb.select(  "resnum 257 258 259 260 261 262")
cdrl1_frag_ref_bb = cdrl1_frag_ref.select('name CA CB N')

#full res loop 4 is acutally the combination of all loop fragments invovlved in 108e binding site, kept variable name the same for ease of writing code 

loop_aln_loop_rmsf_running_ppm=[]
for i, frame in enumerate(dcd):
    #select protein 
    frame_fab = frame.getAtoms().select("protein not resname TIP3")
    #select loop & loop+flanking residues for each frame 
    #manually confirmed same # of atoms in crystal selections - required for later superposition 
    full_res_frame_loop_ext = frame_fab.select(  "resnum 257 258 259 260 261 262" )
    #use backbone atoms for alignemnt & rmsf calculations 
    bb_frame_loop_ext = full_res_frame_loop_ext.select("name CA CB N ")
    superpose(bb_frame_loop_ext, cdrl1_frag_ref_bb) 
    loop_rmsf = calcRMSD(bb_frame_loop_ext, cdrl1_frag_ref_bb)
    loop_aln_loop_rmsf_running_ppm.append(loop_rmsf)

    
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_p15/'
pdb_fp = wd+'final_analysis_input.pdb'
dcd_fp = wd+'final_analysis_traj.dcd'

input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

loop_aln_loop_rmsf_running_p15=[]
fr_aln_loop_rmsf_running_p15=[]
for i, frame in enumerate(dcd):
    #select protein 
    frame_fab = frame.getAtoms().select("protein not resname TIP3")

    #select loop & loop+flanking residues for each frame 
    #manually confirmed same # of atoms in crystal selections - required for later superposition 
    full_res_frame_loop_ext = frame_fab.select(  "resnum 257 258 259 260 261 262" )
    #use backbone atoms for alignemnt & rmsf calculations 
    bb_frame_loop_ext = full_res_frame_loop_ext.select("name CA CB N ")
    superpose(bb_frame_loop_ext, cdrl1_frag_ref_bb) 
    loop_rmsf = calcRMSD(bb_frame_loop_ext, cdrl1_frag_ref_bb)
    loop_aln_loop_rmsf_running_p15.append(loop_rmsf)

    
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_n15/'
pdb_fp = wd+'final_analysis_input.pdb'
dcd_fp = wd+'final_analysis_traj.dcd'

input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

loop_aln_loop_rmsf_running_n15=[]
fr_aln_loop_rmsf_running_n15=[]
for i, frame in enumerate(dcd):
    #select protein 
    frame_fab = frame.getAtoms().select("protein not resname TIP3")

    #select loop & loop+flanking residues for each frame 
    #manually confirmed same # of atoms in crystal selections - required for later superposition 
    full_res_frame_loop_ext = frame_fab.select(  "resnum 257 258 259 260 261 262" )
    #use backbone atoms for alignemnt & rmsf calculations 
    bb_frame_loop_ext = full_res_frame_loop_ext.select("name CA CB N ")
    superpose(bb_frame_loop_ext, cdrl1_frag_ref_bb) 
    loop_rmsf = calcRMSD(bb_frame_loop_ext, cdrl1_frag_ref_bb)
    loop_aln_loop_rmsf_running_n15.append(loop_rmsf)

wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_ppm_rep/'
pdb_fp = wd+'final_analysis_input.pdb'
dcd_fp = wd+'final_analysis_traj.dcd'

input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

loop_aln_loop_rmsf_running_ppm_rep=[]
fr_aln_loop_rmsf_running_ppm_rep=[]
for i, frame in enumerate(dcd):
    #select protein 
    frame_fab = frame.getAtoms().select("protein not resname TIP3")

    #select loop & loop+flanking residues for each frame 
    #manually confirmed same # of atoms in crystal selections - required for later superposition 
    full_res_frame_loop_ext = frame_fab.select(  "resnum 257 258 259 260 261 262" )
    #use backbone atoms for alignemnt & rmsf calculations 
    bb_frame_loop_ext = full_res_frame_loop_ext.select("name CA CB N ")
    superpose(bb_frame_loop_ext, cdrl1_frag_ref_bb) 
    loop_rmsf = calcRMSD(bb_frame_loop_ext, cdrl1_frag_ref_bb)
    loop_aln_loop_rmsf_running_ppm_rep.append(loop_rmsf)


# In[131]:


#go thorugh replicate by replicate & plot - then aggregate time 

#lists : loop bb rmsf & headgroup occupancy 
bound_loop_rmsf = []
unbound_loop_rmsf = [] 
for i in range(len(loop_aln_loop_rmsf_running_ppm_rep)):
    #bound
    if headgroup_occupancy_10e8_ppm[i] == 1: 
        bound_loop_rmsf.append(loop_aln_loop_rmsf_running_ppm[i])
    #unbound
    else: 
        unbound_loop_rmsf.append(loop_aln_loop_rmsf_running_ppm[i])

for i in range(len(loop_aln_loop_rmsf_running_n15)):
    #bound
    if headgroup_occupancy_10e8_p15[i] == 1: 
        bound_loop_rmsf.append(loop_aln_loop_rmsf_running_p15[i])
    #unbound
    else: 
        unbound_loop_rmsf.append(loop_aln_loop_rmsf_running_p15[i])
        
for i in range(len(loop_aln_loop_rmsf_running_n15)):
    #bound
    if headgroup_occupancy_10e8_n15[i] == 1: 
        bound_loop_rmsf.append(loop_aln_loop_rmsf_running_n15[i])
    #unbound
    else: 
        unbound_loop_rmsf.append(loop_aln_loop_rmsf_running_n15[i])
for i in range(len(loop_aln_loop_rmsf_running_n15)):
    #bound
    if headgroup_occupancy_10e8_ppm_rep[i] == 1: 
        bound_loop_rmsf.append(loop_aln_loop_rmsf_running_ppm_rep[i])
    #unbound
    else: 
        unbound_loop_rmsf.append(loop_aln_loop_rmsf_running_ppm_rep[i])


# In[132]:


fr_bb_RMSF_bound_unbound = [bound_loop_rmsf, unbound_loop_rmsf ]

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.3, hspace=0.5)
violin_parts = axs.violinplot(fr_bb_RMSF_bound_unbound , widths=0.4, showextrema=True, showmedians=True)
axs.set_title('', fontsize=16)
axs.set_ylabel(r'RMSF ($\rm \AA$)', fontsize=44)
#axs.set_xlabel(, fontsize=20)
axs.set_xticks([1, 2])
#axs.set_xticklabels([r"${\rm PO_4}$ Bound", r"${\rm PO_4}$ Unbound"], fontsize=44)
axs.set_xticklabels([])
axs.tick_params(axis='y', labelsize= 40)
axs.set_yticks([0, 1, 2])
#axs.set_xticklabels(, .5, 1.0, 1.5])
for partname in ('cbars','cmins','cmaxes','cmedians'):
    vp = violin_parts[partname]
    vp.set_edgecolor('black')
    vp.set_linewidth(1)
#colors = ['#049DBF','#C682D9',  '#F24130', '#72A603', '#F2BD1D', '#F27405', "#4321B0", "#DB749E"]

violin_parts['bodies'][0].set_facecolor('#53C767')
violin_parts['bodies'][1].set_facecolor('#696969')


for vp in violin_parts['bodies']:
    #vp.set_facecolor('grey')
    vp.set_edgecolor('black')
    vp.set_linewidth(1)
    vp.set_alpha(1)
    
plt.savefig("10e8_loop_RMSF_MD2REF.png", transparent=True, bbox_inches="tight")


# In[138]:


print(min(unbound_loop_rmsf))
print(min(unbound_loop_rmsd))

print(max(bound_loop_rmsf))
print(max(bound_loop_rmsd))


# In[12]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/snapshot_pdbs_bound/'
ref_pdb_fp = wd+'10e8_ppm_2500_A.pdb'
ref_pdb  = parsePDB(ref_pdb_fp) 
cdrl1_frag_ref = ref_pdb.select(  "resnum 257 258 259 260 261 262")
cdrl1_frag_ref_bb = cdrl1_frag_ref.select('name CA CB N')
#print(ref_pdb.select('resnum 628 name P').getResnames())
ref_pdb_phos_xyz = ref_pdb.select('resnum 628 name P').getCoords()[0]
#print(ref_pdb_phos_xyz)

wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_ppm/'
pdb_fp = wd+'final_analysis_input.pdb'
dcd_fp = wd+'final_analysis_traj.dcd'

input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

#full res loop 4 is acutally the combination of all loop fragments invovlved in 108e binding site, kept variable name the same for ease of writing code 

loop_aln_PO4_rmsf_running_ppm=[]
for i, frame in enumerate(dcd):
    #select protein 
    frame_fab = frame.getAtoms().select("protein not resname TIP3")
    #select loop & loop+flanking residues for each frame 
    #manually confirmed same # of atoms in crystal selections - required for later superposition 
    full_res_frame_loop_ext = frame_fab.select(  "resnum 257 258 259 260 261 262" )
    #use backbone atoms for alignemnt & rmsf calculations 
    bb_frame_loop_ext = full_res_frame_loop_ext.select("name CA CB N ")
    superpose(bb_frame_loop_ext, cdrl1_frag_ref_bb) 
    frame_po4 = frame.getAtoms().select("index 26855").getCoords()[0]
    #print(frame_po4)
    po4_rmsf = calcRMS(np.linalg.norm(frame_po4-ref_pdb_phos_xyz))
    loop_aln_PO4_rmsf_running_ppm.append(po4_rmsf)

    
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_p15/'
pdb_fp = wd+'final_analysis_input.pdb'
dcd_fp = wd+'final_analysis_traj.dcd'

input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

loop_aln_PO4_rmsf_running_p15=[]
for i, frame in enumerate(dcd):
    #select protein 
    frame_fab = frame.getAtoms().select("protein not resname TIP3")
    #select loop & loop+flanking residues for each frame 
    #manually confirmed same # of atoms in crystal selections - required for later superposition 
    full_res_frame_loop_ext = frame_fab.select(  "resnum 257 258 259 260 261 262" )
    #use backbone atoms for alignemnt & rmsf calculations 
    bb_frame_loop_ext = full_res_frame_loop_ext.select("name CA CB N ")
    superpose(bb_frame_loop_ext, cdrl1_frag_ref_bb) 
    frame_po4 = frame.getAtoms().select("index 24895").getCoords()[0]
    po4_rmsf = calcRMS(np.linalg.norm(frame_po4-ref_pdb_phos_xyz))
    loop_aln_PO4_rmsf_running_p15.append(po4_rmsf)

    
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_n15/'
pdb_fp = wd+'final_analysis_input.pdb'
dcd_fp = wd+'final_analysis_traj.dcd'

input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

loop_aln_PO4_rmsf_running_n15=[]
for i, frame in enumerate(dcd):
    #select protein 
    frame_fab = frame.getAtoms().select("protein not resname TIP3")
    #select loop & loop+flanking residues for each frame 
    #manually confirmed same # of atoms in crystal selections - required for later superposition 
    full_res_frame_loop_ext = frame_fab.select(  "resnum 257 258 259 260 261 262" )
    #use backbone atoms for alignemnt & rmsf calculations 
    bb_frame_loop_ext = full_res_frame_loop_ext.select("name CA CB N ")
    superpose(bb_frame_loop_ext, cdrl1_frag_ref_bb) 
    frame_po4 = frame.getAtoms().select("index 28787").getCoords()[0]
    po4_rmsf = calcRMS(np.linalg.norm(frame_po4-ref_pdb_phos_xyz))
    loop_aln_PO4_rmsf_running_n15.append(po4_rmsf)
    
    
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_ppm_rep/'
pdb_fp = wd+'final_analysis_input.pdb'
dcd_fp = wd+'final_analysis_traj.dcd'

input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

loop_aln_PO4_rmsf_running_ppm_rep=[]
for i, frame in enumerate(dcd):
    #select protein 
    frame_fab = frame.getAtoms().select("protein not resname TIP3")
    #select loop & loop+flanking residues for each frame 
    #manually confirmed same # of atoms in crystal selections - required for later superposition 
    full_res_frame_loop_ext = frame_fab.select(  "resnum 257 258 259 260 261 262" )
    #use backbone atoms for alignemnt & rmsf calculations 
    bb_frame_loop_ext = full_res_frame_loop_ext.select("name CA CB N ")
    superpose(bb_frame_loop_ext, cdrl1_frag_ref_bb) 
    frame_po4 = frame.getAtoms().select("index 24253").getCoords()[0]
    po4_rmsf = calcRMS(np.linalg.norm(frame_po4-ref_pdb_phos_xyz))
    loop_aln_PO4_rmsf_running_ppm_rep.append(po4_rmsf)


# In[22]:


bound_po4_RMSF=[]
unbound_po4_RMSF=[]
for i in range(len(loop_aln_PO4_rmsf_running_ppm)):
    if headgroup_occupancy_10e8_ppm[i]==1:
        bound_po4_RMSF.append(loop_aln_PO4_rmsf_running_ppm[i])
    else:
        unbound_po4_RMSF.append(loop_aln_PO4_rmsf_running_ppm[i])
for i in range(len(loop_aln_PO4_rmsf_running_p15)):
    if headgroup_occupancy_10e8_p15[i]==1:
        bound_po4_RMSF.append(loop_aln_PO4_rmsf_running_p15[i])
    else:
        unbound_po4_RMSF.append(loop_aln_PO4_rmsf_running_p15[i])
for i in range(len(loop_aln_PO4_rmsf_running_n15)):
    if headgroup_occupancy_10e8_n15[i]==1:
        bound_po4_RMSF.append(loop_aln_PO4_rmsf_running_n15[i])
    else:
        unbound_po4_RMSF.append(loop_aln_PO4_rmsf_running_n15[i])
for i in range(len(loop_aln_PO4_rmsf_running_ppm_rep)):
    if headgroup_occupancy_10e8_ppm_rep[i]==1:
        bound_po4_RMSF.append(loop_aln_PO4_rmsf_running_ppm_rep[i])
    else:
        unbound_po4_RMSF.append(loop_aln_PO4_rmsf_running_ppm_rep[i])


# In[21]:




fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(3, 5))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.3, hspace=0.5)
violin_parts = axs.violinplot(bound_po4_RMSF , widths=1, showextrema=True, showmedians=True)
axs.set_title('', fontsize=16)
#axs.set_ylabel(r'RMSD ($\rm \AA$)', fontsize=44)
#axs.set_xlabel(, fontsize=20)
axs.set_xticks([1])
axs.set_xticklabels([])
#axs.set_xticklabels([r"${\rm PO_4}$ Bound"], fontsize=44)
axs.tick_params(axis='y', labelsize= 40)
#axs.set_ylim([0, 10])
for partname in ('cbars','cmins','cmaxes','cmedians'):
    vp = violin_parts[partname]
    vp.set_edgecolor('black')
    vp.set_linewidth(1)
#colors = ['#049DBF','#C682D9',  '#F24130', '#72A603', '#F2BD1D', '#F27405', "#4321B0", "#DB749E"]

violin_parts['bodies'][0].set_facecolor('#53C767')
#violin_parts['bodies'][1].set_facecolor('#696969')
for vp in violin_parts['bodies']:
    #vp.set_facecolor('grey')
    vp.set_edgecolor('black')
    vp.set_linewidth(1)
    vp.set_alpha(1)
    
plt.savefig("10e8_bound_PO4_RMSDF.png", transparent=True, bbox_inches="tight")


# In[23]:




fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(3, 5))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.3, hspace=0.5)
violin_parts = axs.violinplot(unbound_po4_RMSF , widths=1, showextrema=True, showmedians=True)
axs.set_title('', fontsize=16)
#axs.set_ylabel(r'RMSD ($\rm \AA$)', fontsize=44)
#axs.set_xlabel(, fontsize=20)
axs.set_xticks([1])
axs.set_xticklabels([])
#axs.set_xticklabels([r"${\rm PO_4}$ Bound"], fontsize=44)
axs.tick_params(axis='y', labelsize= 40)
#axs.set_ylim([0, 10])
for partname in ('cbars','cmins','cmaxes','cmedians'):
    vp = violin_parts[partname]
    vp.set_edgecolor('black')
    vp.set_linewidth(1)
#colors = ['#049DBF','#C682D9',  '#F24130', '#72A603', '#F2BD1D', '#F27405', "#4321B0", "#DB749E"]

violin_parts['bodies'][0].set_facecolor('#53C767')
#violin_parts['bodies'][1].set_facecolor('#696969')
for vp in violin_parts['bodies']:
    #vp.set_facecolor('grey')
    vp.set_edgecolor('black')
    vp.set_linewidth(1)
    vp.set_alpha(1)
    
plt.savefig("10e8_unbound_PO4_RMSDF.png", transparent=True, bbox_inches="tight")


# In[26]:


print(np.mean(bound_po4_RMSF))
print(np.std(bound_po4_RMSF))

