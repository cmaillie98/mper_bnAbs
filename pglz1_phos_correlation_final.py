#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


# In[5]:


plt.rcParams['font.sans-serif'] = "Arial"
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


# In[6]:


#caclulate distances of loop resdiues to phosphate for crystal strucutre 
crys_4e10 = parsePDB("/Users/cmaillie/Dropbox (Scripps Research)/manuscript/pdbs/4xc1.pdb")
crys_phos = crys_4e10.select( "resid 301 name P" )
crys_phos_xyz = crys_phos.getCoords() 
print(crys_phos_xyz)

#ser28
ser28_crys= crys_4e10.select( "name CA" )[241]
crys_ser_28_xyz = crys_4e10.select( "name CA" )[241].getCoords()
#print(crys_ser_28_xyz)
#print(crys_4e10.select( "name CA" )[241].getResname())
phos_ser_28_dist = np.linalg.norm(crys_ser_28_xyz-crys_phos_xyz)
rms_phos_ser_28_dist = calcRMS(phos_ser_28_dist)
print(phos_ser_28_dist)
#phe29
phe9_crys = crys_4e10.select( "name CA" )[242]
crys_phe_29_xyz = crys_4e10.select( "name CA" )[242].getCoords()
phos_phe_29_dist = np.linalg.norm(crys_phe_29_xyz-crys_phos_xyz)
rms_phos_phe_29_dist = calcRMS(phos_phe_29_dist)
#print(phos_phe_29_dist)
#print(crys_phe_29_xyz)
#print(crys_4e10.select( "name CA" )[242].getResname())
#ser30
ser30_crys= crys_4e10.select( "name CA" )[243]
crys_ser_30_xyz = crys_4e10.select( "name CA" )[243].getCoords()
phos_ser_30_dist = np.linalg.norm(crys_ser_30_xyz-crys_phos_xyz)
rms_phos_ser_30_dist = calcRMS(phos_ser_30_dist)
#print(phos_ser_30_dist)

#BINDING SITE = CENTER OF TWO SERINE RESIDUES 
binding_site = calcCenter(ser30_crys+ser28_crys)
#print(binding_site)
phos2binding_site_dist= np.linalg.norm(crys_phos_xyz-binding_site)
#print(phos2binding_site_dist)


ser28_OH = crys_4e10.select( "resnum 28 and chain H and name OG")
phos2OH1_dist= np.linalg.norm(crys_phos_xyz-ser28_OH.getCoords())
#print(phos2OH1_dist)

ser30_OH = crys_4e10.select( "resnum 30 and chain H and name OG")
phos2OH2_dist= np.linalg.norm(crys_phos_xyz-ser30_OH.getCoords())
#print(phos2OH2_dist)

full_res_xray_loop_4 = crys_4e10.select(  "resnum 24 25 26 27 28 29 30 31 32 33 34 and chain H" )
bb_xray_loop_4 = full_res_xray_loop_4.select("name CA CB N ")


bb_xray_flank_fr_res = crys_4e10.select(  "resnum 19 20 21 22 23 35 36 37 38 39 and chain H" )
bb_xray_flank_fr = bb_xray_flank_fr_res.select("name CA CB N ")


for i in bb_xray_flank_fr.select("name CA"):
    print(i.getResname())


# In[8]:


def phos_correlation_pgzl1(pdb_fp, dcd_fp, phos_index, prefix):
    #pdb_fp : full file path to input pdb 
    #dcd_fp: full filepath to input dcd 
    #phos idex: int of phos that should be tracked 
    #prefix : str of what to append to file paths 
    #NOTE: crystal structure stats must be available in environment before running this method 
    #have edited this for CG reversions 
    input_pdb = parsePDB(pdb_fp)
    dcd = DCDFile(dcd_fp)
    dcd.setCoords(input_pdb)
    dcd.link(input_pdb)
    dcd.reset()
        
    #initialize lists to store relevant values 
    globals()[prefix+'loop_ext_RMSD']=[] 
    globals()[prefix+'po4_xyz_RMSD']=[] 
    
    globals()[prefix+'po4_OH1_dist']=[] 
    globals()[prefix+'po4_OH2_dist']=[] 
    globals()[prefix+'po4_N1_dist']=[] 
    globals()[prefix+'po4_N2_dist']=[]
    
    globals()[prefix+'tracked_po4_site_dist']=[] 
    globals()[prefix+'closest_po4_site_dist']=[] 
 
    globals()[prefix+'po4_occupancy']=[] 
    
    #initalizat list names for writing files 
    loop_ext_RMSD = prefix+'_loop_ext_RMSD'
    po4_xyz_RMSD =prefix+'_po4_xyz_RMSD'
    po4_OH1_dist =prefix+'_po4_OH1_dist'  
    po4_OH2_dist =prefix+'_po4_OH2_dist' 
    po4_N1_dist =prefix+'_po4_N1_dist'
    po4_N2_dist =prefix+'_po4_N2_dist'
    tracked_po4_site_dist = prefix+'_tracked_po4_site_dist' 
    closest_po4_site_dist = prefix+'_closest_po4_site_dist'
    po4_occupancy = prefix+'_po4_occupancy'
    
    loop_ext_RMSD_avg = prefix+'_loop_ext_RMSD_avg'
    po4_xyz_RMSD_avg=prefix+'_po4_xyz_RMSD_avg'
    po4_OH1_dist_avg =prefix+'_po4_OH1_dist_avg'  
    po4_OH2_dist_avg=prefix+'_po4_OH2_dist_avg' 
    po4_N1_dist_avg=prefix+'_po4_N1_dist_avg'
    po4_N2_dist_avg=prefix+'_po4_N2_dist_avg'
    
    loop_ext_RMSD_stdev = prefix+'_loop_ext_RMSD_stdev'
    po4_xyz_RMSD_stdev =prefix+'_po4_xyz_RMSD_stdev'
    po4_OH1_dist_stdev =prefix+'_po4_OH1_dist_stdev'  
    po4_OH2_dist_stdev =prefix+'_po4_OH2_dist_stdev' 
    po4_N1_dist_stdev =prefix+'_po4_N1_dist_stdev'
    po4_N2_dist_stdev =prefix+'_po4_N2_dist_stdev'
    
    for i, frame in enumerate(dcd):

        #select protein 
        frame_fab = frame.getAtoms().select("protein not resname TIP3")

        #select loop & loop+flanking residues for each frame 
        #manually confirmed same # of atoms in crystal selections - required for later superposition 
         
        #full_res_frame_loop_ext = frame_fab.select(  "resnum 24 25 26 27 28 29 30 31 32 33 34" )
        full_res_frame_loop_ext = frame_fab.select(  "resnum 238 239 240 241 242 243 244 245 246 247 248" )
        #238 239 240 241 242 243 244 245 246 247 248
        #use backbone atoms for alignemnt & RMSD calculations 
        bb_frame_loop_ext = full_res_frame_loop_ext.select("name CA CB N ")
        #for i in bb_frame_loop_ext.select("name CA"):
        #    print(i.getResname())
        #select phos of interest - look in VMD final frame index of phos closest to loop 
        index_selection_str = "index "+ str(phos_index)
        frame_phos = frame.getAtoms().select(index_selection_str  ) #"index 26328" 


        #superpose loop extended (no phos) - this will move whole system based on this superpose 
        #this is done so that xyz of crystal to md frame will be relevant in space 
        superpose(bb_frame_loop_ext, bb_xray_loop_4)


        #calc RMSD of extended loop   
        loop_ext_rmsd = calcRMSD(bb_frame_loop_ext, bb_xray_loop_4)
        globals()[prefix+'loop_ext_RMSD'].append(loop_ext_rmsd)

        po4_RMSD = calcRMS(frame_phos.getCoords()[0]- crys_phos_xyz[0])
        globals()[prefix+'po4_xyz_RMSD'].append(po4_RMSD) 

        #calc closest PO4 molecule to binding site 
        
        

#         loop_edges = frame_fab.select( "resnum 28 30 and name CA")                 
#         frame_ser_OH1 = frame_fab.select( "resnum 28 and name OG1" )
#         frame_ser_OH2 = frame_fab.select( "resnum 30 and name OG" )
#         frame_nitrogen_1 = frame_fab.select("resnum 29 and name N")
#         frame_nitrogen_2 = frame_fab.select("resnum 30 and name N")

        loop_edges = frame_fab.select( "resnum 242 244 and name CA")                 
        frame_ser_OH1 = frame_fab.select( "resnum 242 and name OG1" )
        frame_ser_OH2 = frame_fab.select( "resnum 244 and name OG" )
        frame_nitrogen_1 = frame_fab.select("resnum 243 and name N")
        frame_nitrogen_2 = frame_fab.select("resnum 244 and name N")
        binding_site = calcCenter(loop_edges)
        po4_2_binding_site_dist = np.linalg.norm(frame_phos.getCoords()-binding_site)
        globals()[prefix+'tracked_po4_site_dist'].append(po4_2_binding_site_dist)


        #get distances of po4 to relevant atoms in loop
        #print(frame_ser_OH1.getCoords())
        po4_2_ser_OH1_dist = np.linalg.norm(frame_phos.getCoords()-frame_ser_OH1.getCoords())
        globals()[prefix+'po4_OH1_dist'].append(po4_2_ser_OH1_dist)
        po4_2_ser_OH2_dist = np.linalg.norm(frame_phos.getCoords()-frame_ser_OH2.getCoords())
        globals()[prefix+'po4_OH2_dist'].append(po4_2_ser_OH2_dist)
        po4_2_nitrogen_1_dist = np.linalg.norm(frame_phos.getCoords()-frame_nitrogen_1.getCoords())
        globals()[prefix+'po4_N1_dist'].append(po4_2_nitrogen_1_dist)
        po4_2_nitrogen_2_dist = np.linalg.norm(frame_phos.getCoords()-frame_nitrogen_2.getCoords())
        globals()[prefix+'po4_N2_dist'].append(po4_2_nitrogen_2_dist)

        #find closest po4 to site via minimum distance of all P to site  
        total_phos = frame.getAtoms().select( "name P").getCoords()
        frame_phos_distances = [] 
        for i in total_phos:
            frame_phos_distances.append(np.linalg.norm(i-binding_site))
        globals()[prefix+'closest_po4_site_dist'].append(min(frame_phos_distances))

        #determine occupancy state 
        #if score is > 2 and RMSD is < 2.0, phosphate site is occupie 
        occupancy_score = 0 

        if po4_2_ser_OH1_dist<=5.25: 
            occupancy_score = occupancy_score+1 
        if po4_2_ser_OH2_dist<=5.25:
            occupancy_score = occupancy_score+1 
        if po4_2_nitrogen_1_dist<=5.25:
            occupancy_score = occupancy_score+1 
        if po4_2_nitrogen_2_dist<=5.25:
            occupancy_score = occupancy_score+1 

        if po4_RMSD<2.0 and occupancy_score >=2 :
            globals()[prefix+'po4_occupancy'].append(1)
        else:
            globals()[prefix+'po4_occupancy'].append(0)


    #calculate 10 step running avg & stdev for each list of values (for ease of plotting later )
    #dont need to do this for po4 occupancy 
    #also skipped tracked_po4_site_dist & closest_po4_site_disto4
    globals()[prefix+'loop_ext_RMSD_avg'] = [] 
    globals()[prefix+'loop_ext_RMSD_stdev'] = [] 
    calcRunningAvg(globals()[prefix+'loop_ext_RMSD'],
                   globals()[prefix+'loop_ext_RMSD_avg'],
                   globals()[prefix+'loop_ext_RMSD_stdev'], 10)
    
    globals()[prefix+'po4_xyz_RMSD_avg'] = [] 
    globals()[prefix+'po4_xyz_RMSD_stdev'] = [] 
    calcRunningAvg(globals()[prefix+'po4_xyz_RMSD'],
                   globals()[prefix+'po4_xyz_RMSD_avg'],
                   globals()[prefix+'po4_xyz_RMSD_stdev'], 10)
    
    globals()[prefix+'po4_OH1_dist_avg'] = [] 
    globals()[prefix+'po4_OH1_dist_stdev'] = [] 
    calcRunningAvg(globals()[prefix+'po4_OH1_dist'],
                   globals()[prefix+'po4_OH1_dist_avg'],
                   globals()[prefix+'po4_OH1_dist_stdev'], 10)   

    globals()[prefix+'po4_OH2_dist_avg'] = [] 
    globals()[prefix+'po4_OH2_dist_stdev'] = [] 
    calcRunningAvg(globals()[prefix+'po4_OH2_dist'],
                   globals()[prefix+'po4_OH2_dist_avg'],
                   globals()[prefix+'po4_OH2_dist_stdev'], 10)   
    globals()[prefix+'po4_N1_dist_avg'] = [] 
    globals()[prefix+'po4_N1_dist_stdev'] = [] 
    calcRunningAvg(globals()[prefix+'po4_N1_dist'],
                   globals()[prefix+'po4_N1_dist_avg'],
                   globals()[prefix+'po4_N1_dist_stdev'], 10)  
    globals()[prefix+'po4_N2_dist_avg'] = [] 
    globals()[prefix+'po4_N2_dist_stdev'] = [] 
    calcRunningAvg(globals()[prefix+'po4_N2_dist'],
                   globals()[prefix+'po4_N2_dist_avg'],
                   globals()[prefix+'po4_N2_dist_stdev'], 10)  

    #print lists to files for accessing raw data later (i.e. plotting)
    lists_to_print = [globals()[prefix+'loop_ext_RMSD'], 
                      globals()[prefix+'po4_xyz_RMSD'],
                      globals()[prefix+'po4_OH1_dist'],
                      globals()[prefix+'po4_OH2_dist'],
                      globals()[prefix+'po4_N1_dist'],
                      globals()[prefix+'po4_N2_dist'], 
                      globals()[prefix+'tracked_po4_site_dist'], 
                      globals()[prefix+'closest_po4_site_dist'],
                      globals()[prefix+'po4_occupancy']] 
    lists_to_print_names = [loop_ext_RMSD,
                            po4_xyz_RMSD,
                            po4_OH1_dist,
                            po4_OH2_dist,
                            po4_N1_dist,
                            po4_N2_dist,
                            tracked_po4_site_dist,
                            closest_po4_site_dist,
                            po4_occupancy] 

    for i in range(len(lists_to_print)): 
        file_name = lists_to_print_names[i]+".csv"
        np.savetxt(file_name, lists_to_print[i], delimiter=",", fmt='%1.2f')
        
    #print avg & stdev lists to csv files         
    avg_lists_to_print = [globals()[prefix+'loop_ext_RMSD_avg'], 
                      globals()[prefix+'po4_xyz_RMSD_avg'],
                      globals()[prefix+'po4_OH1_dist_avg'],
                      globals()[prefix+'po4_OH2_dist_avg'],
                      globals()[prefix+'po4_N1_dist_avg'],
                      globals()[prefix+'po4_N2_dist_avg']] 
    
    avg_lists_to_print_names = [loop_ext_RMSD_avg,
                                po4_xyz_RMSD_avg,
                                po4_OH1_dist_avg,
                                po4_OH2_dist_avg,
                                po4_N1_dist_avg,
                                po4_N2_dist_avg]
    
    stdev_lists_to_print = [globals()[prefix+'loop_ext_RMSD_stdev'], 
                      globals()[prefix+'po4_xyz_RMSD_stdev'],
                      globals()[prefix+'po4_OH1_dist_stdev'],
                      globals()[prefix+'po4_OH2_dist_stdev'],
                      globals()[prefix+'po4_N1_dist_stdev'],
                      globals()[prefix+'po4_N2_dist_stdev']] 
    
    stdev_lists_to_print_names = [loop_ext_RMSD_stdev,
                                  po4_xyz_RMSD_stdev,
                                  po4_OH1_dist_stdev,
                                  po4_OH2_dist_stdev,
                                  po4_N1_dist_stdev,
                                  po4_N2_dist_stdev ]

    for i in range(len(avg_lists_to_print)): 
        #write avg list 
        file_name = avg_lists_to_print_names[i]+".csv"
        np.savetxt(file_name, avg_lists_to_print[i], delimiter=",", fmt='%1.2f')
        #write stdev list 
        file_name = stdev_lists_to_print_names[i]+".csv"
        np.savetxt(file_name, stdev_lists_to_print[i], delimiter=",", fmt='%1.2f')
    return lists_to_print_names 
      
def plot_phos_correlation(avg_rmsd, stdev_rmsd, po4_occupancy, prefix): 

    avg_rmsd = np.array(avg_rmsd)
    stdev_rmsd = np.array(stdev_rmsd)
    po4_occupancy = np.array(po4_occupancy)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 7), gridspec_kw={'height_ratios': [0.25, 6]})
    plt.subplots_adjust(hspace=0.1)
    avg_rmsd = avg_rmsd[0:2492]
    stdev_rmsd = stdev_rmsd[0:2492]

    #x = np.arange(0, len(avg_po4_xyz_RMSD_4e10_n15)*.0002, .0002)
    x = np.arange(0, len(avg_rmsd),1)

    ax1.fill_between(x, -1, 0, where=np.array(po4_occupancy[0:2492])==1, 
                     alpha=1, color='#53C767') 
    ax1.fill_between(x, -1, 0, where=np.array(po4_occupancy[0:2492])==0, 
                     alpha=1, color='#696969') 

    ax1.axhline(y=-1, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
    ax1.axhline(y=0, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
    ax1.axvline(x=2500, color='#000000', linestyle='-', ymin=0, linewidth=1)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    #ax1.set_ylabel(r"${\rm PO_4}$ Bound", fontsize=40, rotation='horizontal', 
    #               va="center", labelpad=120)
    ax1.set_xlim(-15, 2500)
    ax1.set_ylim(-1, 0)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    ax2.plot(x, avg_rmsd, lw=1, color='black')#00122E
    ax2.fill_between(x, avg_rmsd-stdev_rmsd, 
                     avg_rmsd+stdev_rmsd, 
                     color='black', alpha=0.25)
    ax2.set_xticks([0, 1250, 2500]) 
    ax2.set_xticklabels(['0', '0.25', '0.5' ])
    ax2.tick_params(axis='x', labelsize=50)
    ax2.tick_params(axis='y', labelsize=50)#, fontsize=16
    ax2.set_ylim(0, 2)
    ax2.set_xlim(-15, 2500)
    #ax2.set_xlabel(r"Time ($\rm \mu s$)", fontsize=60)
    #ax2.set_ylabel(r"Loop BB RMSD ($\rm \AA$)", fontsize=60) #, wrap=True / to X-Ray 
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    
    fig_name = prefix + '_po4_rmsd.png' 
    plt.savefig(fig_name, transparent=True, bbox_inches="tight")

    plt.show()
    return "Made figure: ", prefix


# In[9]:



def plot_phos_correlationv2(avg_rmsd, stdev_rmsd, po4_occupancy, prefix): 

    avg_rmsd = np.array(avg_rmsd)
    stdev_rmsd = np.array(stdev_rmsd)
    po4_occupancy = np.array(po4_occupancy)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 7), gridspec_kw={'height_ratios': [0.25, 6]})
    plt.subplots_adjust(hspace=0.1)
    avg_rmsd = avg_rmsd[0:2492]
    stdev_rmsd = stdev_rmsd[0:2492]

    #x = np.arange(0, len(avg_po4_xyz_RMSD_4e10_n15)*.0002, .0002)
    x = np.arange(0, len(avg_rmsd),1)

    ax1.fill_between(x, -1, 0, where=np.array(po4_occupancy[0:2492])==1, 
                     alpha=1, color='#53C767') 
    ax1.fill_between(x, -1, 0, where=np.array(po4_occupancy[0:2492])==0, 
                     alpha=1, color='#696969') 

    ax1.axhline(y=-1, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
    ax1.axhline(y=0, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
    ax1.axvline(x=5023, color='#000000', linestyle='-', ymin=0, linewidth=1)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    #ax1.set_ylabel(r"${\rm PO_4}$ Bound", fontsize=40, rotation='horizontal', 
    #               va="center", labelpad=120)
    ax1.set_xlim(0, 2492)
    ax1.set_ylim(-1, 0)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    ax2.plot(x, avg_rmsd, lw=1, color='black')#00122E
    ax2.fill_between(x, avg_rmsd-stdev_rmsd, 
                     avg_rmsd+stdev_rmsd, 
                     color='black', alpha=0.25)
    ax2.set_xticks([0, 2500, 4990]) # 3030, 4030, 5030
    ax2.set_xticklabels(['0', "0.5", '1.0' ])#'0.6', '0.8', '1.0' 
    ax2.tick_params(axis='x', labelsize=50)
    ax2.tick_params(axis='y', labelsize=50)#, fontsize=16
    ax2.set_ylim(0, 2)
    ax2.set_xlim(0, 4990)
    ax2.set_xlabel(r"Time ($\rm \mu s$)", fontsize=60)
    ax2.set_ylabel(r" RMSD ($\rm \AA$)", fontsize=60) #, wrap=True / to X-Ray 
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    
    fig_name = prefix + '_po4_rmsd.png' 
    plt.savefig(fig_name, transparent=True, bbox_inches="tight")

    plt.show()
    return "Made figure: ", prefix


# In[10]:



def plot_phos_correlationv3(avg_rmsd, stdev_rmsd, po4_occupancy, avg_rmsd_repl, stdev_rmsd_repl, po4_repl, prefix): 

    avg_rmsd = np.array(avg_rmsd)
    stdev_rmsd = np.array(stdev_rmsd)
    po4_occupancy = np.array(po4_occupancy)
    avg_rmsd_repl = np.array(avg_rmsd_repl)
    stdev_rmsd_repl = np.array(stdev_rmsd_repl)
    po4_repl = np.array(po4_repl)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 4), gridspec_kw={'height_ratios': [0.25, 6]})
    plt.subplots_adjust(hspace=0.1)
    avg_rmsd = avg_rmsd[0:4990]
    stdev_rmsd = stdev_rmsd[0:4990]
    avg_rmsd_repl = avg_rmsd_repl[0:4990]
    stdev_rmsd_repl = stdev_rmsd_repl[0:4990]
    #x = np.arange(0, len(avg_po4_xyz_RMSD_4e10_n15)*.0002, .0002)
    x = np.arange(0, len(avg_rmsd),1)
    
    ax1.fill_between(x, -1, 0, where=np.array(po4_occupancy[0:4990])==0, 
                     alpha=1, color='#696969') 
    ax1.fill_between(x, -1, 0, where=np.array(po4_occupancy[0:4990])==1, 
                     alpha=1, color='#53C767') 
    ax1.fill_between(x, -1, 0, where=np.array(po4_repl[0:4990])==1, 
                     alpha=1, color='#53C767')
    

    ax1.axhline(y=-1, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
    ax1.axhline(y=0, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
    ax1.axvline(x=5000, color='#000000', linestyle='-', ymin=0, linewidth=1)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    #ax1.set_ylabel(r"${\rm PO_4}$ Bound", fontsize=40, rotation='horizontal', 
    #               va="center", labelpad=120)
    ax1.set_xlim(0, 5000)
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
    
    ax2.set_xticks([0, 2500, 4990]) # 3030, 4030, 5030
    ax2.set_xticklabels(['0', "0.5", '1.0' ])#'0.6', '0.8', '1.0' 
    ax2.tick_params(axis='x', labelsize=50)
    ax2.tick_params(axis='y', labelsize=50)#, fontsize=16
    ax2.set_ylim(-.5, 2)
    ax2.set_yticks([0, 1, 2])
    ax2.set_xlim(0, 4990)
    ax2.set_xlabel(r"Time ($\rm \mu s$)", fontsize=60)
    ax2.set_ylabel(r" RMSD ($\rm \AA$)", fontsize=60) #, wrap=True / to X-Ray 
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    
    fig_name = prefix + '_po4_rmsd.png' 
    plt.savefig(fig_name, transparent=True, bbox_inches="tight")

    plt.show()
    return "Made figure: ", prefix

