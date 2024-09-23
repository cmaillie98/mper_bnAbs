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



def plot_phos_correlation(avg_rmsd, stdev_rmsd, po4_occupancy, prefix): 

    avg_rmsd = np.array(avg_rmsd)
    stdev_rmsd = np.array(stdev_rmsd)
    po4_occupancy = np.array(po4_occupancy)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 7), gridspec_kw={'height_ratios': [0.25, 6]})
    plt.subplots_adjust(hspace=0.1)
    avg_rmsd = avg_rmsd[0:2500]
    stdev_rmsd = stdev_rmsd[0:2500]

    #x = np.arange(0, len(avg_po4_xyz_RMSD_4e10_n15)*.0002, .0002)
    x = np.arange(0, len(avg_rmsd),1)

    ax1.fill_between(x, -1, 0, where=np.array(po4_occupancy[0:2500])==1, 
                     alpha=1, color='#53C767') 
    ax1.fill_between(x, -1, 0, where=np.array(po4_occupancy[0:2500])==0, 
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
    ax2.set_xticklabels([ '0', '0.25', '0.5' ])
    ax2.tick_params(axis='x', labelsize=50)
    ax2.tick_params(axis='y', labelsize=50)#, fontsize=16
    ax2.set_ylim(-.5, 5)
    ax2.set_xlim(-15, 2500)
    #ax2.set_xlabel(r"Time ($\rm \mu s$)", fontsize=60)
    #ax2.set_ylabel(r"Loop BB RMSD ($\rm \AA$)", fontsize=60) #, wrap=True / to X-Ray 
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    
    fig_name = prefix + '_po4_rmsd.png' 
    plt.savefig(fig_name, transparent=True, bbox_inches="tight")

    plt.show()
    return "Made figure: ", prefix


# In[4]:


#caclulate distances of loop resdiues to phosphate for crystal strucutre 
crys_5t85 = parsePDB("/Users/cmaillie/Dropbox (Scripps Research)/manuscript/pdbs/5t85.pdb")
crys_phos = crys_5t85.select( "resid 301 name P" )
crys_phos_xyz = crys_phos.getCoords() 
#print(crys_phos_xyz)

# for i in range(len(crys_5t85.select( "name CA" ))):
#     print(i, crys_5t85.select( "name CA" )[i].getResname())
print(crys_5t85.select( "name CA" )[296].getResname()) #SER 100cH
print(crys_5t85.select( "name CA" )[297].getResname()) #LEU 28L
print(crys_5t85.select( "name CA" )[298].getResname()) #SER 30L 

print(crys_5t85.select( "name CA" )[258].getResname()) #LEU 28L
print(crys_5t85.select( "name CA" )[260].getResname()) #SER 30L 
print(crys_5t85.select( "name CA" )[261].getResname()) #HIS 31H 


print(crys_5t85.select( "name CA" )[108].getResname()) #SER 100cH



#crys_distances = [phos_ser_28_dist, phos_phe_29_dist, phos_ser_30_dist]
xray_phos_selection =  crys_5t85.select( "resid 301 name P" )
#ull_res_xray_loop = crys_5t85.select( "resnum 28 29 30 and chain A" )
#remove gly for furute clacluations to superimpose 
l_loop_frags = crys_5t85.select(  "resnum  65 66 67 68 69 70 25 26 27 28 29 30 31 32 33 34  and chain L")
#print(l_loop_frags.getResnames())
h_loop_frags = crys_5t85.select(  "resnum 100B 100C 100D and chain H ")
#print(h_loop_frags.getResnames())

#full res loop 4 is acutally the combination of all loop fragments invovlved in 108e binding site, kept variable name the same for ease of writing code 
full_res_xray_loop_4 =  l_loop_frags #+ h_loop_frags #108 259 260 261 262 296 297 298" )
bb_xray_loop_4 = full_res_xray_loop_4.select("name CA ")
print(bb_xray_loop_4.getResnames())



['GLY' 'ASP' 'SER' 'LEU' 'ARG' 'SER' 'HIS' 'TYR' 'ALA' 'SER' 'SER' 'ALA'
 'SER' 'GLY' 'ASN' 'ARG']


# In[23]:


def phos_correlation_10e8(pdb_fp, dcd_fp, phos_index, prefix):
    #pdb_fp : full file path to input pdb 
    #dcd_fp: full filepath to input dcd 
    #phos idex: int of phos that should be tracked 
    #prefix : str of what to append to file paths 
    #NOTE: crystal structure stats must be available in environment before running this method 
    
    input_pdb = parsePDB(pdb_fp)
    dcd = DCDFile(dcd_fp)
    dcd.setCoords(input_pdb)
    dcd.link(input_pdb)
    dcd.reset()
        
    #initialize lists to store relevant values 
    globals()[prefix+'loop_ext_RMSD']=[] 
    globals()[prefix+'po4_xyz_RMSD']=[] 
    
    globals()[prefix+'po4_OH1_dist']=[] 
    globals()[prefix+'po4_N1_dist']=[] 
 
    globals()[prefix+'po4_occupancy']=[] 
    
    #initalizat list names for writing files 
    loop_ext_RMSD = prefix+'_loop_ext_RMSD'
    po4_xyz_RMSD =prefix+'_po4_xyz_RMSD'
    po4_OH1_dist =prefix+'_po4_OH1_dist'  
    po4_N1_dist =prefix+'_po4_N1_dist'
    po4_occupancy = prefix+'_po4_occupancy'
    
    loop_ext_RMSD_avg = prefix+'_loop_ext_RMSD_avg'
    po4_xyz_RMSD_avg=prefix+'_po4_xyz_RMSD_avg'
    po4_OH1_dist_avg =prefix+'_po4_OH1_dist_avg'  
    po4_N1_dist_avg=prefix+'_po4_N1_dist_avg'

    
    loop_ext_RMSD_stdev = prefix+'_loop_ext_RMSD_stdev'
    po4_xyz_RMSD_stdev =prefix+'_po4_xyz_RMSD_stdev'
    po4_OH1_dist_stdev =prefix+'_po4_OH1_dist_stdev'  
    po4_N1_dist_stdev =prefix+'_po4_N1_dist_stdev'

    
    for i, frame in enumerate(dcd):

        #select protein 
        frame_fab = frame.getAtoms().select("protein not resname TIP3")

        #select loop & loop+flanking residues for each frame 
        #manually confirmed same # of atoms in crystal selections - required for later superposition 
        h_frag = frame_fab.select(  "resnum 108 109 110 " )
        l_frag = frame_fab.select(  "resnum 255 256 257 258 259 260 261 262 263  266 295 296 297 298 299 300" )

        full_res_frame_loop_ext = l_frag

        #use backbone atoms for alignemnt & RMSD calculations 
        bb_frame_loop_ext = full_res_frame_loop_ext.select("name CA ")#CB N 
        #print(bb_frame_loop_ext.getResnames())
        #select phos of interest - look in VMD final frame index of phos closest to loop 
        index_selection_str = "index "+ str(phos_index)
        frame_phos = frame.getAtoms().select(index_selection_str ) #"index 26328" 


        #superpose loop extended (no phos) - this will move whole system based on this superpose 
        #this is done so that xyz of crystal to md frame will be relevant in space 
        superpose(bb_frame_loop_ext, bb_xray_loop_4)


        #calc RMSD of extended loop   
        loop_ext_rmsd = calcRMSD(bb_frame_loop_ext, bb_xray_loop_4)
        globals()[prefix+'loop_ext_RMSD'].append(loop_ext_rmsd)

        po4_RMSD = calcRMS(frame_phos.getCoords()[0]- crys_phos_xyz[0])
        globals()[prefix+'po4_xyz_RMSD'].append(po4_RMSD) 

        #calc closest PO4 molecule to binding site 
        loop_edges = frame_fab.select( "resnum 259 261 262 109 and name CA")                 
        #frame_O1 = frame_fab.select( "resnum 260 and name O" )
        #frame_O2 = frame_fab.select( "resnum 261 and name O" )
        #frame_O3 = frame_fab.select( "resnum 262 and name O" )
        frame_O4 = frame_fab.select( "resnum 298 and name OG" )
        frame_nitrogen_1 = frame_fab.select("resnum 298 and name N")
        #frame_nitrogen_2 = frame_fab.select("resnum 263 and name N")


        #get distances of po4 to relevant atoms in loop
        po4_2_ser_OH1_dist = np.linalg.norm(frame_phos.getCoords()-frame_O4.getCoords())
        globals()[prefix+'po4_OH1_dist'].append(po4_2_ser_OH1_dist)
#         po4_2_ser_OH2_dist = np.linalg.norm(frame_phos.getCoords()-frame_ser_OH2.getCoords())
#         globals()[prefix+'po4_OH2_dist'].append(po4_2_ser_OH2_dist)
        po4_2_nitrogen_1_dist = np.linalg.norm(frame_phos.getCoords()-frame_nitrogen_1.getCoords())
        globals()[prefix+'po4_N1_dist'].append(po4_2_nitrogen_1_dist)
#         po4_2_nitrogen_2_dist = np.linalg.norm(frame_phos.getCoords()-frame_nitrogen_2.getCoords())
#         globals()[prefix+'po4_N2_dist'].append(po4_2_nitrogen_2_dist)


        #determine occupancy state 
        #if score is > 2 and RMSD is < 2.0, phosphate site is occupie 
        occupancy_score = 0 

        if po4_2_ser_OH1_dist<=5.25: 
            occupancy_score = occupancy_score+1 
#         if po4_2_ser_OH2_dist<=5.25:
#             occupancy_score = occupancy_score+1 
        if po4_2_nitrogen_1_dist<=5.25:
            occupancy_score = occupancy_score+1 
#         if po4_2_nitrogen_2_dist<=5.25:
#             occupancy_score = occupancy_score+1 

        if po4_RMSD<3.5 and occupancy_score >=0 :
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

    globals()[prefix+'po4_N1_dist_avg'] = [] 
    globals()[prefix+'po4_N1_dist_stdev'] = [] 
    calcRunningAvg(globals()[prefix+'po4_N1_dist'],
                   globals()[prefix+'po4_N1_dist_avg'],
                   globals()[prefix+'po4_N1_dist_stdev'], 10)  


    #print lists to files for accessing raw data later (i.e. plotting)
    lists_to_print = [globals()[prefix+'loop_ext_RMSD'], 
                      globals()[prefix+'po4_xyz_RMSD'],
                      globals()[prefix+'po4_OH1_dist'],
                      globals()[prefix+'po4_N1_dist'],
                      globals()[prefix+'po4_occupancy']] 
    lists_to_print_names = [loop_ext_RMSD,
                            po4_xyz_RMSD,
                            po4_OH1_dist,
                            po4_N1_dist,
                            po4_occupancy] 

    for i in range(len(lists_to_print)): 
        file_name = lists_to_print_names[i]+".csv"
        np.savetxt(file_name, lists_to_print[i], delimiter=",", fmt='%1.2f')
        
    #print avg & stdev lists to csv files         
    avg_lists_to_print = [globals()[prefix+'loop_ext_RMSD_avg'], 
                      globals()[prefix+'po4_xyz_RMSD_avg'],
                      globals()[prefix+'po4_OH1_dist_avg'],
                      globals()[prefix+'po4_N1_dist_avg']] 
    
    avg_lists_to_print_names = [loop_ext_RMSD_avg,
                                po4_xyz_RMSD_avg,
                                po4_OH1_dist_avg,
                                po4_N1_dist_avg,]
    
    stdev_lists_to_print = [globals()[prefix+'loop_ext_RMSD_stdev'], 
                      globals()[prefix+'po4_xyz_RMSD_stdev'],
                      globals()[prefix+'po4_OH1_dist_stdev'],
                      globals()[prefix+'po4_N1_dist_stdev']] 
    
    stdev_lists_to_print_names = [loop_ext_RMSD_stdev,
                                  po4_xyz_RMSD_stdev,
                                  po4_OH1_dist_stdev,
                                  po4_N1_dist_stdev ]

    for i in range(len(avg_lists_to_print)): 
        #write avg list 
        file_name = avg_lists_to_print_names[i]+".csv"
        np.savetxt(file_name, avg_lists_to_print[i], delimiter=",", fmt='%1.2f')
        #write stdev list 
        file_name = stdev_lists_to_print_names[i]+".csv"
        np.savetxt(file_name, stdev_lists_to_print[i], delimiter=",", fmt='%1.2f')
    return globals()[prefix+'po4_xyz_RMSD']  #lists_to_print_names 
      
def plot_phos_correlation(avg_rmsd, stdev_rmsd, po4_occupancy, prefix): 
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
    ax1.set_ylabel(r"${\rm PO_4}$ Bound", fontsize=40, rotation='horizontal', 
                   va="center", labelpad=120)
    ax1.set_xlim(-15, 5030)
    ax1.set_ylim(-1, 0)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    ax2.plot(x, avg_rmsd, lw=1, color='black')#00122E
    ax2.fill_between(x, avg_rmsd-stdev_rmsd, 
                     avg_rmsd+stdev_rmsd, 
                     color='black', alpha=0.25)
    ax2.set_xticks([0, 30, 1030, 2030, 3030, 4030, 5030]) 
    ax2.set_xticklabels(['', '0', '0.2', '0.4', '0.6', '0.8', '1.0' ])
    ax2.tick_params(axis='x', labelsize=40)
    ax2.tick_params(axis='y', labelsize=40)#, fontsize=16
    ax2.set_ylim(-.5, 2)
    ax2.set_xlim(-15, 5030)
    ax2.set_xlabel(r"Time ($\rm \mu s$)", fontsize=44)
    ax2.set_ylabel(r"${\rm PO_4}$ RMSD to X-Ray ($\rm \AA$)", fontsize=44, wrap=True)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    
    
    fig_name = prefix + '_po4_rmsd.png' 
    plt.savefig(fig_name, transparent=True, bbox_inches="tight")

    plt.show()
    return "Made figure: ", prefix


# In[6]:


def loop_RMSD_10e8(pdb_fp, dcd_fp, phos_index, prefix):
    #pdb_fp : full file path to input pdb 
    #dcd_fp: full filepath to input dcd 
    #phos idex: int of phos that should be tracked 
    #prefix : str of what to append to file paths 
    #NOTE: crystal structure stats must be available in environment before running this method 
    
    input_pdb = parsePDB(pdb_fp)
    dcd = DCDFile(dcd_fp)
    dcd.setCoords(input_pdb)
    dcd.link(input_pdb)
    dcd.reset()
        
    #initialize lists to store relevant values 
    #CDRL1 RMSD
    globals()[prefix+'loop_ext_RMSD']=[] 
#     globals()[prefix+'po4_xyz_RMSD']=[] 
    
#     globals()[prefix+'po4_OH1_dist']=[] 
#     globals()[prefix+'po4_N1_dist']=[] 
 
#     globals()[prefix+'po4_occupancy']=[] 
    
    #initalizat list names for writing files 
    loop_ext_RMSD = prefix+'_loop_ext_RMSD'

    loop_ext_RMSD_avg = prefix+'_loop_ext_RMSD_avg'
    
    loop_ext_RMSD_stdev = prefix+'_loop_ext_RMSD_stdev'
#  g

    
    for i, frame in enumerate(dcd):

        #select protein 
        frame_fab = frame.getAtoms().select("protein not resname TIP3")

        #select loop & loop+flanking residues for each frame 
        #manually confirmed same # of atoms in crystal selections - required for later superposition 
        h_frag = frame_fab.select(  "resnum 108 109 110 " )
        l_frag = frame_fab.select(  "resnum 255 256 257 258 259 260 261 262 263  266 295 296 297 298 299 300" )

        full_res_frame_loop_ext = l_frag

        #use backbone atoms for alignemnt & RMSD calculations 
        bb_frame_loop_ext = full_res_frame_loop_ext.select("name CA ")#CB N 
        #print(bb_frame_loop_ext.getResnames())
        #select phos of interest - look in VMD final frame index of phos closest to loop 
        index_selection_str = "index "+ str(phos_index)
        frame_phos = frame.getAtoms().select(index_selection_str ) #"index 26328" 


        #superpose loop extended (no phos) - this will move whole system based on this superpose 
        #this is done so that xyz of crystal to md frame will be relevant in space 
        superpose(bb_frame_loop_ext, bb_xray_loop_4)


        #calc RMSD of extended loop   
        loop_ext_rmsd = calcRMSD(bb_frame_loop_ext, bb_xray_loop_4)
        globals()[prefix+'loop_ext_RMSD'].append(loop_ext_rmsd)

        
    #calculate 10 step running avg & stdev for each list of values (for ease of plotting later )
    #dont need to do this for po4 occupancy 
    #also skipped tracked_po4_site_dist & closest_po4_site_disto4
    globals()[prefix+'loop_ext_RMSD_avg'] = [] 
    globals()[prefix+'loop_ext_RMSD_stdev'] = [] 
    calcRunningAvg(globals()[prefix+'loop_ext_RMSD'],
                   globals()[prefix+'loop_ext_RMSD_avg'],
                   globals()[prefix+'loop_ext_RMSD_stdev'], 10)

    lists_to_print_names = [loop_ext_RMSD] 

    for i in range(len(lists_to_print)): 
        file_name = lists_to_print_names[i]+".csv"
        np.savetxt(file_name, lists_to_print[i], delimiter=",", fmt='%1.2f')
        
    #print avg & stdev lists to csv files         
    avg_lists_to_print = [globals()[prefix+'loop_ext_RMSD_avg']] 
    
    avg_lists_to_print_names = [loop_ext_RMSD_avg]
    
    stdev_lists_to_print = [globals()[prefix+'loop_ext_RMSD_stdev']] 
    
    stdev_lists_to_print_names = [loop_ext_RMSD_stdev]

    for i in range(len(avg_lists_to_print)): 
        #write avg list 
        file_name = avg_lists_to_print_names[i]+".csv"
        np.savetxt(file_name, avg_lists_to_print[i], delimiter=",", fmt='%1.2f')
        #write stdev list 
        file_name = stdev_lists_to_print_names[i]+".csv"
        np.savetxt(file_name, stdev_lists_to_print[i], delimiter=",", fmt='%1.2f')
    return lists_to_print_names 


# In[7]:



def plot_phos_correlationv2(avg_rmsd, stdev_rmsd, po4_occupancy, prefix): 

    avg_rmsd = np.array(avg_rmsd)
    stdev_rmsd = np.array(stdev_rmsd)
    po4_occupancy = np.array(po4_occupancy)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 7), gridspec_kw={'height_ratios': [0.25, 6]})
    plt.subplots_adjust(hspace=0.1)
    avg_rmsd = avg_rmsd[0:2003]
    stdev_rmsd = stdev_rmsd[0:2003]

    #x = np.arange(0, len(avg_po4_xyz_RMSD_4e10_n15)*.0002, .0002)
    x = np.arange(0, len(avg_rmsd),1)

    ax1.fill_between(x, -1, 0, where=np.array(po4_occupancy[0:2003])==1, 
                     alpha=1, color='#53C767') 
    ax1.fill_between(x, -1, 0, where=np.array(po4_occupancy[0:2003])==0, 
                     alpha=1, color='#696969') 

    ax1.axhline(y=-1, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
    ax1.axhline(y=0, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
    ax1.axvline(x=5023, color='#000000', linestyle='-', ymin=0, linewidth=1)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    #ax1.set_ylabel(r"${\rm PO_4}$ Bound", fontsize=40, rotation='horizontal', 
    #               va="center", labelpad=120)
    ax1.set_xlim(0, 2003)
    ax1.set_ylim(-1, 0)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    ax2.plot(x, avg_rmsd, lw=1, color='black')#00122E
    ax2.fill_between(x, avg_rmsd-stdev_rmsd, 
                     avg_rmsd+stdev_rmsd, 
                     color='black', alpha=0.25)
    ax2.set_xticks([0, 530, 1030, 1530, 2030, 2530]) # 3030, 4030, 5030
    ax2.set_xticklabels(['0', '0.1', '0.2', "0.3", '0.4', "0.5" ])#'0.6', '0.8', '1.0' 
    ax2.tick_params(axis='x', labelsize=50)
    ax2.tick_params(axis='y', labelsize=50)#, fontsize=16
    ax2.set_ylim(-.5, 10)
    ax2.set_xlim(0, 2530)
    ax2.set_xlabel(r"Time ($\rm \mu s$)", fontsize=60)
    ax2.set_ylabel(r" RMSD ($\rm \AA$)", fontsize=60) #, wrap=True / to X-Ray 
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    
    fig_name = prefix + '_po4_rmsd.png' 
    plt.savefig(fig_name, transparent=True, bbox_inches="tight")

    plt.show()
    return "Made figure: ", prefix


