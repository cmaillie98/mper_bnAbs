#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os, numpy as np, re
from prody import *
from numpy import linalg as LA, zeros, arccos, roll
from collections import defaultdict
import matplotlib.pyplot as plt
from numpy.linalg import svd
import scipy.stats
from itertools import combinations
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import binom_test
import time
import Bio
from Bio.Cluster import kmedoids
from collections import Counter
import math
import csv
import seaborn as sns 


# Say, "the default sans-serif font is Arial"
plt.rcParams['font.sans-serif'] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
plt.rcParams['font.family'] = "sans-serif"


# In[2]:


xray_6snd = parsePDB("/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/LN01_TM-MPER/Xray_toCompare/LN01_peptideMPER_1PC-1PS-bound_ProtomerDE_6snd.pdb")
xray_6snd_fab = xray_6snd.select('protein and chain L and name CA or protein and chain H and name CA')
xray_tm = xray_6snd.select('chain P and name C')
print(len(xray_tm))
#print(len(xray_6snd_fab))
xray_trim=xray_6snd_fab[0:213]+xray_6snd_fab[214:272]+xray_6snd_fab[273:]
xray_phos_A = xray_6snd.select('resid 301 and name P')
xray_phos_B = xray_6snd.select('resid 302 and name P1')
xray_chol_B = xray_6snd.select('resid 302 and name N1')

# print(xray_phos_A.getResnames())
dcd_frame = parsePDB("/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/LN01_OPM_rep1/100ns_all.pdb")
dcd_fab = dcd_frame.select('protein and chain A and name CA or protein and chain B and name CA')
#print(len(dcd_fab))
xray_trim=xray_6snd_fab[0:213]+xray_6snd_fab[214:272]+xray_6snd_fab[273:]
dcd_trim=dcd_fab[0:212]+dcd_fab[213:355]+dcd_fab[361:442]

#CDRL1 : QSVTKY
print(dcd_trim[26:32].getResnames())
print(xray_trim[26:32].getResnames())
xray_cdrl1 = xray_trim[23:35]

wd="/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_tm_mper_02/"
pdb = parsePDB(wd+'1000ns_out.pdb')
#pdb_fab = pdb.select('protein and chain A and name CA or protein and chain B and name CA')
pdb_tm = pdb.select('chain C and name CA')
pdb_tm_trim = pdb_tm[3:22]
print(len(pdb_tm_trim))
# pdb_trim=pdb_fab[0:212]+pdb_fab[213:355]+pdb_fab[361:442]
# print(pdb_trim[26:32].getResnames())

print(pdb_tm_trim.getResnames())
print(xray_tm.getResnames())

xray_phos_coords = xray_phos_B.getCoords()
xray_chol_coords = xray_chol_B.getCoords()


# In[15]:


def lipid_site_analysis_ln01(frame, lipidA):
        #example input: 
        #outside / line above method: lipidA = frame.getAtoms().select('resnum 480 and resname POPC')
        #frame, lipidA
        #select ring carbons of aromatic residues
        tyr100g_aro = calcCenter(frame.getAtoms().select(' resnum 100G and resname TYR').select('name CD1 or name CD2 or name CE1 or name CE2 or name CZ '))
        tyr100i_aro = calcCenter(frame.getAtoms().select(' resnum 100I and resname TYR').select('name CD1 or name CD2 or name CE1 or name CE2 or name CZ '))
    
        trp100h_aro = calcCenter(frame.getAtoms().select(' resnum 100H and resname TRP').select('name CD2 or name CE2 or name CE3 or name CH2 or name CZ2 or name CZ3 '))
        tyr52_aro = calcCenter(frame.getAtoms().select(' resnum 52 and resname TYR and chain A').select('name CD1 or name CD2 or name CE1 or name CE2 or name CZ '))
        tyr32_aro = calcCenter(frame.getAtoms().select(' resnum 32 and resname TYR').select('name CD1 or name CD2 or name CE1 or name CE2 or name CZ '))
        tyr49_aro = calcCenter(frame.getAtoms().select(' resnum 49 and resname TYR').select('name CD1 or name CD2 or name CE1 or name CE2 or name CZ '))
        #trp680_aro = calcCenter(frame.getAtoms().select(' resnum 680 and resname TRP and chain C').select('name CD2 or name CE2 or name CE3 or name CH2 or name CZ2 or name CZ3 '))
        #tyr681_aro = calcCenter(frame.getAtoms().select(' resnum 681 and resname TYR and chain C').select('name CD1 or name CD2 or name CE1 or name CE2 or name CZ '))
        
        #print(trp100h_aro)
        #print(len(tyr52_aro.getResnames()))
    
        #select polar oxygens
        tyr100g_oh = getCoords(frame.getAtoms().select(' resnum 100G and resname TYR').select('name OH'))[0]
        tyr100i_oh = getCoords(frame.getAtoms().select(' resnum 100I and resname TYR').select('name OH'))[0]
        tyr52_oh = getCoords(frame.getAtoms().select(' resnum 52 and resname TYR and chain A ').select('name OH'))[0]
        tyr32_oh = getCoords(frame.getAtoms().select(' resnum 32 and resname TYR').select('name OH'))[0]
        tyr49_oh = getCoords(frame.getAtoms().select(' resnum 49 and resname TYR').select('name OH'))[0]
        #tyr681_oh = getCoords(frame.getAtoms().select(' resnum 681 and resname TYR').select('name OH'))[0]
        ser100d_oh = getCoords(frame.getAtoms().select(' resnum 100D and resname SER').select('name OG'))[0]\        ser100d_oh = getCoords(frame.getAtoms().select(' resnum 100D and resname SER').select('name OG'))[0]
        ser100f_oh = getCoords(frame.getAtoms().select(' resnum 100F and resname SER').select('name OG'))[0]

        thr100c_oh = getCoords(frame.getAtoms().select(' resnum 100C and resname THR').select('name OG1'))[0]
        thr30_o = getCoords(frame.getAtoms().select(' resnum 30 and resname THR').select('name O'))[0] #backbone carbonyl
    
        thr53_oh = getCoords(frame.getAtoms().select(' resnum 53 and resname THR and chain A').select('name OG1'))[0]
        #print(tyr100g_oh.getResnames())
    
        #select nitrogen aceptors 
        #trp680_n = getCoords(frame.getAtoms().select(' resnum 680 and resname TRP').select('name NE1'))[0]
        #lys683_n = getCoords(frame.getAtoms().select(' resnum 683 and resname LYS').select('name NZ'))[0]
        ser100_n = getCoords(frame.getAtoms().select(' resnum 100D and resname SER').select('name N'))[0]
        lys31_n = getCoords(frame.getAtoms().select(' resnum 31 and resname LYS').select('name NZ'))[0]
        
        lipidA_choline = getCoords(lipidA.select('name N'))[0]
        lipidA_phos = getCoords(lipidA.select('name P'))[0]
        
    
        loading_test_score = 0 
        ####cation pi cage with residues tyr49, tyr100G, tyr52
        #Deifniton - @ least 2 ring COM to Choline Nitrogen distances must be <5.5 A 
        cation_pi_score = 0 
        choline_test_1 = round(np.linalg.norm(lipidA_choline-tyr49_aro),2)
        choline_test_2 = round(np.linalg.norm(lipidA_choline-tyr52_aro),2)
        choline_test_3 = round(np.linalg.norm(lipidA_choline-tyr100g_aro),2)
        choline_test_4 = round(np.linalg.norm(lipidA_choline-trp100h_aro),2)
        choline_test_5 = round(np.linalg.norm(lipidA_choline-tyr100i_aro),2)
    
        #print(choline_test_1, choline_test_2, choline_test_3, choline_test_4)
        if choline_test_1<5.5:
            cation_pi_score=cation_pi_score+1
        if choline_test_2<5.5:
            cation_pi_score=cation_pi_score+1
        if choline_test_3<5.5:
            cation_pi_score=cation_pi_score+1
        if choline_test_4<5.5:
            cation_pi_score=cation_pi_score+1
        if choline_test_5<5.5:
            cation_pi_score=cation_pi_score+1    
        #print()
        
        choline_polar_score = 0 
        
        choline_polar_test_1 = round(np.linalg.norm(lipidA_choline-tyr49_oh),2)
        choline_polar_test_2 = round(np.linalg.norm(lipidA_choline-tyr52_oh),2)
        choline_polar_test_3 = round(np.linalg.norm(lipidA_choline-thr53_oh),2)
        choline_polar_test_4 = round(np.linalg.norm(lipidA_choline-thr100c_oh),2)
        choline_polar_test_5 = round(np.linalg.norm(lipidA_choline-ser100d_oh),2)
        choline_polar_test_6 = round(np.linalg.norm(lipidA_choline-tyr100g_oh),2)
        choline_polar_test_6 = round(np.linalg.norm(lipidA_choline-tyr100i_oh),2)
    
        if choline_polar_test_1<5.25:
            choline_polar_score+=1
        if choline_polar_test_2<5.25:
            choline_polar_score+=1
        if choline_polar_test_3<5.25:
            choline_polar_score+=1
        if choline_polar_test_4<5.25:
            choline_polar_score+=1
        if choline_polar_test_5<5.25:
            choline_polar_score+=1
        if choline_polar_test_6<5.25:
            choline_polar_score+=1
            
            
        phosphate_hbond_score = 0
        #phosphate_hbond_test_1 = round(np.linalg.norm(lipidA_phos-trp680_n),2)
        #phosphate_hbond_test_2 = round(np.linalg.norm(lipidA_phos-lys683_n),2)
        phosphate_hbond_test_3 = round(np.linalg.norm(lipidA_phos-ser100_n),2)
    #     if phosphate_hbond_test_1<5.25:
    #         phosphate_hbond_score+=1
    #     if phosphate_hbond_test_2<5.25:
    #         phosphate_hbond_score+=1
        if phosphate_hbond_test_3<5.25:
            phosphate_hbond_score+=1
    
        #define xray site interactions 
        xray_cation_pi_score = 0 
        xray_cationpi_test_1 = round(np.linalg.norm(lipidA_choline-tyr32_aro),2)
        xray_cationpi_test_2 = round(np.linalg.norm(lipidA_choline-tyr100g_aro),2)
        #xray_cationpi_test_3 = round(np.linalg.norm(lipidA_choline-trp680_aro),2)
        #xray_cationpi_test_4 = round(np.linalg.norm(lipidA_choline-tyr681_aro),2)
        
        if xray_cationpi_test_1<5.5:
            xray_cation_pi_score+=1
        if xray_cationpi_test_2<5.5:
            xray_cation_pi_score+=1
    #     if xray_cationpi_test_3<5.5:
    #         xray_cation_pi_score+=1
    #     if xray_cationpi_test_4<5.5:
    #         xray_cation_pi_score+=1
            
        xray_choline_polar_score = 0
        xray_choline_polar_test_1 = round(np.linalg.norm(lipidA_choline-tyr32_oh),2)
        xray_choline_polar_test_2 = round(np.linalg.norm(lipidA_choline-tyr100g_oh),2)
        xray_choline_polar_test_3 = round(np.linalg.norm(lipidA_choline-thr30_o),2)
        xray_choline_polar_test_4 = round(np.linalg.norm(lipidA_choline-ser100f_oh),2)
        #xray_choline_polar_test_4 = round(np.linalg.norm(lipidA_choline-tyr681_oh),2)
        if xray_choline_polar_test_1<5.25:
            xray_choline_polar_score+=1
        if xray_choline_polar_test_2<5.25:
            xray_choline_polar_score+=1
        if xray_choline_polar_test_3<5.25:
            xray_choline_polar_score+=1
        if xray_choline_polar_test_4<5.25:
            xray_choline_polar_score+=1
        
        xray_phosphate_hbond_score = 0
        xray_phosphate_hbond_test_1 = round(np.linalg.norm(lipidA_phos-lys31_n),2)
    
        if xray_phosphate_hbond_test_1<5.25:
            xray_phosphate_hbond_score+=1
            
        return cation_pi_score, choline_polar_score, phosphate_hbond_score, xray_cation_pi_score, xray_choline_polar_score, xray_phosphate_hbond_score


# In[36]:


def lipid_site_analysis_ln01_tm(frame, lipidA):
        #example input: lipidA = frame.getAtoms().select('resnum 480 and resname POPC')
        #example input: 
        #outside / line above method: lipidA = frame.getAtoms().select('resnum 480 and resname POPC')
        #frame, lipidA
        #select ring carbons of aromatic residues
        tyr100g_aro = calcCenter(frame.getAtoms().select(' resnum 100G and resname TYR').select('name CD1 or name CD2 or name CE1 or name CE2 or name CZ '))
        tyr100i_aro = calcCenter(frame.getAtoms().select(' resnum 100I and resname TYR').select('name CD1 or name CD2 or name CE1 or name CE2 or name CZ '))
    
        trp100h_aro = calcCenter(frame.getAtoms().select(' resnum 100H and resname TRP').select('name CD2 or name CE2 or name CE3 or name CH2 or name CZ2 or name CZ3 '))
        tyr52_aro = calcCenter(frame.getAtoms().select(' resnum 52 and resname TYR and chain A').select('name CD1 or name CD2 or name CE1 or name CE2 or name CZ '))
        tyr32_aro = calcCenter(frame.getAtoms().select(' resnum 32 and resname TYR').select('name CD1 or name CD2 or name CE1 or name CE2 or name CZ '))
        tyr49_aro = calcCenter(frame.getAtoms().select(' resnum 49 and resname TYR').select('name CD1 or name CD2 or name CE1 or name CE2 or name CZ '))
        trp680_aro = calcCenter(frame.getAtoms().select(' resnum 680 and resname TRP and chain C').select('name CD2 or name CE2 or name CE3 or name CH2 or name CZ2 or name CZ3 '))
        tyr681_aro = calcCenter(frame.getAtoms().select(' resnum 681 and resname TYR and chain C').select('name CD1 or name CD2 or name CE1 or name CE2 or name CZ '))
        
        #print(trp100h_aro)
        #print(len(tyr52_aro.getResnames()))
    
        #select polar oxygens
        tyr100g_oh = getCoords(frame.getAtoms().select(' resnum 100G and resname TYR').select('name OH'))[0]
        tyr100i_oh = getCoords(frame.getAtoms().select(' resnum 100I and resname TYR').select('name OH'))[0]
        tyr52_oh = getCoords(frame.getAtoms().select(' resnum 52 and resname TYR and chain A ').select('name OH'))[0]
        tyr32_oh = getCoords(frame.getAtoms().select(' resnum 32 and resname TYR').select('name OH'))[0]
        tyr49_oh = getCoords(frame.getAtoms().select(' resnum 49 and resname TYR').select('name OH'))[0]
        tyr681_oh = getCoords(frame.getAtoms().select(' resnum 681 and resname TYR').select('name OH'))[0]
        ser100d_oh = getCoords(frame.getAtoms().select(' resnum 100D and resname SER').select('name OG'))[0]
        thr100c_oh = getCoords(frame.getAtoms().select(' resnum 100C and resname THR').select('name OG1'))[0]
        thr30_o = getCoords(frame.getAtoms().select(' resnum 30 and resname THR').select('name O'))[0] #backbone carbonyl
    
        thr53_oh = getCoords(frame.getAtoms().select(' resnum 53 and resname THR and chain A').select('name OG1'))[0]
        #print(tyr100g_oh.getResnames())
    
        #select nitrogen aceptors 
        tyr681_oh = getCoords(frame.getAtoms().select(' resnum 681 and resname TYR').select('name OH'))[0]
        trp680_n = getCoords(frame.getAtoms().select(' resnum 680 and resname TRP').select('name NE1'))[0]
        lys683_n = getCoords(frame.getAtoms().select(' resnum 683 and resname LYS').select('name NZ'))[0]
        ser100_n = getCoords(frame.getAtoms().select(' resnum 100D and resname SER').select('name N'))[0]
        lys31_n = getCoords(frame.getAtoms().select(' resnum 31 and resname LYS').select('name NZ'))[0]
        
        lipidA_choline = getCoords(lipidA.select('name N'))[0]
        lipidA_phos = getCoords(lipidA.select('name P'))[0]
    
        loading_test_score = 0 
        ####cation pi cage with residues tyr49, tyr100G, tyr52
        #Deifniton - @ least 2 ring COM to Choline Nitrogen distances must be <5.5 A 
        cation_pi_score = 0 
        choline_test_1 = round(np.linalg.norm(lipidA_choline-tyr49_aro),2)
        choline_test_2 = round(np.linalg.norm(lipidA_choline-tyr52_aro),2)
        choline_test_3 = round(np.linalg.norm(lipidA_choline-tyr100g_aro),2)
        choline_test_4 = round(np.linalg.norm(lipidA_choline-trp100h_aro),2)
        choline_test_5 = round(np.linalg.norm(lipidA_choline-tyr100i_aro),2)
    
        #print(choline_test_1, choline_test_2, choline_test_3, choline_test_4)
        if choline_test_1<5.5:
            cation_pi_score=cation_pi_score+1
        if choline_test_2<5.5:
            cation_pi_score=cation_pi_score+1
        if choline_test_3<5.5:
            cation_pi_score=cation_pi_score+1
        if choline_test_4<5.5:
            cation_pi_score=cation_pi_score+1
        if choline_test_5<5.5:
            cation_pi_score=cation_pi_score+1    
        #print()
        
        choline_polar_score = 0 
        
        choline_polar_test_1 = round(np.linalg.norm(lipidA_choline-tyr49_oh),2)
        choline_polar_test_2 = round(np.linalg.norm(lipidA_choline-tyr52_oh),2)
        choline_polar_test_3 = round(np.linalg.norm(lipidA_choline-thr53_oh),2)
        choline_polar_test_4 = round(np.linalg.norm(lipidA_choline-thr100c_oh),2)
        choline_polar_test_5 = round(np.linalg.norm(lipidA_choline-ser100d_oh),2)
        choline_polar_test_6 = round(np.linalg.norm(lipidA_choline-tyr100g_oh),2)
        choline_polar_test_6 = round(np.linalg.norm(lipidA_choline-tyr100i_oh),2)
    
        if choline_polar_test_1<5.25:
            choline_polar_score+=1
        if choline_polar_test_2<5.25:
            choline_polar_score+=1
        if choline_polar_test_3<5.25:
            choline_polar_score+=1
        if choline_polar_test_4<5.25:
            choline_polar_score+=1
        if choline_polar_test_5<5.25:
            choline_polar_score+=1
        if choline_polar_test_6<5.25:
            choline_polar_score+=1
            
            
        phosphate_hbond_score = 0
        phosphate_hbond_test_1 = round(np.linalg.norm(lipidA_phos-trp680_n),2)
        phosphate_hbond_test_2 = round(np.linalg.norm(lipidA_phos-lys683_n),2)
        phosphate_hbond_test_3 = round(np.linalg.norm(lipidA_phos-ser100_n),2)
        if phosphate_hbond_test_1<5.25:
            phosphate_hbond_score+=1
        if phosphate_hbond_test_2<5.25:
            phosphate_hbond_score+=1
        if phosphate_hbond_test_3<5.25:
            phosphate_hbond_score+=1
    
        #define xray site interactions 
        xray_cation_pi_score = 0 
        xray_cationpi_test_1 = round(np.linalg.norm(lipidA_choline-tyr32_aro),2)
        xray_cationpi_test_2 = round(np.linalg.norm(lipidA_choline-tyr100g_aro),2)
        xray_cationpi_test_3 = round(np.linalg.norm(lipidA_choline-trp680_aro),2)
        xray_cationpi_test_4 = round(np.linalg.norm(lipidA_choline-tyr681_aro),2)
        
        if xray_cationpi_test_1<5.5:
            xray_cation_pi_score+=1
        if xray_cationpi_test_2<5.5:
            xray_cation_pi_score+=1
        if xray_cationpi_test_3<5.5:
            xray_cation_pi_score+=1
        if xray_cationpi_test_4<5.5:
            xray_cation_pi_score+=1
            
        xray_choline_polar_score = 0
        xray_choline_polar_test_1 = round(np.linalg.norm(lipidA_choline-tyr32_oh),2)
        xray_choline_polar_test_2 = round(np.linalg.norm(lipidA_choline-tyr100g_oh),2)
        xray_choline_polar_test_3 = round(np.linalg.norm(lipidA_choline-thr30_o),2)
        xray_choline_polar_test_4 = round(np.linalg.norm(lipidA_choline-tyr681_oh),2)
        if xray_choline_polar_test_1<5.25:
            xray_choline_polar_score+=1
        if xray_choline_polar_test_2<5.25:
            xray_choline_polar_score+=1
        if xray_choline_polar_test_3<5.25:
            xray_choline_polar_score+=1
        if xray_choline_polar_test_4<5.25:
            xray_choline_polar_score+=1
        
        xray_phosphate_hbond_score = 0
        xray_phosphate_hbond_test_1 = round(np.linalg.norm(lipidA_phos-lys31_n),2)
    
        if xray_phosphate_hbond_test_1<5.25:
            xray_phosphate_hbond_score+=1
            
        return cation_pi_score, choline_polar_score, phosphate_hbond_score, xray_cation_pi_score, xray_choline_polar_score, xray_phosphate_hbond_score


# In[3]:


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
def codify_scores(lipidB_scores):
    loading_site_occupancy_B  = 0
    if lipidB_scores[0]>=2:
        loading_site_occupancy_B=1
    elif lipidB_scores[1]>=2:
        loading_site_occupancy_B=1
    elif lipidB_scores[1]>=1 and lipidB_scores[2]>=1:
        loading_site_occupancy_B=1
    else:
        loading_site_occupancy_B=0 
    
    xray_site_occupancy_B  = 0
    if lipidB_scores[3]>=2:
        xray_site_occupancy_B=1
    elif lipidB_scores[4]>=2:
        xray_site_occupancy_B=1
    elif lipidB_scores[4]>=1 and lipidB_scores[5]>=1:
        xray_site_occupancy_B=1
    elif lipidB_scores[3]>=1 and lipidB_scores[4]>=1:
        xray_site_occupancy_B=1
    else:
        xray_site_occupancy_B=0 
    return loading_site_occupancy_B, xray_site_occupancy_B
    


# In[ ]:





# In[26]:


wd="/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_ppm_6snd_wyf_rep01/gromacs/"
pdb_fp = wd+"500ns_all.pdb"
dcd_fp = wd+"analysis.dcd"
input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

#initialize lists to store relevant values 
xray_site_track_01_A=[]
loading_site_track_01_A=[]

xray_site_track_01_B=[]
loading_site_track_01_B=[]

xray_site_track_01_C=[]
loading_site_track_01_C=[]


frame2xray_phos_rmsd_A=[]
frame2xray_chol_rmsd_A=[]

frame2xray_phos_rmsd_B=[]
frame2xray_chol_rmsd_B=[]

frame2xray_phos_rmsd_C=[]
frame2xray_chol_rmsd_C=[]
for i, frame in enumerate(dcd):

    #select protein 
    frame_fab = frame.getAtoms().select('protein and chain A and name CA or protein and chain B and name CA')



#     load_B xray_B  = codify_scores(lipidB_scores) 
#     loading_site_track_03_B.append(load_B)
#     xray_site_track_03_B.append(xray_B)


    frame_fab_trim = frame_fab[0:212]+frame_fab[213:355]+frame_fab[361:442]

    frame_crdl1 = frame_fab_trim[23:35] 

    #aln on CDR loop
    superpose(frame_crdl1, xray_cdrl1)
    
    
    #lipid to track 
    lipidA = frame.getAtoms().select('resnum 378 and resname POPC')
    
    #RMSD - A 
    lipidA_choline = getCoords(lipidA.select('name N'))[0]
    lipidA_phos = getCoords(lipidA.select('name P'))[0]

    phos_rmsd = calcRMS(lipidA_phos-xray_phos_coords)
    frame2xray_phos_rmsd_A.append(phos_rmsd)
    chol_rmsd = calcRMS(lipidA_choline-xray_chol_coords)
    frame2xray_chol_rmsd_A.append(chol_rmsd)

    #site occupancy - A 
    lipidA_scores = lipid_site_analysis_ln01(frame, lipidA)
    load_A, xray_A  = codify_scores(lipidA_scores) 
    if frame2xray_phos_rmsd_A<=2.5 and load_A==1:
        xray_site_track_01_A.append(1)
    elif frame2xray_chol_rmsd_A<=2.5 and load_A==1:
        xray_site_track_01_A.append(1) 
    else:
        xray_site_track_01_A.append(0)
    loading_site_track_01_A.append(load_A)
    


    #lipid to track -B 
    lipidB = frame.getAtoms().select('resnum 413 and resname POPC')
    lipidB_scores = lipid_site_analysis_ln01(frame, lipidB)

    #site occupancy - B
    load_B, xray_B  = codify_scores(lipidB_scores) 
    loading_site_track_01_B.append(load_B)
    xray_site_track_01_B.append(xray_B)

    #RMSD - B
    lipidB_choline = getCoords(lipidB.select('name N'))[0]
    lipidB_phos = getCoords(lipidB.select('name P'))[0]

    phos_rmsd = calcRMS(lipidB_phos-xray_phos_coords)
    frame2xray_phos_rmsd_B.append(phos_rmsd)
    chol_rmsd = calcRMS(lipidB_choline-xray_chol_coords)
    frame2xray_chol_rmsd_B.append(chol_rmsd)
    
    
        #lipid to track 
    lipidC = frame.getAtoms().select('resnum 398 and resname POPC')
    lipidC_scores = lipid_site_analysis_ln01(frame, lipidC)

    #site occupancy - B
    load_C, xray_C  = codify_scores(lipidC_scores) 
    loading_site_track_01_C.append(load_C)
    xray_site_track_01_C.append(xray_C)

    #RMSD - B
    lipidC_choline = getCoords(lipidC.select('name N'))[0]
    lipidC_phos = getCoords(lipidC.select('name P'))[0]

    phos_rmsd = calcRMS(lipidC_phos-xray_phos_coords)
    frame2xray_phos_rmsd_C.append(phos_rmsd)
    chol_rmsd = calcRMS(lipidC_choline-xray_chol_coords)
    frame2xray_chol_rmsd_C.append(chol_rmsd)



# In[29]:


file_out = 'ln01_ppm_6snd_wyf_rep01_phos_rmsd_A.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_phos_rmsd_A)
f.close()

file_out = 'ln01_ppm_6snd_wyf_rep01_chol_rmsd_A.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_chol_rmsd_A)
f.close()

with open('ln01_ppm_6snd_wyf_rep01_loading_occupancy_A.npy', 'wb') as f:
     np.save(f, loading_site_track_01_A)
f.close()

with open('ln01_ppm_6snd_wyf_rep01_xray_occupancy_A.npy', 'wb') as f:
     np.save(f, xray_site_track_01_A)
f.close()


file_out = 'ln01_ppm_6snd_wyf_rep01_phos_rmsd_B.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_phos_rmsd_B)
f.close()

file_out = 'ln01_ppm_6snd_wyf_rep01_chol_rmsd_B.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_chol_rmsd_B)
f.close()

with open('ln01_ppm_6snd_wyf_rep01_loading_occupancy_B.npy', 'wb') as f:
    np.save(f, loading_site_track_01_B)
f.close()

with open('ln01_ppm_6snd_wyf_rep01_xray_occupancy_B.npy', 'wb') as f:
     np.save(f, xray_site_track_01_B)
f.close()


file_out = 'ln01_ppm_6snd_wyf_rep01_phos_rmsd_C.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_phos_rmsd_C)
f.close()

file_out = 'ln01_ppm_6snd_wyf_rep01_chol_rmsd_C.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_chol_rmsd_C)
f.close()

with open('ln01_ppm_6snd_wyf_rep01_loading_occupancy_C.npy', 'wb') as f:
     np.save(f, loading_site_track_01_C)
f.close()

with open('ln01_ppm_6snd_wyf_rep01_xray_occupancy_C.npy', 'wb') as f:
     np.save(f, xray_site_track_01_C )
f.close()


# In[30]:


wd="/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_ppm_6snd_wyf_rep02/gromacs/"
pdb_fp = wd+"500ns_all.pdb"
dcd_fp = wd+"analysis.dcd"
input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

#initialize lists to store relevant values 
xray_site_track_02_A=[]
loading_site_track_02_A=[]

xray_site_track_02_B=[]
loading_site_track_02_B=[]



frame2xray_phos_rmsd_A=[]
frame2xray_chol_rmsd_A=[]

frame2xray_phos_rmsd_B=[]
frame2xray_chol_rmsd_B=[]


for i, frame in enumerate(dcd):

    #select protein 
    frame_fab = frame.getAtoms().select('protein and chain A and name CA or protein and chain B and name CA')



#     load_B xray_B  = codify_scores(lipidB_scores) 
#     loading_site_track_03_B.append(load_B)
#     xray_site_track_03_B.append(xray_B)


    frame_fab_trim = frame_fab[0:212]+frame_fab[213:355]+frame_fab[361:442]

    frame_crdl1 = frame_fab_trim[23:35] 

    #aln on CDR loop
    superpose(frame_crdl1, xray_cdrl1)
    
    
        #lipid to track 
    lipidA = frame.getAtoms().select('resnum 382 and resname POPC')
    lipidA_scores = lipid_site_analysis_ln01(frame, lipidA)

    #site occupancy - A 
    load_A, xray_A  = codify_scores(lipidA_scores) 
    loading_site_track_02_A.append(load_A)
    xray_site_track_02_A.append(xray_A)

    #RMSD - A 
    lipidA_choline = getCoords(lipidA.select('name N'))[0]
    lipidA_phos = getCoords(lipidA.select('name P'))[0]

    phos_rmsd = calcRMS(lipidA_phos-xray_phos_coords)
    frame2xray_phos_rmsd_A.append(phos_rmsd)
    chol_rmsd = calcRMS(lipidA_choline-xray_chol_coords)
    frame2xray_chol_rmsd_A.append(chol_rmsd)

    #lipid to track -B 
    lipidB = frame.getAtoms().select('resnum 385 and resname POPC')
    lipidB_scores = lipid_site_analysis_ln01(frame, lipidB)

    #site occupancy - B
    load_B, xray_B  = codify_scores(lipidB_scores) 
    loading_site_track_02_B.append(load_B)
    xray_site_track_02_B.append(xray_B)

    #RMSD - B
    lipidB_choline = getCoords(lipidB.select('name N'))[0]
    lipidB_phos = getCoords(lipidB.select('name P'))[0]

    phos_rmsd = calcRMS(lipidB_phos-xray_phos_coords)
    frame2xray_phos_rmsd_B.append(phos_rmsd)
    chol_rmsd = calcRMS(lipidB_choline-xray_chol_coords)
    frame2xray_chol_rmsd_B.append(chol_rmsd)
    



# In[32]:


file_out = 'ln01_ppm_6snd_wyf_rep02_phos_rmsd_A.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_phos_rmsd_A)
f.close()

file_out = 'ln01_ppm_6snd_wyf_rep02_chol_rmsd_A.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_chol_rmsd_A)
f.close()

with open('ln01_ppm_6snd_wyf_rep02_loading_occupancy_A.npy', 'wb') as f:
     np.save(f, loading_site_track_02_A)
f.close()

with open('ln01_ppm_6snd_wyf_rep02_xray_occupancy_A.npy', 'wb') as f:
     np.save(f, xray_site_track_02_A)
f.close()


file_out = 'ln01_ppm_6snd_wyf_rep02_phos_rmsd_B.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_phos_rmsd_B)
f.close()

file_out = 'ln01_ppm_6snd_wyf_rep02_chol_rmsd_B.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_chol_rmsd_B)
f.close()

with open('ln01_ppm_6snd_wyf_rep02_loading_occupancy_B.npy', 'wb') as f:
    np.save(f, loading_site_track_02_B)
f.close()

with open('ln01_ppm_6snd_wyf_rep02_xray_occupancy_B.npy', 'wb') as f:
     np.save(f, xray_site_track_02_B)
f.close()



# In[33]:


wd="/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_ppm_6snd_wyf_rep03/gromacs/"
pdb_fp = wd+"500ns_all.pdb"
dcd_fp = wd+"analysis.dcd"
input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

#initialize lists to store relevant values 
xray_site_track_03_A=[]
loading_site_track_03_A=[]

xray_site_track_03_B=[]
loading_site_track_03_B=[]



frame2xray_phos_rmsd_A=[]
frame2xray_chol_rmsd_A=[]

frame2xray_phos_rmsd_B=[]
frame2xray_chol_rmsd_B=[]

for i, frame in enumerate(dcd):

    #select protein 
    frame_fab = frame.getAtoms().select('protein and chain A and name CA or protein and chain B and name CA')



#     load_B xray_B  = codify_scores(lipidB_scores) 
#     loading_site_track_03_B.append(load_B)
#     xray_site_track_03_B.append(xray_B)


    frame_fab_trim = frame_fab[0:212]+frame_fab[213:355]+frame_fab[361:442]

    frame_crdl1 = frame_fab_trim[23:35] 

    #aln on CDR loop
    superpose(frame_crdl1, xray_cdrl1)
    
    
        #lipid to track 
    lipidA = frame.getAtoms().select('resnum 384 and resname POPC')
    lipidA_scores = lipid_site_analysis_ln01(frame, lipidA)

    #site occupancy - A 
    load_A, xray_A  = codify_scores(lipidA_scores) 
    loading_site_track_03_A.append(load_A)
    xray_site_track_03_A.append(xray_A)

    #RMSD - A 
    lipidA_choline = getCoords(lipidA.select('name N'))[0]
    lipidA_phos = getCoords(lipidA.select('name P'))[0]

    phos_rmsd = calcRMS(lipidA_phos-xray_phos_coords)
    frame2xray_phos_rmsd_A.append(phos_rmsd)
    chol_rmsd = calcRMS(lipidA_choline-xray_chol_coords)
    frame2xray_chol_rmsd_A.append(chol_rmsd)

    #lipid to track -B 
    lipidB = frame.getAtoms().select('resnum 375 and resname POPC')
    lipidB_scores = lipid_site_analysis_ln01(frame, lipidB)

    #site occupancy - B
    load_B, xray_B  = codify_scores(lipidB_scores) 
    loading_site_track_03_B.append(load_B)
    xray_site_track_03_B.append(xray_B)

    #RMSD - B
    lipidB_choline = getCoords(lipidB.select('name N'))[0]
    lipidB_phos = getCoords(lipidB.select('name P'))[0]

    phos_rmsd = calcRMS(lipidB_phos-xray_phos_coords)
    frame2xray_phos_rmsd_B.append(phos_rmsd)
    chol_rmsd = calcRMS(lipidB_choline-xray_chol_coords)
    frame2xray_chol_rmsd_B.append(chol_rmsd)

    




# In[34]:


file_out = 'ln01_ppm_6snd_wyf_rep03_phos_rmsd_A.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_phos_rmsd_A)
f.close()

file_out = 'ln01_ppm_6snd_wyf_rep03_chol_rmsd_A.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_chol_rmsd_A)
f.close()

with open('ln01_ppm_6snd_wyf_rep03_loading_occupancy_A.npy', 'wb') as f:
     np.save(f, loading_site_track_03_A)
f.close()

with open('ln01_ppm_6snd_wyf_rep03_xray_occupancy_A.npy', 'wb') as f:
     np.save(f, xray_site_track_03_A)
f.close()


file_out = 'ln01_ppm_6snd_wyf_rep03_phos_rmsd_B.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_phos_rmsd_B)
f.close()

file_out = 'ln01_ppm_6snd_wyf_rep03_chol_rmsd_B.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_chol_rmsd_B)
f.close()

with open('ln01_ppm_6snd_wyf_rep03_loading_occupancy_B.npy', 'wb') as f:
    np.save(f, loading_site_track_03_B)
f.close()

with open('ln01_ppm_6snd_wyf_rep03_xray_occupancy_B.npy', 'wb') as f:
     np.save(f, xray_site_track_03_B)
f.close()



# In[37]:


wd="/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_TM_ppm_6snd_wyf_rep02/gromacs/"
pdb_fp = wd+"500ns_all.pdb"
dcd_fp = wd+"analysis.dcd"
input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

#initialize lists to store relevant values 
xray_site_track_03_A=[]
loading_site_track_03_A=[]

xray_site_track_03_B=[]
loading_site_track_03_B=[]



frame2xray_phos_rmsd_A=[]
frame2xray_chol_rmsd_A=[]

frame2xray_phos_rmsd_B=[]
frame2xray_chol_rmsd_B=[]

for i, frame in enumerate(dcd):

    #select protein 
    frame_fab = frame.getAtoms().select('protein and chain A and name CA or protein and chain B and name CA')



#     load_B xray_B  = codify_scores(lipidB_scores) 
#     loading_site_track_03_B.append(load_B)
#     xray_site_track_03_B.append(xray_B)


    frame_fab_trim = frame_fab[0:212]+frame_fab[213:355]+frame_fab[361:442]

    frame_crdl1 = frame_fab_trim[23:35] 

    #aln on CDR loop
    superpose(frame_crdl1, xray_cdrl1)
    
    
        #lipid to track 
    lipidA = frame.getAtoms().select('resnum 907 and resname POPC')
    lipidA_scores = lipid_site_analysis_ln01_tm(frame, lipidA)

    #site occupancy - A 
    load_A, xray_A  = codify_scores(lipidA_scores) 
    loading_site_track_03_A.append(load_A)
    xray_site_track_03_A.append(xray_A)

    #RMSD - A 
    lipidA_choline = getCoords(lipidA.select('name N'))[0]
    lipidA_phos = getCoords(lipidA.select('name P'))[0]

    phos_rmsd = calcRMS(lipidA_phos-xray_phos_coords)
    frame2xray_phos_rmsd_A.append(phos_rmsd)
    chol_rmsd = calcRMS(lipidA_choline-xray_chol_coords)
    frame2xray_chol_rmsd_A.append(chol_rmsd)

    #lipid to track -B 
    lipidB = frame.getAtoms().select('resnum 908 and resname POPC')
    lipidB_scores = lipid_site_analysis_ln01_tm(frame, lipidB)

    #site occupancy - B
    load_B, xray_B  = codify_scores(lipidB_scores) 
    loading_site_track_03_B.append(load_B)
    xray_site_track_03_B.append(xray_B)

    #RMSD - B
    lipidB_choline = getCoords(lipidB.select('name N'))[0]
    lipidB_phos = getCoords(lipidB.select('name P'))[0]

    phos_rmsd = calcRMS(lipidB_phos-xray_phos_coords)
    frame2xray_phos_rmsd_B.append(phos_rmsd)
    chol_rmsd = calcRMS(lipidB_choline-xray_chol_coords)
    frame2xray_chol_rmsd_B.append(chol_rmsd)

    


# In[40]:


file_out = 'ln01_tm_ppm_6snd_wyf_rep02_phos_rmsd_A.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_phos_rmsd_A)
f.close()

file_out = 'ln01_tm_ppm_6snd_wyf_rep02_chol_rmsd_A.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_chol_rmsd_A)
f.close()

with open('ln01_tm_ppm_6snd_wyf_rep02_loading_occupancy_A.npy', 'wb') as f:
     np.save(f, loading_site_track_03_A)
f.close()

with open('ln01_tm_ppm_6snd_wyf_rep02_xray_occupancy_A.npy', 'wb') as f:
     np.save(f, xray_site_track_03_A)
f.close()


file_out = 'ln01_tm_ppm_6snd_wyf_rep02_phos_rmsd_B.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_phos_rmsd_B)
f.close()

file_out = 'ln01_tm_ppm_6snd_wyf_rep02_chol_rmsd_B.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_chol_rmsd_B)
f.close()

with open('ln01_tm_ppm_6snd_wyf_rep02_loading_occupancy_B.npy', 'wb') as f:
    np.save(f, loading_site_track_03_B)
f.close()

with open('ln01_tm_ppm_6snd_wyf_rep02_xray_occupancy_B.npy', 'wb') as f:
     np.save(f, xray_site_track_03_B)
f.close()



# In[43]:


wd="/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/LN01_TM-MPER/"
pdb_fp = wd+"500ns_all.pdb"
dcd_fp = wd+"analysis.dcd"
input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

#initialize lists to store relevant values 
xray_site_track_03_A=[]
loading_site_track_03_A=[]

xray_site_track_03_B=[]
loading_site_track_03_B=[]


frame2xray_phos_rmsd_A=[]
frame2xray_chol_rmsd_A=[]

frame2xray_phos_rmsd_B=[]
frame2xray_chol_rmsd_B=[]

for i, frame in enumerate(dcd):

    #select protein 
    frame_fab = frame.getAtoms().select('protein and chain A and name CA or protein and chain B and name CA')



#     load_B xray_B  = codify_scores(lipidB_scores) 
#     loading_site_track_03_B.append(load_B)
#     xray_site_track_03_B.append(xray_B)


    frame_fab_trim = frame_fab[0:212]+frame_fab[213:355]+frame_fab[361:442]

    frame_crdl1 = frame_fab_trim[23:35] 

    #aln on CDR loop
    superpose(frame_crdl1, xray_cdrl1)
    
    
        #lipid to track 
    lipidA = frame.getAtoms().select('resnum 1006 and resname POPC')
    lipidA_scores = lipid_site_analysis_ln01_tm(frame, lipidA)

    #site occupancy - A 
    load_A, xray_A  = codify_scores(lipidA_scores) 
    loading_site_track_03_A.append(load_A)
    xray_site_track_03_A.append(xray_A)

    #RMSD - A 
    lipidA_choline = getCoords(lipidA.select('name N'))[0]
    lipidA_phos = getCoords(lipidA.select('name P'))[0]

    phos_rmsd = calcRMS(lipidA_phos-xray_phos_coords)
    frame2xray_phos_rmsd_A.append(phos_rmsd)
    chol_rmsd = calcRMS(lipidA_choline-xray_chol_coords)
    frame2xray_chol_rmsd_A.append(chol_rmsd)

    #lipid to track -B 
    lipidB = frame.getAtoms().select('resnum 1019 and resname POPC')
    lipidB_scores = lipid_site_analysis_ln01_tm(frame, lipidB)

    #site occupancy - B
    load_B, xray_B  = codify_scores(lipidB_scores) 
    loading_site_track_03_B.append(load_B)
    xray_site_track_03_B.append(xray_B)

    #RMSD - B
    lipidB_choline = getCoords(lipidB.select('name N'))[0]
    lipidB_phos = getCoords(lipidB.select('name P'))[0]

    phos_rmsd = calcRMS(lipidB_phos-xray_phos_coords)
    frame2xray_phos_rmsd_B.append(phos_rmsd)
    chol_rmsd = calcRMS(lipidB_choline-xray_chol_coords)
    frame2xray_chol_rmsd_B.append(chol_rmsd)

    


# In[44]:


file_out = 'ln01_tm_ppm_6snd_wyf_MM_phos_rmsd_A.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_phos_rmsd_A)
f.close()

file_out = 'ln01_tm_ppm_6snd_wyf_MM_chol_rmsd_A.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_chol_rmsd_A)
f.close()

with open('ln01_tm_ppm_6snd_wyf_MM_loading_occupancy_A.npy', 'wb') as f:
     np.save(f, loading_site_track_03_A)
f.close()

with open('ln01_tm_ppm_6snd_wyf_MM_xray_occupancy_A.npy', 'wb') as f:
     np.save(f, xray_site_track_03_A)
f.close()


file_out = 'ln01_tm_ppm_6snd_wyf_MM_phos_rmsd_B.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_phos_rmsd_B)
f.close()

file_out = 'ln01_tm_ppm_6snd_wyf_MM_chol_rmsd_B.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_chol_rmsd_B)
f.close()

with open('ln01_tm_ppm_6snd_wyf_MM_loading_occupancy_B.npy', 'wb') as f:
    np.save(f, loading_site_track_03_B)
f.close()

with open('ln01_tm_ppm_6snd_wyf_MM_xray_occupancy_B.npy', 'wb') as f:
     np.save(f, xray_site_track_03_B)
f.close()



# In[45]:


wd="/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_TM_ppm_6snd_wyf_rep03/gromacs/"
pdb_fp = wd+"500ns_all.pdb"
dcd_fp = wd+"analysis.dcd"
input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

#initialize lists to store relevant values 
xray_site_track_03_A=[]
loading_site_track_03_A=[]

xray_site_track_03_B=[]
loading_site_track_03_B=[]



frame2xray_phos_rmsd_A=[]
frame2xray_chol_rmsd_A=[]

frame2xray_phos_rmsd_B=[]
frame2xray_chol_rmsd_B=[]

for i, frame in enumerate(dcd):

    #select protein 
    frame_fab = frame.getAtoms().select('protein and chain A and name CA or protein and chain B and name CA')



#     load_B xray_B  = codify_scores(lipidB_scores) 
#     loading_site_track_03_B.append(load_B)
#     xray_site_track_03_B.append(xray_B)


    frame_fab_trim = frame_fab[0:212]+frame_fab[213:355]+frame_fab[361:442]

    frame_crdl1 = frame_fab_trim[23:35] 

    #aln on CDR loop
    superpose(frame_crdl1, xray_cdrl1)
    
    
        #lipid to track 
    lipidA = frame.getAtoms().select('resnum 920 and resname POPC')
    lipidA_scores = lipid_site_analysis_ln01_tm(frame, lipidA)

    #site occupancy - A 
    load_A, xray_A  = codify_scores(lipidA_scores) 
    loading_site_track_03_A.append(load_A)
    xray_site_track_03_A.append(xray_A)

    #RMSD - A 
    lipidA_choline = getCoords(lipidA.select('name N'))[0]
    lipidA_phos = getCoords(lipidA.select('name P'))[0]

    phos_rmsd = calcRMS(lipidA_phos-xray_phos_coords)
    frame2xray_phos_rmsd_A.append(phos_rmsd)
    chol_rmsd = calcRMS(lipidA_choline-xray_chol_coords)
    frame2xray_chol_rmsd_A.append(chol_rmsd)

    #lipid to track -B 
#     lipidB = frame.getAtoms().select('resnum 908 and resname POPC')
#     lipidB_scores = lipid_site_analysis_ln01_tm(frame, lipidB)

#     #site occupancy - B
#     load_B, xray_B  = codify_scores(lipidB_scores) 
#     loading_site_track_03_B.append(load_B)
#     xray_site_track_03_B.append(xray_B)

#     #RMSD - B
#     lipidB_choline = getCoords(lipidB.select('name N'))[0]
#     lipidB_phos = getCoords(lipidB.select('name P'))[0]

#     phos_rmsd = calcRMS(lipidB_phos-xray_phos_coords)
#     frame2xray_phos_rmsd_B.append(phos_rmsd)
#     chol_rmsd = calcRMS(lipidB_choline-xray_chol_coords)
#     frame2xray_chol_rmsd_B.append(chol_rmsd)

    


# In[46]:


file_out = 'ln01_tm_wyf_03_lipidA_phos_rsmd.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_phos_rmsd_A)
f.close()

file_out = 'ln01_tm_wyf_03_lipidA_chol_rsmd.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_chol_rmsd_A)
f.close()

with open('ln01_tm_wyf_03_loading_occupancy_A.npy', 'wb') as f:
     np.save(f, loading_site_track_03_A)
f.close()

with open('ln01_tm_wyf_03_xray_occupancy_A.npy', 'wb') as f:
     np.save(f, xray_site_track_03_A)
f.close()


# In[33]:


def lipid_site_analysis_ln01(frame, lipidA):
        #example input: 
        #outside / line above method: lipidA = frame.getAtoms().select('resnum 480 and resname POPC')
        #frame, lipidA
        #select ring carbons of aromatic residues
        tyr100g_aro = calcCenter(frame.getAtoms().select(' resnum 100G and resname TYR').select('name CD1 or name CD2 or name CE1 or name CE2 or name CZ '))
        tyr100i_aro = calcCenter(frame.getAtoms().select(' resnum 100I and resname TYR').select('name CD1 or name CD2 or name CE1 or name CE2 or name CZ '))
        trp100h_aro = calcCenter(frame.getAtoms().select(' resnum 100H and resname TRP').select('name CD2 or name CE2 or name CE3 or name CH2 or name CZ2 or name CZ3 '))
        tyr52_aro = calcCenter(frame.getAtoms().select(' resnum 52 and resname TYR and chain A').select('name CD1 or name CD2 or name CE1 or name CE2 or name CZ '))
        tyr32_aro = calcCenter(frame.getAtoms().select(' resnum 32 and resname TYR').select('name CD1 or name CD2 or name CE1 or name CE2 or name CZ '))
        tyr49_aro = calcCenter(frame.getAtoms().select(' resnum 49 and resname TYR').select('name CD1 or name CD2 or name CE1 or name CE2 or name CZ '))
        #trp680_aro = calcCenter(frame.getAtoms().select(' resnum 680 and resname TRP and chain C').select('name CD2 or name CE2 or name CE3 or name CH2 or name CZ2 or name CZ3 '))
        #tyr681_aro = calcCenter(frame.getAtoms().select(' resnum 681 and resname TYR and chain C').select('name CD1 or name CD2 or name CE1 or name CE2 or name CZ '))
        

    
        #select polar oxygens
        tyr100g_oh = getCoords(frame.getAtoms().select(' resnum 100G and resname TYR').select('name OH'))[0]
        tyr100i_oh = getCoords(frame.getAtoms().select(' resnum 100I and resname TYR').select('name OH'))[0]
        tyr52_oh = getCoords(frame.getAtoms().select(' resnum 52 and resname TYR and chain A ').select('name OH'))[0]
        tyr32_oh = getCoords(frame.getAtoms().select(' resnum 32 and resname TYR').select('name OH'))[0]
        tyr49_oh = getCoords(frame.getAtoms().select(' resnum 49 and resname TYR').select('name OH'))[0]
        ser100d_oh = getCoords(frame.getAtoms().select(' resnum 100D and resname SER').select('name OG'))[0]#        ser100d_oh = getCoords(frame.getAtoms().select(' resnum 100D and resname SER').select('name OG'))[0]
        ser100f_oh = getCoords(frame.getAtoms().select(' resnum 100F and resname SER').select('name OG'))[0]

        thr100c_oh = getCoords(frame.getAtoms().select(' resnum 100C and resname THR').select('name OG1'))[0]
        thr30_o = getCoords(frame.getAtoms().select(' resnum 30 and resname THR').select('name O'))[0] #backbone carbonyl
        thr30_oh = getCoords(frame.getAtoms().select(' resnum 30 and resname THR').select('name OG1'))[0]
        thr53_oh = getCoords(frame.getAtoms().select(' resnum 53 and resname THR and chain A').select('name OG1'))[0]
        thr53_o = getCoords(frame.getAtoms().select(' resnum 53 and resname THR and chain A').select('name O'))[0]

    
        #select nitrogen aceptors 
        ser100_n = getCoords(frame.getAtoms().select(' resnum 100D and resname SER').select('name N'))[0]
        lys31_n = getCoords(frame.getAtoms().select(' resnum 31 and resname LYS').select('name NZ'))[0]
        
        lipidA_choline = getCoords(lipidA.select('name N'))[0]
        lipidA_phos = getCoords(lipidA.select('name P'))[0]
        
    
        loading_test_score = 0 
        
        #Deifniton - @ least 2 ring COM to Choline Nitrogen distances must be <5.5 A 
        cation_pi_score = 0 
        
        choline_test_1 = round(np.linalg.norm(lipidA_choline-tyr49_aro),2)
        choline_test_2 = round(np.linalg.norm(lipidA_choline-tyr52_aro),2)
        choline_test_3 = round(np.linalg.norm(lipidA_choline-tyr100g_aro),2)
    
        if choline_test_1<5.5:
            cation_pi_score=cation_pi_score+1
        if choline_test_2<5.5:
            cation_pi_score=cation_pi_score+1
        if choline_test_3<5.5:
            cation_pi_score=cation_pi_score+1
 
        
        choline_polar_score = 0 
        
        choline_polar_test_1 = round(np.linalg.norm(lipidA_choline-tyr49_oh),2)
        choline_polar_test_2 = round(np.linalg.norm(lipidA_choline-thr53_oh),2)
        choline_polar_test_3 = round(np.linalg.norm(lipidA_choline-thr53_o),2)
        choline_polar_test_4 = round(np.linalg.norm(lipidA_choline-thr100c_oh),2)
        choline_polar_test_5 = round(np.linalg.norm(lipidA_choline-ser100d_oh),2)
        choline_polar_test_6 = round(np.linalg.norm(lipidA_choline-tyr100g_oh),2)
        choline_polar_test_7 = round(np.linalg.norm(lipidA_choline-tyr52_oh),2)

        if choline_polar_test_1<5.25:
            choline_polar_score+=1
        if choline_polar_test_2<5.25:
            choline_polar_score+=1
        if choline_polar_test_3<5.25:
            choline_polar_score+=1
        if choline_polar_test_4<5.25:
            choline_polar_score+=1
        if choline_polar_test_5<5.25:
            choline_polar_score+=1
        if choline_polar_test_6<5.25:
            choline_polar_score+=1
        if choline_polar_test_7<5.25:
            choline_polar_score+=1
        phosphate_hbond_score=0
        #no phosphate hbonds in loading site interaction defintion 
        
        #define xray site interactions 
        xray_cation_pi_score = 0 
        
        xray_cationpi_test_1 = round(np.linalg.norm(lipidA_choline-tyr32_aro),2)
        xray_cationpi_test_2 = round(np.linalg.norm(lipidA_choline-tyr100g_aro),2)

        if xray_cationpi_test_1<5.5:
            xray_cation_pi_score+=1
        if xray_cationpi_test_2<5.5:
            xray_cation_pi_score+=1
            
            
        xray_choline_polar_score = 0
        
        xray_choline_polar_test_1 = round(np.linalg.norm(lipidA_choline-tyr32_oh),2)
        xray_choline_polar_test_2 = round(np.linalg.norm(lipidA_choline-tyr100g_oh),2)
        xray_choline_polar_test_3 = round(np.linalg.norm(lipidA_choline-thr30_o),2)
        xray_choline_polar_test_4 = round(np.linalg.norm(lipidA_choline-thr30_oh),2)
        xray_choline_polar_test_5 = round(np.linalg.norm(lipidA_choline-ser100f_oh),2)

        if xray_choline_polar_test_1<5.25:
            xray_choline_polar_score+=1
        if xray_choline_polar_test_2<5.25:
            xray_choline_polar_score+=1
        if xray_choline_polar_test_3<5.25:
            xray_choline_polar_score+=1
        if xray_choline_polar_test_4<5.25:
            xray_choline_polar_score+=1
        if xray_choline_polar_test_5<5.25:
            xray_choline_polar_score+=1
        
        
        xray_phosphate_hbond_score = 0
    
        xray_phosphate_hbond_test_1 = round(np.linalg.norm(lipidA_phos-lys31_n),2)
        
        if xray_phosphate_hbond_test_1<5.25:
            xray_phosphate_hbond_score+=1
            
        return cation_pi_score, choline_polar_score, phosphate_hbond_score, xray_cation_pi_score, xray_choline_polar_score, xray_phosphate_hbond_score


# In[46]:


def lipid_site_analysis_ln01_tm(frame, lipidA):
        #example input: 
        #outside / line above method: lipidA = frame.getAtoms().select('resnum 480 and resname POPC')
        #frame, lipidA
        #select ring carbons of aromatic residues
        tyr100g_aro = calcCenter(frame.getAtoms().select(' resnum 100G and resname TYR').select('name CD1 or name CD2 or name CE1 or name CE2 or name CZ '))
        tyr100i_aro = calcCenter(frame.getAtoms().select(' resnum 100I and resname TYR').select('name CD1 or name CD2 or name CE1 or name CE2 or name CZ '))
        trp100h_aro = calcCenter(frame.getAtoms().select(' resnum 100H and resname TRP').select('name CD2 or name CE2 or name CE3 or name CH2 or name CZ2 or name CZ3 '))
        tyr52_aro = calcCenter(frame.getAtoms().select(' resnum 52 and resname TYR and chain A').select('name CD1 or name CD2 or name CE1 or name CE2 or name CZ '))
        tyr32_aro = calcCenter(frame.getAtoms().select(' resnum 32 and resname TYR').select('name CD1 or name CD2 or name CE1 or name CE2 or name CZ '))
        tyr49_aro = calcCenter(frame.getAtoms().select(' resnum 49 and resname TYR').select('name CD1 or name CD2 or name CE1 or name CE2 or name CZ '))
        trp680_aro = calcCenter(frame.getAtoms().select(' resnum 680 and resname TRP and chain C').select('name CD2 or name CE2 or name CE3 or name CH2 or name CZ2 or name CZ3 '))
        tyr681_aro = calcCenter(frame.getAtoms().select(' resnum 681 and resname TYR and chain C').select('name CD1 or name CD2 or name CE1 or name CE2 or name CZ '))
        

        
        #select polar oxygens
        tyr100g_oh = getCoords(frame.getAtoms().select(' resnum 100G and resname TYR').select('name OH'))[0]
        tyr100i_oh = getCoords(frame.getAtoms().select(' resnum 100I and resname TYR').select('name OH'))[0]
        tyr52_oh = getCoords(frame.getAtoms().select(' resnum 52 and resname TYR and chain A ').select('name OH'))[0]
        tyr32_oh = getCoords(frame.getAtoms().select(' resnum 32 and resname TYR').select('name OH'))[0]
        tyr49_oh = getCoords(frame.getAtoms().select(' resnum 49 and resname TYR').select('name OH'))[0]
        ser100d_oh = getCoords(frame.getAtoms().select(' resnum 100D and resname SER').select('name OG'))[0]
        ser100f_oh = getCoords(frame.getAtoms().select(' resnum 100F and resname SER').select('name OG'))[0]

        thr100c_oh = getCoords(frame.getAtoms().select(' resnum 100C and resname THR').select('name OG1'))[0]
        thr30_o = getCoords(frame.getAtoms().select(' resnum 30 and resname THR').select('name O'))[0] #backbone carbonyl
        thr30_oh = getCoords(frame.getAtoms().select(' resnum 30 and resname THR').select('name OG1'))[0]
        thr53_oh = getCoords(frame.getAtoms().select(' resnum 53 and resname THR and chain A').select('name OG1'))[0]
        thr53_o = getCoords(frame.getAtoms().select(' resnum 53 and resname THR and chain A').select('name O'))[0]
        tyr681_oh = getCoords(frame.getAtoms().select(' resnum 681 and resname TYR').select('name OH'))[0]
    
        #select nitrogen aceptors 
        ser100_n = getCoords(frame.getAtoms().select(' resnum 100D and resname SER').select('name N'))[0]
        lys31_n = getCoords(frame.getAtoms().select(' resnum 31 and resname LYS').select('name NZ'))[0]
        trp680_n = getCoords(frame.getAtoms().select(' resnum 680 and resname TRP').select('name NE1'))[0]

        lipidA_choline = getCoords(lipidA.select('name N'))[0]
        lipidA_phos = getCoords(lipidA.select('name P'))[0]
        
    
        loading_test_score = 0 
        
        #Deifniton - @ least 2 ring COM to Choline Nitrogen distances must be <5.5 A 
        cation_pi_score = 0 
        
        choline_test_1 = round(np.linalg.norm(lipidA_choline-tyr49_aro),2)
        choline_test_2 = round(np.linalg.norm(lipidA_choline-tyr52_aro),2)
        choline_test_3 = round(np.linalg.norm(lipidA_choline-tyr100g_aro),2)
        
        if choline_test_1<5.5:
            cation_pi_score=cation_pi_score+1
        if choline_test_2<5.5:
            cation_pi_score=cation_pi_score+1
        if choline_test_3<5.5:
            cation_pi_score=cation_pi_score+1
        
        choline_polar_score = 0 
        
        choline_polar_test_1 = round(np.linalg.norm(lipidA_choline-tyr49_oh),2)
        choline_polar_test_2 = round(np.linalg.norm(lipidA_choline-thr53_oh),2)
        choline_polar_test_3 = round(np.linalg.norm(lipidA_choline-thr53_o),2)
        choline_polar_test_4 = round(np.linalg.norm(lipidA_choline-thr100c_oh),2)
        choline_polar_test_5 = round(np.linalg.norm(lipidA_choline-ser100d_oh),2)
        choline_polar_test_6 = round(np.linalg.norm(lipidA_choline-tyr100g_oh),2)
        choline_polar_test_7 = round(np.linalg.norm(lipidA_choline-tyr52_oh),2)

        if choline_polar_test_1<5.25:
            choline_polar_score+=1
        if choline_polar_test_2<5.25:
            choline_polar_score+=1
        if choline_polar_test_3<5.25:
            choline_polar_score+=1
        if choline_polar_test_4<5.25:
            choline_polar_score+=1
        if choline_polar_test_5<5.25:
            choline_polar_score+=1
        if choline_polar_test_6<5.25:
            choline_polar_score+=1
        if choline_polar_test_7<5.25:
            choline_polar_score+=1
        phosphate_hbond_score=0
        
        phosphate_hbond_test_1 = round(np.linalg.norm(lipidA_phos-trp680_n),2)
        
        if phosphate_hbond_test_1<5.25:
            phosphate_hbond_score+=1
        
        #define xray site interactions 
        xray_cation_pi_score = 0 
        
        xray_cationpi_test_1 = round(np.linalg.norm(lipidA_choline-tyr32_aro),2)
        xray_cationpi_test_2 = round(np.linalg.norm(lipidA_choline-tyr100g_aro),2)
        xray_cationpi_test_3 = round(np.linalg.norm(lipidA_choline-trp680_aro),2)
        xray_cationpi_test_4 = round(np.linalg.norm(lipidA_choline-tyr681_aro),2)

        if xray_cationpi_test_1<5.5:
            xray_cation_pi_score+=1
        if xray_cationpi_test_2<5.5:
            xray_cation_pi_score+=1
        if xray_cationpi_test_3<5.5:
            xray_cation_pi_score+=1
        if xray_cationpi_test_4<5.5:
            xray_cation_pi_score+=1
            
        xray_choline_polar_score = 0
        
        xray_choline_polar_test_1 = round(np.linalg.norm(lipidA_choline-tyr32_oh),2)
        xray_choline_polar_test_2 = round(np.linalg.norm(lipidA_choline-tyr100g_oh),2)
        xray_choline_polar_test_3 = round(np.linalg.norm(lipidA_choline-thr30_o),2)
        xray_choline_polar_test_4 = round(np.linalg.norm(lipidA_choline-thr30_oh),2)
        xray_choline_polar_test_5 = round(np.linalg.norm(lipidA_choline-ser100f_oh),2)
        xray_choline_polar_test_6 = round(np.linalg.norm(lipidA_choline-trp680_n),2)
        xray_choline_polar_test_7 = round(np.linalg.norm(lipidA_choline-tyr681_oh),2)
        if xray_choline_polar_test_1<5.25:
            xray_choline_polar_score+=1
        if xray_choline_polar_test_2<5.25:
            xray_choline_polar_score+=1
        if xray_choline_polar_test_3<5.25:
            xray_choline_polar_score+=1
        if xray_choline_polar_test_4<5.25:
            xray_choline_polar_score+=1
        if xray_choline_polar_test_5<5.25:
            xray_choline_polar_score+=1
        if xray_choline_polar_test_6<5.25:
            xray_choline_polar_score+=1        
        if xray_choline_polar_test_7<5.25:
            xray_choline_polar_score+=1
            
        xray_phosphate_hbond_score = 0
    
        xray_phosphate_hbond_test_1 = round(np.linalg.norm(lipidA_phos-lys31_n),2)
        
        if xray_phosphate_hbond_test_1<5.25:
            xray_phosphate_hbond_score+=1
            
        return cation_pi_score, choline_polar_score, phosphate_hbond_score, xray_cation_pi_score, xray_choline_polar_score, xray_phosphate_hbond_score


# In[35]:


def codify_scores(lipidB_scores, xray2phos_rmsd, xray2chol_rmsd):
    loading_site_occupancy_B  = 0
    #2 or more cation-pi interactions means a cation pi cage 
    if lipidB_scores[0]>=2:
        loading_site_occupancy_B=1
    #2 choline polar intereactions 
    elif lipidB_scores[1]>=2:
        loading_site_occupancy_B=1
    #1 polar choline and one polar phosphate 
    elif lipidB_scores[1]>=1 and lipidB_scores[2]>=1:
        loading_site_occupancy_B=1
    else:
        loading_site_occupancy_B=0 
        
    
    xray_site_occupancy_B  = 0
    #two cation-pi interactions forms cation pi cage 
    if lipidB_scores[3]>=2:
        xray_site_occupancy_B=1
    #two polar choline interactions 
    elif lipidB_scores[4]>=2:
        xray_site_occupancy_B=1
    #one polar choline and one polar phos 
    elif lipidB_scores[4]>=1 and lipidB_scores[5]>=1:
        xray_site_occupancy_B=1
    else:
        xray_site_occupancy_B=0 
        
    #add RMSD check to xray site 
    if xray_site_occupancy_B==1 and xray2phos_rmsd<=2.5:
        xray_site_occupancy_B=1
    elif xray_site_occupancy_B==1 and xray2chol_rmsd<=2.5:
        xray_site_occupancy_B=1
    else: xray_site_occupancy_B=0
    return loading_site_occupancy_B, xray_site_occupancy_B


# In[36]:


wd="/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_ppm_6snd_wyf_rep01/gromacs/"
pdb_fp = wd+"500ns_all.pdb"
dcd_fp = wd+"analysis.dcd"
input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

#initialize lists to store relevant values 
xray_site_track_01_A=[]
loading_site_track_01_A=[]

xray_site_track_01_B=[]
loading_site_track_01_B=[]


frame2xray_phos_rmsd_A=[]
frame2xray_chol_rmsd_A=[]

frame2xray_phos_rmsd_B=[]
frame2xray_chol_rmsd_B=[]


for i, frame in enumerate(dcd):

    #select protein 
    frame_fab = frame.getAtoms().select('protein and chain A and name CA or protein and chain B and name CA')

    frame_fab_trim = frame_fab[0:212]+frame_fab[213:355]+frame_fab[361:442]
    frame_crdl1 = frame_fab_trim[23:35] 

    #aln on CDR loop
    superpose(frame_crdl1, xray_cdrl1)
    
    
    #lipid to track  - should bine xray site in this simulation 
    lipidA = frame.getAtoms().select('resnum 413 and resname POPC')
    
    #RMSD - A 
    lipidA_choline = getCoords(lipidA.select('name N'))[0]
    lipidA_phos = getCoords(lipidA.select('name P'))[0]

    phos_rmsd = calcRMS(lipidA_phos-xray_phos_coords)
    chol_rmsd = calcRMS(lipidA_choline-xray_chol_coords)
    
    frame2xray_phos_rmsd_A.append(phos_rmsd)
    frame2xray_chol_rmsd_A.append(chol_rmsd)

    #site occupancy - A 
    lipidA_scores = lipid_site_analysis_ln01(frame, lipidA)
    load_A, xray_A  = codify_scores(lipidA_scores, phos_rmsd, chol_rmsd) 

    xray_site_track_01_A.append(xray_A)
    loading_site_track_01_A.append(load_A)

    #lipid to track -B should bind MD site in this simluation 
    lipidB = frame.getAtoms().select('resnum 398 and resname POPC')
    
    #RMSD 
    lipidB_choline = getCoords(lipidB.select('name N'))[0]
    lipidB_phos = getCoords(lipidB.select('name P'))[0]

    phos_rmsd = calcRMS(lipidB_phos-xray_phos_coords)
    chol_rmsd = calcRMS(lipidB_choline-xray_chol_coords)
    
    frame2xray_phos_rmsd_B.append(phos_rmsd)
    frame2xray_chol_rmsd_B.append(chol_rmsd)

    #site occupancy - A 
    lipidB_scores = lipid_site_analysis_ln01(frame, lipidB)
    load_B, xray_B  = codify_scores(lipidB_scores, phos_rmsd, chol_rmsd) 
    
    xray_site_track_01_B.append(xray_B)
    loading_site_track_01_B.append(load_B)
    #print(i)


# In[37]:


file_out = 'ln01_ppm_6snd_wyf_rep01_phos_rmsd_413_122222.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_phos_rmsd_A)
f.close()

file_out = 'ln01_ppm_6snd_wyf_rep01_chol_rmsd_413_122222.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_chol_rmsd_A)
f.close()

with open('ln01_ppm_6snd_wyf_rep01_loading_occupancy_413_122222.npy', 'wb') as f:
     np.save(f, loading_site_track_01_A)
f.close()

with open('ln01_ppm_6snd_wyf_rep01_xray_occupancy_413_122222.npy', 'wb') as f:
     np.save(f, xray_site_track_01_A)
f.close()


file_out = 'ln01_ppm_6snd_wyf_rep01_phos_rmsd_398_122222.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_phos_rmsd_B)
f.close()

file_out = 'ln01_ppm_6snd_wyf_rep01_chol_rmsd_398_122222.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_chol_rmsd_B)
f.close()

with open('ln01_ppm_6snd_wyf_rep01_loading_occupancy_398_122222.npy', 'wb') as f:
    np.save(f, loading_site_track_01_B)
f.close()

with open('ln01_ppm_6snd_wyf_rep01_xray_occupancy_398_122222.npy', 'wb') as f:
     np.save(f, xray_site_track_01_B)
f.close()


# In[38]:


wd="/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_ppm_6snd_wyf_rep02/gromacs/"
pdb_fp = wd+"500ns_all.pdb"
dcd_fp = wd+"analysis.dcd"
input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

#initialize lists to store relevant values 
xray_site_track_01_A=[]
loading_site_track_01_A=[]

xray_site_track_01_B=[]
loading_site_track_01_B=[]


frame2xray_phos_rmsd_A=[]
frame2xray_chol_rmsd_A=[]

frame2xray_phos_rmsd_B=[]
frame2xray_chol_rmsd_B=[]


for i, frame in enumerate(dcd):

    #select protein 
    frame_fab = frame.getAtoms().select('protein and chain A and name CA or protein and chain B and name CA')

    frame_fab_trim = frame_fab[0:212]+frame_fab[213:355]+frame_fab[361:442]
    frame_crdl1 = frame_fab_trim[23:35] 

    #aln on CDR loop
    superpose(frame_crdl1, xray_cdrl1)
    
    
    #lipid to track  - should bine xray site in this simulation 
    lipidA = frame.getAtoms().select('resnum 385 and resname POPC')
    
    #RMSD - A 
    lipidA_choline = getCoords(lipidA.select('name N'))[0]
    lipidA_phos = getCoords(lipidA.select('name P'))[0]

    phos_rmsd = calcRMS(lipidA_phos-xray_phos_coords)
    chol_rmsd = calcRMS(lipidA_choline-xray_chol_coords)
    
    frame2xray_phos_rmsd_A.append(phos_rmsd)
    frame2xray_chol_rmsd_A.append(chol_rmsd)

    #site occupancy - A 
    lipidA_scores = lipid_site_analysis_ln01(frame, lipidA)
    load_A, xray_A  = codify_scores(lipidA_scores, phos_rmsd, chol_rmsd) 

    xray_site_track_01_A.append(xray_A)
    loading_site_track_01_A.append(load_A)

    #lipid to track -B should bind MD site in this simluation 
    lipidB = frame.getAtoms().select('resnum 377 and resname POPC')
    
    #RMSD 
    lipidB_choline = getCoords(lipidB.select('name N'))[0]
    lipidB_phos = getCoords(lipidB.select('name P'))[0]

    phos_rmsd = calcRMS(lipidB_phos-xray_phos_coords)
    chol_rmsd = calcRMS(lipidB_choline-xray_chol_coords)
    
    frame2xray_phos_rmsd_B.append(phos_rmsd)
    frame2xray_chol_rmsd_B.append(chol_rmsd)

    #site occupancy - A 
    lipidB_scores = lipid_site_analysis_ln01(frame, lipidB)
    load_B, xray_B  = codify_scores(lipidB_scores, phos_rmsd, chol_rmsd) 
    
    xray_site_track_01_B.append(xray_B)
    loading_site_track_01_B.append(load_B)
    #print(i)


# In[39]:


file_out = 'ln01_ppm_6snd_wyf_rep02_phos_rmsd_385_122722.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_phos_rmsd_A)
f.close()

file_out = 'ln01_ppm_6snd_wyf_rep02_chol_rmsd_385_122722.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_chol_rmsd_A)
f.close()

with open('ln01_ppm_6snd_wyf_rep02_loading_occupancy_385_122722.npy', 'wb') as f:
     np.save(f, loading_site_track_01_A)
f.close()

with open('ln01_ppm_6snd_wyf_rep02_xray_occupancy_385_122722.npy', 'wb') as f:
     np.save(f, xray_site_track_01_A)
f.close()


file_out = 'ln01_ppm_6snd_wyf_rep02_phos_rmsd_377_122722.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_phos_rmsd_B)
f.close()

file_out = 'ln01_ppm_6snd_wyf_rep02_chol_rmsd_377_122722.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_chol_rmsd_B)
f.close()

with open('ln01_ppm_6snd_wyf_rep02_loading_occupancy_377_122722.npy', 'wb') as f:
    np.save(f, loading_site_track_01_B)
f.close()

with open('ln01_ppm_6snd_wyf_rep02_xray_occupancy_377_122722.npy', 'wb') as f:
     np.save(f, xray_site_track_01_B)
f.close()


# In[40]:


wd="/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_ppm_6snd_wyf_rep03/gromacs/"
pdb_fp = wd+"500ns_all.pdb"
dcd_fp = wd+"analysis.dcd"
input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

#initialize lists to store relevant values 
xray_site_track_01_A=[]
loading_site_track_01_A=[]

xray_site_track_01_B=[]
loading_site_track_01_B=[]


frame2xray_phos_rmsd_A=[]
frame2xray_chol_rmsd_A=[]

frame2xray_phos_rmsd_B=[]
frame2xray_chol_rmsd_B=[]


for i, frame in enumerate(dcd):

    #select protein 
    frame_fab = frame.getAtoms().select('protein and chain A and name CA or protein and chain B and name CA')

    frame_fab_trim = frame_fab[0:212]+frame_fab[213:355]+frame_fab[361:442]
    frame_crdl1 = frame_fab_trim[23:35] 

    #aln on CDR loop
    superpose(frame_crdl1, xray_cdrl1)
    
    
    #lipid to track  - should bine xray site in this simulation 
    lipidA = frame.getAtoms().select('resnum 375 and resname POPC')
    
    #RMSD - A 
    lipidA_choline = getCoords(lipidA.select('name N'))[0]
    lipidA_phos = getCoords(lipidA.select('name P'))[0]

    phos_rmsd = calcRMS(lipidA_phos-xray_phos_coords)
    chol_rmsd = calcRMS(lipidA_choline-xray_chol_coords)
    
    frame2xray_phos_rmsd_A.append(phos_rmsd)
    frame2xray_chol_rmsd_A.append(chol_rmsd)

    #site occupancy - A 
    lipidA_scores = lipid_site_analysis_ln01(frame, lipidA)
    load_A, xray_A  = codify_scores(lipidA_scores, phos_rmsd, chol_rmsd) 

    xray_site_track_01_A.append(xray_A)
    loading_site_track_01_A.append(load_A)

    #lipid to track -B should bind MD site in this simluation 
    lipidB = frame.getAtoms().select('resnum 384 and resname POPC')
    
    #RMSD 
    lipidB_choline = getCoords(lipidB.select('name N'))[0]
    lipidB_phos = getCoords(lipidB.select('name P'))[0]

    phos_rmsd = calcRMS(lipidB_phos-xray_phos_coords)
    chol_rmsd = calcRMS(lipidB_choline-xray_chol_coords)
    
    frame2xray_phos_rmsd_B.append(phos_rmsd)
    frame2xray_chol_rmsd_B.append(chol_rmsd)

    #site occupancy - A 
    lipidB_scores = lipid_site_analysis_ln01(frame, lipidB)
    load_B, xray_B  = codify_scores(lipidB_scores, phos_rmsd, chol_rmsd) 
    
    xray_site_track_01_B.append(xray_B)
    loading_site_track_01_B.append(load_B)
    #print(i)


# In[41]:


file_out = 'ln01_ppm_6snd_wyf_rep03_phos_rmsd_375_122722.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_phos_rmsd_A)
f.close()

file_out = 'ln01_ppm_6snd_wyf_rep03_chol_rmsd_375_122722.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_chol_rmsd_A)
f.close()

with open('ln01_ppm_6snd_wyf_rep03_loading_occupancy_375_122722.npy', 'wb') as f:
     np.save(f, loading_site_track_01_A)
f.close()

with open('ln01_ppm_6snd_wyf_rep03_xray_occupancy_375_122722.npy', 'wb') as f:
     np.save(f, xray_site_track_01_A)
f.close()


file_out = 'ln01_ppm_6snd_wyf_rep03_phos_rmsd_384_122722.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_phos_rmsd_B)
f.close()

file_out = 'ln01_ppm_6snd_wyf_rep03_chol_rmsd_384_122722.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_chol_rmsd_B)
f.close()

with open('ln01_ppm_6snd_wyf_rep03_loading_occupancy_384_122722.npy', 'wb') as f:
    np.save(f, loading_site_track_01_B)
f.close()

with open('ln01_ppm_6snd_wyf_rep03_xray_occupancy_384_122722.npy', 'wb') as f:
     np.save(f, xray_site_track_01_B)
f.close()


# In[42]:


wd="/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/LN01_OPM_rep1/"
pdb_fp = wd+"100ns_all.pdb"
dcd_fp = wd+"analysis.dcd"
input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

#initialize lists to store relevant values 
xray_site_track_01_A=[]
loading_site_track_01_A=[]

xray_site_track_01_B=[]
loading_site_track_01_B=[]


frame2xray_phos_rmsd_A=[]
frame2xray_chol_rmsd_A=[]

frame2xray_phos_rmsd_B=[]
frame2xray_chol_rmsd_B=[]


for i, frame in enumerate(dcd):

    #select protein 
    frame_fab = frame.getAtoms().select('protein and chain A and name CA or protein and chain B and name CA')

    frame_fab_trim = frame_fab[0:212]+frame_fab[213:355]+frame_fab[361:442]
    frame_crdl1 = frame_fab_trim[23:35] 

    #aln on CDR loop
    superpose(frame_crdl1, xray_cdrl1)
    
    
    #lipid to track  - should bine xray site in this simulation 
    lipidA = frame.getAtoms().select('resnum 480 and resname POPC')
    
    #RMSD - A 
    lipidA_choline = getCoords(lipidA.select('name N'))[0]
    lipidA_phos = getCoords(lipidA.select('name P'))[0]

    phos_rmsd = calcRMS(lipidA_phos-xray_phos_coords)
    chol_rmsd = calcRMS(lipidA_choline-xray_chol_coords)
    
    frame2xray_phos_rmsd_A.append(phos_rmsd)
    frame2xray_chol_rmsd_A.append(chol_rmsd)

    #site occupancy - A 
    lipidA_scores = lipid_site_analysis_ln01(frame, lipidA)
    load_A, xray_A  = codify_scores(lipidA_scores, phos_rmsd, chol_rmsd) 

    xray_site_track_01_A.append(xray_A)
    loading_site_track_01_A.append(load_A)

    #lipid to track -B should bind MD site in this simluation 
    lipidB = frame.getAtoms().select('resnum 500 and resname POPC')
    
    #RMSD 
    lipidB_choline = getCoords(lipidB.select('name N'))[0]
    lipidB_phos = getCoords(lipidB.select('name P'))[0]

    phos_rmsd = calcRMS(lipidB_phos-xray_phos_coords)
    chol_rmsd = calcRMS(lipidB_choline-xray_chol_coords)
    
    frame2xray_phos_rmsd_B.append(phos_rmsd)
    frame2xray_chol_rmsd_B.append(chol_rmsd)

    #site occupancy - A 
    lipidB_scores = lipid_site_analysis_ln01(frame, lipidB)
    load_B, xray_B  = codify_scores(lipidB_scores, phos_rmsd, chol_rmsd) 
    
    xray_site_track_01_B.append(xray_B)
    loading_site_track_01_B.append(load_B)
    #print(i)


# In[43]:


file_out = 'ln01_ppm_6snd_wyf_repMM_phos_rmsd_480_122722.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_phos_rmsd_A)
f.close()

file_out = 'ln01_ppm_6snd_wyf_repMM_chol_rmsd_480_122722.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_chol_rmsd_A)
f.close()

with open('ln01_ppm_6snd_wyf_repMM_loading_occupancy_480_122722.npy', 'wb') as f:
     np.save(f, loading_site_track_01_A)
f.close()

with open('ln01_ppm_6snd_wyf_repMM_xray_occupancy_480_122722.npy', 'wb') as f:
     np.save(f, xray_site_track_01_A)
f.close()


file_out = 'ln01_ppm_6snd_wyf_repMM_phos_rmsd_500_122722.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_phos_rmsd_B)
f.close()

file_out = 'ln01_ppm_6snd_wyf_repMM_chol_rmsd_500_122722.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_chol_rmsd_B)
f.close()

with open('ln01_ppm_6snd_wyf_repMM_loading_occupancy_500_122722.npy', 'wb') as f:
    np.save(f, loading_site_track_01_B)
f.close()

with open('ln01_ppm_6snd_wyf_repMM_xray_occupancy_500_122722.npy', 'wb') as f:
     np.save(f, xray_site_track_01_B)
f.close()


# In[47]:


wd="/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/LN01_TM-MPER/"
pdb_fp = wd+"500ns_all.pdb"
dcd_fp = wd+"analysis.dcd"
input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

#initialize lists to store relevant values 
xray_site_track_01_A=[]
loading_site_track_01_A=[]

xray_site_track_01_B=[]
loading_site_track_01_B=[]


frame2xray_phos_rmsd_A=[]
frame2xray_chol_rmsd_A=[]

frame2xray_phos_rmsd_B=[]
frame2xray_chol_rmsd_B=[]


for i, frame in enumerate(dcd):

    #select protein 
    frame_fab = frame.getAtoms().select('protein and chain A and name CA or protein and chain B and name CA')

    frame_fab_trim = frame_fab[0:212]+frame_fab[213:355]+frame_fab[361:442]
    frame_crdl1 = frame_fab_trim[23:35] 

    #aln on CDR loop
    superpose(frame_crdl1, xray_cdrl1)
    
    
    #lipid to track  - should bine xray site in this simulation 
    lipidA = frame.getAtoms().select('resnum 1006 and resname POPC')
    
    #RMSD - A 
    lipidA_choline = getCoords(lipidA.select('name N'))[0]
    lipidA_phos = getCoords(lipidA.select('name P'))[0]

    phos_rmsd = calcRMS(lipidA_phos-xray_phos_coords)
    chol_rmsd = calcRMS(lipidA_choline-xray_chol_coords)
    
    frame2xray_phos_rmsd_A.append(phos_rmsd)
    frame2xray_chol_rmsd_A.append(chol_rmsd)

    #site occupancy - A 
    lipidA_scores = lipid_site_analysis_ln01_tm(frame, lipidA)
    load_A, xray_A  = codify_scores(lipidA_scores, phos_rmsd, chol_rmsd) 

    xray_site_track_01_A.append(xray_A)
    loading_site_track_01_A.append(load_A)

    #lipid to track -B should bind MD site in this simluation 
    lipidB = frame.getAtoms().select('resnum 1019 and resname POPC')
    
    #RMSD 
    lipidB_choline = getCoords(lipidB.select('name N'))[0]
    lipidB_phos = getCoords(lipidB.select('name P'))[0]

    phos_rmsd = calcRMS(lipidB_phos-xray_phos_coords)
    chol_rmsd = calcRMS(lipidB_choline-xray_chol_coords)
    
    frame2xray_phos_rmsd_B.append(phos_rmsd)
    frame2xray_chol_rmsd_B.append(chol_rmsd)

    #site occupancy - A 
    lipidB_scores = lipid_site_analysis_ln01_tm(frame, lipidB)
    load_B, xray_B  = codify_scores(lipidB_scores, phos_rmsd, chol_rmsd) 
    
    xray_site_track_01_B.append(xray_B)
    loading_site_track_01_B.append(load_B)
    #print(i)


# In[48]:


file_out = 'ln01_tm_ppm_6snd_wyf_repMM_phos_rmsd_1006_122722.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_phos_rmsd_A)
f.close()

file_out = 'ln01_tm_ppm_6snd_wyf_repMM_chol_rmsd_1006_122722.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_chol_rmsd_A)
f.close()

with open('ln01_tm_ppm_6snd_wyf_repMM_loading_occupancy_1006_122722.npy', 'wb') as f:
     np.save(f, loading_site_track_01_A)
f.close()

with open('ln01_tm_ppm_6snd_wyf_repMM_xray_occupancy_1006_122722.npy', 'wb') as f:
     np.save(f, xray_site_track_01_A)
f.close()


file_out = 'ln01_tm_ppm_6snd_wyf_repMM_phos_rmsd_1019_122722.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_phos_rmsd_B)
f.close()

file_out = 'ln01_tm_ppm_6snd_wyf_repMM_chol_rmsd_1019_122722.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_chol_rmsd_B)
f.close()

with open('ln01_tm_ppm_6snd_wyf_repMM_loading_occupancy_1019_122722.npy', 'wb') as f:
    np.save(f, loading_site_track_01_B)
f.close()

with open('ln01_tm_ppm_6snd_wyf_repMM_xray_occupancy_1019_122722.npy', 'wb') as f:
     np.save(f, xray_site_track_01_B)
f.close()


# In[49]:


wd="/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_TM_ppm_6snd_wyf_rep02/gromacs/"
pdb_fp = wd+"500ns_all.pdb"
dcd_fp = wd+"analysis.dcd"
input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

#initialize lists to store relevant values 
xray_site_track_01_A=[]
loading_site_track_01_A=[]

xray_site_track_01_B=[]
loading_site_track_01_B=[]


frame2xray_phos_rmsd_A=[]
frame2xray_chol_rmsd_A=[]

frame2xray_phos_rmsd_B=[]
frame2xray_chol_rmsd_B=[]


for i, frame in enumerate(dcd):

    #select protein 
    frame_fab = frame.getAtoms().select('protein and chain A and name CA or protein and chain B and name CA')

    frame_fab_trim = frame_fab[0:212]+frame_fab[213:355]+frame_fab[361:442]
    frame_crdl1 = frame_fab_trim[23:35] 

    #aln on CDR loop
    superpose(frame_crdl1, xray_cdrl1)
    
    
    #lipid to track  - should bine xray site in this simulation 
    lipidA = frame.getAtoms().select('resnum 907 and resname POPC')
    
    #RMSD - A 
    lipidA_choline = getCoords(lipidA.select('name N'))[0]
    lipidA_phos = getCoords(lipidA.select('name P'))[0]

    phos_rmsd = calcRMS(lipidA_phos-xray_phos_coords)
    chol_rmsd = calcRMS(lipidA_choline-xray_chol_coords)
    
    frame2xray_phos_rmsd_A.append(phos_rmsd)
    frame2xray_chol_rmsd_A.append(chol_rmsd)

    #site occupancy - A 
    lipidA_scores = lipid_site_analysis_ln01_tm(frame, lipidA)
    load_A, xray_A  = codify_scores(lipidA_scores, phos_rmsd, chol_rmsd) 

    xray_site_track_01_A.append(xray_A)
    loading_site_track_01_A.append(load_A)

    #lipid to track -B should bind MD site in this simluation 
    lipidB = frame.getAtoms().select('resnum 908 and resname POPC')
    
    #RMSD 
    lipidB_choline = getCoords(lipidB.select('name N'))[0]
    lipidB_phos = getCoords(lipidB.select('name P'))[0]

    phos_rmsd = calcRMS(lipidB_phos-xray_phos_coords)
    chol_rmsd = calcRMS(lipidB_choline-xray_chol_coords)
    
    frame2xray_phos_rmsd_B.append(phos_rmsd)
    frame2xray_chol_rmsd_B.append(chol_rmsd)

    #site occupancy - A 
    lipidB_scores = lipid_site_analysis_ln01_tm(frame, lipidB)
    load_B, xray_B  = codify_scores(lipidB_scores, phos_rmsd, chol_rmsd) 
    
    xray_site_track_01_B.append(xray_B)
    loading_site_track_01_B.append(load_B)
    #print(i)


# In[50]:


file_out = 'ln01_tm_ppm_6snd_wyf_rep02_phos_rmsd_907_122722.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_phos_rmsd_A)
f.close()

file_out = 'ln01_tm_ppm_6snd_wyf_rep02_chol_rmsd_907_122722.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_chol_rmsd_A)
f.close()

with open('ln01_tm_ppm_6snd_wyf_rep02_loading_occupancy_907_122722.npy', 'wb') as f:
     np.save(f, loading_site_track_01_A)
f.close()

with open('ln01_tm_ppm_6snd_wyf_rep02_xray_occupancy_907_122722.npy', 'wb') as f:
     np.save(f, xray_site_track_01_A)
f.close()


file_out = 'ln01_tm_ppm_6snd_wyf_rep02_phos_rmsd_908_122722.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_phos_rmsd_B)
f.close()

file_out = 'ln01_tm_ppm_6snd_wyf_rep02_chol_rmsd_908_122722.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_chol_rmsd_B)
f.close()

with open('ln01_tm_ppm_6snd_wyf_rep02_loading_occupancy_908_122722.npy', 'wb') as f:
    np.save(f, loading_site_track_01_B)
f.close()

with open('ln01_tm_ppm_6snd_wyf_rep02_xray_occupancy_908_122722.npy', 'wb') as f:
     np.save(f, xray_site_track_01_B)
f.close()


# In[51]:


wd="/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_TM_ppm_6snd_wyf_rep03/gromacs/"
pdb_fp = wd+"500ns_all.pdb"
dcd_fp = wd+"analysis.dcd"
input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

#initialize lists to store relevant values 
xray_site_track_01_A=[]
loading_site_track_01_A=[]

xray_site_track_01_B=[]
loading_site_track_01_B=[]


frame2xray_phos_rmsd_A=[]
frame2xray_chol_rmsd_A=[]

# frame2xray_phos_rmsd_B=[]
# frame2xray_chol_rmsd_B=[]


for i, frame in enumerate(dcd):

    #select protein 
    frame_fab = frame.getAtoms().select('protein and chain A and name CA or protein and chain B and name CA')

    frame_fab_trim = frame_fab[0:212]+frame_fab[213:355]+frame_fab[361:442]
    frame_crdl1 = frame_fab_trim[23:35] 

    #aln on CDR loop
    superpose(frame_crdl1, xray_cdrl1)
    
    
    #lipid to track  - should bine xray site in this simulation 
    lipidA = frame.getAtoms().select('resnum 920 and resname POPC')
    
    #RMSD - A 
    lipidA_choline = getCoords(lipidA.select('name N'))[0]
    lipidA_phos = getCoords(lipidA.select('name P'))[0]

    phos_rmsd = calcRMS(lipidA_phos-xray_phos_coords)
    chol_rmsd = calcRMS(lipidA_choline-xray_chol_coords)
    
    frame2xray_phos_rmsd_A.append(phos_rmsd)
    frame2xray_chol_rmsd_A.append(chol_rmsd)

    #site occupancy - A 
    lipidA_scores = lipid_site_analysis_ln01_tm(frame, lipidA)
    load_A, xray_A  = codify_scores(lipidA_scores, phos_rmsd, chol_rmsd) 

    xray_site_track_01_A.append(xray_A)
    loading_site_track_01_A.append(load_A)

#     #lipid to track -B should bind MD site in this simluation 
#     lipidB = frame.getAtoms().select('resnum 908 and resname POPC')
    
#     #RMSD 
#     lipidB_choline = getCoords(lipidB.select('name N'))[0]
#     lipidB_phos = getCoords(lipidB.select('name P'))[0]

#     phos_rmsd = calcRMS(lipidB_phos-xray_phos_coords)
#     chol_rmsd = calcRMS(lipidB_choline-xray_chol_coords)
    
#     frame2xray_phos_rmsd_B.append(phos_rmsd)
#     frame2xray_chol_rmsd_B.append(chol_rmsd)

#     #site occupancy - A 
#     lipidB_scores = lipid_site_analysis_ln01_tm(frame, lipidB)
#     load_B, xray_B  = codify_scores(lipidB_scores, phos_rmsd, chol_rmsd) 
    
#     xray_site_track_01_B.append(xray_B)
#     loading_site_track_01_B.append(load_B)
    #print(i)


# In[52]:


file_out = 'ln01_tm_ppm_6snd_wyf_rep03_phos_rmsd_920_122722.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_phos_rmsd_A)
f.close()

file_out = 'ln01_tm_ppm_6snd_wyf_rep03_chol_rmsd_920_122722.npy'
with open(file_out, 'wb') as f:
    np.save(f, frame2xray_chol_rmsd_A)
f.close()

with open('ln01_tm_ppm_6snd_wyf_rep03_loading_occupancy_920_122722.npy', 'wb') as f:
     np.save(f, loading_site_track_01_A)
f.close()

with open('ln01_tm_ppm_6snd_wyf_rep03_xray_occupancy_920_122722.npy', 'wb') as f:
     np.save(f, xray_site_track_01_A)
f.close()



# In[4]:



#LN01 
# reigons in xray that headgroup inter interaction 
# CDRH3: 93-102
# CDRL1: 27-32
# CDRL2: 50-52 
    
#caclulate distances of loop resdiues to phosphate for crystal strucutre 
crys_6snd = parsePDB("/Users/cmaillie/Dropbox (Scripps Research)/manuscript/pdbs/6snd.pdb")


xray_ln01_cdrh3_loop = crys_6snd.select('resnum 93 94 95 96 97 98 99 100 101 102 and chain H')
xray_ln01_cdrh3_loop_bb = xray_ln01_cdrh3_loop.select('name CA CB N ')


xray_ln01_cdrl1 = crys_6snd.select(  "resnum  27 28 29 30 31 32 and chain L") 
xray_ln01_cdrl1_bb = xray_ln01_cdrl1.select('name CA CB N ')

xray_ln01_cdrl2 = crys_6snd.select(  "resnum  50 51 52 and chain L") 
xray_ln01_cdrl2_bb = xray_ln01_cdrl2.select('name CA CB N ')

xray_aln_region =  crys_6snd.select(  "resnum  23 24 25 26 33 34 35 36 and chain L") +  crys_6snd.select(  "resnum  89 90 91 92 103 104 105 106 and chain H")
xray_aln_region_bb = xray_aln_region.select('name CA CB N ')

for i in xray_aln_region_bb.select('name CA'):
    print(i.getResname())


# In[5]:


#find average positions of choline bound for LN01 suystems 

#for each frame, select loop and store coordinates 
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_ppm_6snd_wyf_rep01/gromacs/'
pdb_fp = wd+'500ns_all.pdb'
dcd_fp = wd+'analysis.dcd'

input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()


simluation_chol_coords = []



for i, frame in enumerate(dcd):
    #select protein 
    frame_fab = frame.getAtoms().select("protein not resname TIP3")
    #select loop & loop+flanking residues for each frame 

    md_aln_region =  frame_fab.select(  "resnum  23 24 25 26 33 34 35 36 and chain A") +  input_pdb.select(  "resnum  89 90 91 92 103 104 105 106 and chain B")
    md_aln_region_bb = md_aln_region.select('name CA CB N ') 


    #align strucutres on flanking framework regions 
    superpose(md_aln_region_bb, xray_aln_region_bb) 

    #save coordinates of loop
    choline_coords = frame.getAtoms().select('resnum 398 and name N').getCoords()[0]
    simluation_chol_coords.append(choline_coords)

    
    
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_ppm_6snd_wyf_rep02/gromacs/'
pdb_fp = wd+'500ns_all.pdb'
dcd_fp = wd+'analysis.dcd'

input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

for i, frame in enumerate(dcd):
    #select protein 
    frame_fab = frame.getAtoms().select("protein not resname TIP3")
    #select loop & loop+flanking residues for each frame 

    md_aln_region =  frame_fab.select(  "resnum  23 24 25 26 33 34 35 36 and chain A") +  input_pdb.select(  "resnum  89 90 91 92 103 104 105 106 and chain B")
    md_aln_region_bb = md_aln_region.select('name CA CB N ') 


    #align strucutres on flanking framework regions 
    superpose(md_aln_region_bb, xray_aln_region_bb) 

    #save coordinates of loop
    choline_coords = frame.getAtoms().select('resnum 377 and name N').getCoords()[0]
    simluation_chol_coords.append(choline_coords)
    
    
    
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_ppm_6snd_wyf_rep03/gromacs/'
pdb_fp = wd+'500ns_all.pdb'
dcd_fp = wd+'analysis.dcd'

input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

for i, frame in enumerate(dcd):
    #select protein 
    frame_fab = frame.getAtoms().select("protein not resname TIP3")
    #select loop & loop+flanking residues for each frame 

    md_aln_region =  frame_fab.select(  "resnum  23 24 25 26 33 34 35 36 and chain A") +  input_pdb.select(  "resnum  89 90 91 92 103 104 105 106 and chain B")
    md_aln_region_bb = md_aln_region.select('name CA CB N ') 


    #align strucutres on flanking framework regions 
    superpose(md_aln_region_bb, xray_aln_region_bb) 

    #save coordinates of loop
    choline_coords = frame.getAtoms().select('resnum 384 and name N').getCoords()[0]
    simluation_chol_coords.append(choline_coords)

    


# In[8]:


#selcet coordinates for subset of bound time 
#split by XRAY SITE BOUDN TIME: 
with open('ln01_ppm_6snd_wyf_rep01_loading_occupancy_398_122222.npy', 'rb') as f:
    ln01_ppm_6snd_wyf_rep01_loading_occupancy_398_122222 =  np.load(f) 
f.close()

with open('ln01_ppm_6snd_wyf_rep01_loading_occupancy_413_122222.npy', 'rb') as f:
     ln01_ppm_6snd_wyf_rep01_loading_occupancy_413_122222= np.load(f)
f.close()

with open('ln01_ppm_6snd_wyf_rep02_loading_occupancy_377_122722.npy', 'rb') as f:
     ln01_ppm_6snd_wyf_rep02_loading_occupancy_377_122722 = np.load(f)
f.close()

with open('ln01_ppm_6snd_wyf_rep02_loading_occupancy_385_122722.npy', 'rb') as f:
     ln01_ppm_6snd_wyf_rep02_loading_occupancy_385_122722= np.load(f)
f.close()

with open('ln01_ppm_6snd_wyf_rep03_loading_occupancy_375_122722.npy', 'rb') as f:
     ln01_ppm_6snd_wyf_rep03_loading_occupancy_375_122722 = np.load(f) 
f.close()


with open('ln01_ppm_6snd_wyf_rep03_loading_occupancy_384_122722.npy', 'rb') as f:
     ln01_ppm_6snd_wyf_rep03_loading_occupancy_384_122722 = np.load(f) 
f.close()


choline_bound_coords = [] 
choline_ubbound_coords = [] 

counter = 0 
for i in range(len(ln01_ppm_6snd_wyf_rep01_loading_occupancy_398_122222)):
    if ln01_ppm_6snd_wyf_rep01_loading_occupancy_398_122222[i]==1:
        choline_bound_coords.append(simluation_chol_coords[counter])
    
    else:
        choline_ubbound_coords.append(simluation_chol_coords[counter])
    counter+=1
        
for i in range(len(ln01_ppm_6snd_wyf_rep02_loading_occupancy_377_122722)):
    if ln01_ppm_6snd_wyf_rep02_loading_occupancy_377_122722[i]==1:
        choline_bound_coords.append(simluation_chol_coords[counter])
    
    else:
        choline_ubbound_coords.append(simluation_chol_coords[counter])
    counter+=1
        
for i in range(len(ln01_ppm_6snd_wyf_rep03_loading_occupancy_384_122722)):
    if ln01_ppm_6snd_wyf_rep03_loading_occupancy_384_122722[i]==1:
        choline_bound_coords.append(simluation_chol_coords[counter])
    
    else:
        choline_ubbound_coords.append(simluation_chol_coords[counter])
    counter+=1
        
        
avg_chol_coords = np.mean(choline_bound_coords, axis=0)
print(avg_chol_coords)


# In[11]:


#RMSD of chol position vs average positions of choline 

#for each frame, select loop and store coordinates 
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_ppm_6snd_wyf_rep01/gromacs/'
pdb_fp = wd+'500ns_all.pdb'
dcd_fp = wd+'analysis.dcd'

input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()


chol_rmsf_ln01_rep01 = []



for i, frame in enumerate(dcd):
    #select protein 
    frame_fab = frame.getAtoms().select("protein not resname TIP3")
    #select loop & loop+flanking residues for each frame 

    md_aln_region =  frame_fab.select(  "resnum  23 24 25 26 33 34 35 36 and chain A") +  input_pdb.select(  "resnum  89 90 91 92 103 104 105 106 and chain B")
    md_aln_region_bb = md_aln_region.select('name CA CB N ') 


    #align strucutres on flanking framework regions 
    superpose(md_aln_region_bb, xray_aln_region_bb) 

    #save coordinates of loop
    choline_coords = frame.getAtoms().select('resnum 398 and name N').getCoords()[0]
    #simluation_chol_coords.append(simluation_chol_coords)
    chol_RMSD = calcRMS(choline_coords - avg_chol_coords)
    chol_rmsf_ln01_rep01.append(chol_RMSD)
    
    
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_ppm_6snd_wyf_rep02/gromacs/'
pdb_fp = wd+'500ns_all.pdb'
dcd_fp = wd+'analysis.dcd'

input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()
chol_rmsf_ln01_rep02=[]
for i, frame in enumerate(dcd):
    #select protein 
    frame_fab = frame.getAtoms().select("protein not resname TIP3")
    #select loop & loop+flanking residues for each frame 

    md_aln_region =  frame_fab.select(  "resnum  23 24 25 26 33 34 35 36 and chain A") +  input_pdb.select(  "resnum  89 90 91 92 103 104 105 106 and chain B")
    md_aln_region_bb = md_aln_region.select('name CA CB N ') 


    #align strucutres on flanking framework regions 
    superpose(md_aln_region_bb, xray_aln_region_bb) 

    #save coordinates of loop
    choline_coords = frame.getAtoms().select('resnum 377 and name N').getCoords()[0]
    chol_RMSD = calcRMS(choline_coords - avg_chol_coords)
    chol_rmsf_ln01_rep02.append(chol_RMSD)
    
    
    
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_ppm_6snd_wyf_rep03/gromacs/'
pdb_fp = wd+'500ns_all.pdb'
dcd_fp = wd+'analysis.dcd'

input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()
chol_rmsf_ln01_rep03=[]
for i, frame in enumerate(dcd):
    #select protein 
    frame_fab = frame.getAtoms().select("protein not resname TIP3")
    #select loop & loop+flanking residues for each frame 

    md_aln_region =  frame_fab.select(  "resnum  23 24 25 26 33 34 35 36 and chain A") +  input_pdb.select(  "resnum  89 90 91 92 103 104 105 106 and chain B")
    md_aln_region_bb = md_aln_region.select('name CA CB N ') 


    #align strucutres on flanking framework regions 
    superpose(md_aln_region_bb, xray_aln_region_bb) 

    #save coordinates of loop
    choline_coords = frame.getAtoms().select('resnum 384 and name N').getCoords()[0]
    chol_RMSD = calcRMS(choline_coords - avg_chol_coords)
    chol_rmsf_ln01_rep03.append(chol_RMSD)
    


# In[10]:


def calcRMS(x, axis=None):
    return np.sqrt(np.nanmean(x**2, axis=axis))


# In[17]:


file_out = 'ln01_ppm_6snd_wyf_rep01_chol_RMSF_398.npy'
with open(file_out, 'wb') as f:
    np.save(f, np.array(chol_rmsf_ln01_rep01))
f.close()


file_out = 'ln01_ppm_6snd_wyf_rep02_chol_RMSF_377.npy'
with open(file_out, 'wb') as f:
    np.save(f, np.array(chol_rmsf_ln01_rep02))
f.close()

file_out = 'ln01_ppm_6snd_wyf_rep03_chol_RMSF_384.npy'
with open(file_out, 'wb') as f:
    np.save(f, np.array(chol_rmsf_ln01_rep03))
f.close()


# In[18]:


#find average positions of choline bound for LN01 suystems 

#for each frame, select loop and store coordinates 
wd="/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/LN01_TM-MPER/"
pdb_fp = wd+'500ns_all.pdb'
dcd_fp = wd+'analysis.dcd'

input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()


simluation_chol_coords = []



for i, frame in enumerate(dcd):
    #select protein 
    frame_fab = frame.getAtoms().select("protein not resname TIP3")
    #select loop & loop+flanking residues for each frame 

    md_aln_region =  frame_fab.select(  "resnum  23 24 25 26 33 34 35 36 and chain A") +  input_pdb.select(  "resnum  89 90 91 92 103 104 105 106 and chain B")
    md_aln_region_bb = md_aln_region.select('name CA CB N ') 


    #align strucutres on flanking framework regions 
    superpose(md_aln_region_bb, xray_aln_region_bb) 

    #save coordinates of loop
    choline_coords = frame.getAtoms().select('resnum 1019 and name N').getCoords()[0]
    simluation_chol_coords.append(choline_coords)

    
    
wd="/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_TM_ppm_6snd_wyf_rep02/gromacs/"
pdb_fp = wd+'500ns_all.pdb'
dcd_fp = wd+'analysis.dcd'

input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

for i, frame in enumerate(dcd):
    #select protein 
    frame_fab = frame.getAtoms().select("protein not resname TIP3")
    #select loop & loop+flanking residues for each frame 

    md_aln_region =  frame_fab.select(  "resnum  23 24 25 26 33 34 35 36 and chain A") +  input_pdb.select(  "resnum  89 90 91 92 103 104 105 106 and chain B")
    md_aln_region_bb = md_aln_region.select('name CA CB N ') 


    #align strucutres on flanking framework regions 
    superpose(md_aln_region_bb, xray_aln_region_bb) 

    #save coordinates of loop
    choline_coords = frame.getAtoms().select('resnum 908 and name N').getCoords()[0]
    simluation_chol_coords.append(choline_coords)
    
    
    
wd="/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_TM_ppm_6snd_wyf_rep03/gromacs/"
pdb_fp = wd+'500ns_all.pdb'
dcd_fp = wd+'analysis.dcd'

input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

for i, frame in enumerate(dcd):
    #select protein 
    frame_fab = frame.getAtoms().select("protein not resname TIP3")
    #select loop & loop+flanking residues for each frame 

    md_aln_region =  frame_fab.select(  "resnum  23 24 25 26 33 34 35 36 and chain A") +  input_pdb.select(  "resnum  89 90 91 92 103 104 105 106 and chain B")
    md_aln_region_bb = md_aln_region.select('name CA CB N ') 


    #align strucutres on flanking framework regions 
    superpose(md_aln_region_bb, xray_aln_region_bb) 

    #save coordinates of loop
    choline_coords = frame.getAtoms().select('resnum 920 and name N').getCoords()[0]
    simluation_chol_coords.append(choline_coords)

    


# In[19]:


#selcet coordinates for subset of bound time 
#split by XRAY SITE BOUDN TIME: 
with open('ln01_tm_ppm_6snd_wyf_rep02_loading_occupancy_A.npy', 'rb') as f:
    ln01_tm_wyf_02_loading_occupancy_A =np.load(f)
f.close()

with open('ln01_tm_ppm_6snd_wyf_rep03_loading_occupancy_920_122722.npy', 'rb') as f:
    ln01_tm_ppm_6snd_wyf_rep03_loading_occupancy_A =np.load(f)
f.close()

with open('ln01_tm_ppm_6snd_wyf_MM_loading_occupancy_A.npy', 'rb') as f:
    ln01_tm_ppm_6snd_wyf_MM_loading_occupancy_A =np.load(f)
f.close() 

choline_bound_coords = [] 
choline_ubbound_coords = [] 

counter = 0 
for i in range(len(ln01_tm_ppm_6snd_wyf_MM_loading_occupancy_A)):
    if ln01_tm_ppm_6snd_wyf_MM_loading_occupancy_A[i]==1:
        choline_bound_coords.append(simluation_chol_coords[counter])
    
    else:
        choline_ubbound_coords.append(simluation_chol_coords[counter])
    counter+=1
        
for i in range(len(ln01_tm_wyf_02_loading_occupancy_A)):
    if ln01_tm_wyf_02_loading_occupancy_A[i]==1:
        choline_bound_coords.append(simluation_chol_coords[counter])
    
    else:
        choline_ubbound_coords.append(simluation_chol_coords[counter])
    counter+=1
        
for i in range(len(ln01_tm_ppm_6snd_wyf_rep03_loading_occupancy_A)):
    if ln01_tm_ppm_6snd_wyf_rep03_loading_occupancy_A[i]==1:
        choline_bound_coords.append(simluation_chol_coords[counter])
    
    else:
        choline_ubbound_coords.append(simluation_chol_coords[counter])
    counter+=1
        
        
avg_chol_coords = np.mean(choline_bound_coords, axis=0)
print(avg_chol_coords)


# In[20]:


#find average positions of choline bound for LN01 suystems 

#for each frame, select loop and store coordinates 
wd="/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/LN01_TM-MPER/"
pdb_fp = wd+'500ns_all.pdb'
dcd_fp = wd+'analysis.dcd'

input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()




chol_rmsf_ln01_tm_repMM=[]
for i, frame in enumerate(dcd):
    #select protein 
    frame_fab = frame.getAtoms().select("protein not resname TIP3")
    #select loop & loop+flanking residues for each frame 

    md_aln_region =  frame_fab.select(  "resnum  23 24 25 26 33 34 35 36 and chain A") +  input_pdb.select(  "resnum  89 90 91 92 103 104 105 106 and chain B")
    md_aln_region_bb = md_aln_region.select('name CA CB N ') 


    #align strucutres on flanking framework regions 
    superpose(md_aln_region_bb, xray_aln_region_bb) 

    #save coordinates of loop
    choline_coords = frame.getAtoms().select('resnum 1019 and name N').getCoords()[0]
    chol_RMSD = calcRMS(choline_coords - avg_chol_coords)
    chol_rmsf_ln01_tm_repMM.append(chol_RMSD)

    
    
wd="/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_TM_ppm_6snd_wyf_rep02/gromacs/"
pdb_fp = wd+'500ns_all.pdb'
dcd_fp = wd+'analysis.dcd'

input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()
chol_rmsf_ln01_tm_rep02=[]
for i, frame in enumerate(dcd):
    #select protein 
    frame_fab = frame.getAtoms().select("protein not resname TIP3")
    #select loop & loop+flanking residues for each frame 

    md_aln_region =  frame_fab.select(  "resnum  23 24 25 26 33 34 35 36 and chain A") +  input_pdb.select(  "resnum  89 90 91 92 103 104 105 106 and chain B")
    md_aln_region_bb = md_aln_region.select('name CA CB N ') 


    #align strucutres on flanking framework regions 
    superpose(md_aln_region_bb, xray_aln_region_bb) 

    #save coordinates of loop
    choline_coords = frame.getAtoms().select('resnum 908 and name N').getCoords()[0]
    chol_RMSD = calcRMS(choline_coords - avg_chol_coords)
    chol_rmsf_ln01_tm_rep02.append(chol_RMSD)

    
    
    
wd="/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_TM_ppm_6snd_wyf_rep03/gromacs/"
pdb_fp = wd+'500ns_all.pdb'
dcd_fp = wd+'analysis.dcd'

input_pdb = parsePDB(pdb_fp)
dcd = DCDFile(dcd_fp)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()
chol_rmsf_ln01_tm_rep03=[]
for i, frame in enumerate(dcd):
    #select protein 
    frame_fab = frame.getAtoms().select("protein not resname TIP3")
    #select loop & loop+flanking residues for each frame 

    md_aln_region =  frame_fab.select(  "resnum  23 24 25 26 33 34 35 36 and chain A") +  input_pdb.select(  "resnum  89 90 91 92 103 104 105 106 and chain B")
    md_aln_region_bb = md_aln_region.select('name CA CB N ') 


    #align strucutres on flanking framework regions 
    superpose(md_aln_region_bb, xray_aln_region_bb) 

    #save coordinates of loop
    choline_coords = frame.getAtoms().select('resnum 920 and name N').getCoords()[0]
    chol_RMSD = calcRMS(choline_coords - avg_chol_coords)
    chol_rmsf_ln01_tm_rep03.append(chol_RMSD)

    


# In[22]:


plt.plot(chol_rmsf_ln01_tm_repMM)
plt.show()
plt.plot(chol_rmsf_ln01_tm_rep02)
plt.show()
plt.plot(chol_rmsf_ln01_tm_rep03)
plt.show()


# In[23]:


file_out = 'ln01_tm_repMM_chol_RMSF_1019.npy'
with open(file_out, 'wb') as f:
    np.save(f, np.array(chol_rmsf_ln01_tm_repMM))
f.close()


file_out = 'ln01_tm_rep02_chol_RMSF_908.npy'
with open(file_out, 'wb') as f:
    np.save(f, np.array(chol_rmsf_ln01_tm_rep02))
f.close()

file_out = 'ln01_tm_rep03_chol_RMSF_920.npy'
with open(file_out, 'wb') as f:
    np.save(f, np.array(chol_rmsf_ln01_tm_rep03))
f.close()

