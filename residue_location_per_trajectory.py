#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
plt.rcParams['font.sans-serif'] = "Arial"
import pickle


# In[4]:


#from MM 
def planeFit(points):
	points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
	assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
	ctr = points.mean(axis=1)
	x = points - ctr[:,np.newaxis]
	M = np.dot(x, x.T) # Could also use np.cov(x) here.
	return ctr, svd(M)[0][:,-1]


#MM
def VectorAlign(a, b):
    u=np.cross(a,b)
    s = LA.norm(u)
    c=np.dot(a,b)
    skw=skew(u)
    return np.identity(3) + skw + (1/(1+c)) * np.dot(skw, skw)


#MM
def skew(v):
    if len(v) == 4: 
        v=v[:3]/v[3]
    skv = np.roll(np.roll(np.diag(v.flatten()), 1, 1), -1, 0)
    return skv - skv.T


# In[3]:


ln01 = parsePDB('/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_ppm_6snd_wyf_rep02/gromacs/500ns_all.pdb')
prot = ln01.select("name CA")
ln01_seq = []
for i in range(len(prot)):
    ln01_seq.append(prot[i].getResname())


# In[4]:



ab_4e10 = parsePDB('/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_ppm_rep/final_analysis_input.pdb')
prot = ab_4e10.select("name CA")
ab_4e10_seq = []
for i in range(len(prot)):
    ab_4e10_seq.append(prot[i].getResname())


# In[5]:



ab_pgzl1 = parsePDB('/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_ppm/final_analysis_input.pdb')
prot = ab_pgzl1.select("name CA")
ab_pgzl1_seq = []
for i in range(len(prot)):
    ab_pgzl1_seq.append(prot[i].getResname())


# In[ ]:





# In[6]:



  
def plot_interaction_profile_ln01(acyl_loc, glycerol_loc, po4_loc, l_chain_residues, prefix): 
  fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 8), gridspec_kw={'height_ratios': [.5, 6]})
  plt.subplots_adjust(hspace=0.01)
  #x = np.arange(0, len(l_chain_residues),1)
  xticks = [i for i in range(0,len(l_chain_residues))]
  #add cdr loop annotation 
  l_chain_annotate = [] 
  for i in range(len(l_chain_residues)):
      #add cdrl1 loop 
      if i in range(26, 31):
          l_chain_annotate.append(1)
      elif i in range(49,51):
          l_chain_annotate.append(1)
      elif i in range(88,93):
          l_chain_annotate.append(1)
      #add H chain loops 
      elif i in range(236, 245):
          l_chain_annotate.append(2)
      elif i in range(263,269):
          l_chain_annotate.append(2)
      elif i in range(310,327):
          l_chain_annotate.append(2)        
      else:
          l_chain_annotate.append(0)
  #light
  ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==1, 
                   alpha=1, color='#9BD47D') #DBBC8F
  #heavy
  ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==2, 
                  alpha=1, color='#385931') #724E3A
  #frameowrk - grey
  ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==0, 
                  alpha=.5, color='#BFBFBF')

  ax1.axhline(y=-10, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
  ax1.axhline(y=0, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
  #mark antibody chain start/stop 
  ax1.axvline(x=213, color='#000000', linestyle='-', ymin=0, linewidth=1.5)
  ax1.axvline(x=443, color='#000000', linestyle='-', ymin=0, linewidth=1.5)
  ax1.spines["right"].set_visible(False)
  ax1.spines["top"].set_visible(False)
  ax1.spines["bottom"].set_visible(False)
  ax1.set_xticks([])
  ax1.set_yticks([])

  width=1
  ax2.bar(xticks, acyl_loc, width, label='Acyl', color='blue')
  ax2.bar(xticks, glycerol_loc, width, bottom=acyl_loc, label='Glycerol', color='red')
  ax2.bar(xticks, po4_loc, width, bottom=glycerol_loc+acyl_loc, label='PO4', color='orange')
  
 
  ax2.set_xticks([]) #xticks
  ax2.set_xticklabels([])#l_chain_residues
  ax2.set_yticks([0, 0.50,  1]) 
  ax2.set_yticklabels(['0',  '0.5',  '1.0'], fontsize=50) 
  #ax2.set_yticklabels(['0', '1', '2', '3', '4'])
  ax2.tick_params(axis='x', labelsize=50)
  ax2.tick_params(axis='y', labelsize=50)#, fontsize=16
  #ax2.set_ylabel(r"Time ($\rm \mu s$)", fontsize=54)
  #ax2.set_xlabel('Residue', fontsize54)
  ax2.spines["right"].set_visible(False)
  ax2.spines["top"].set_visible(False)
  
  
  ax1.set_xlim(0, len(xticks))
  ax2.set_xlim(0, len(xticks))
  
  fig_name = prefix + '_interaction_profile.png' 
  plt.savefig(fig_name, transparent=True, bbox_inches="tight")

  plt.show()
  return "Made figure: "#, prefix



# In[15]:



  
def plot_interaction_profile_ln01_ticks(acyl_loc, glycerol_loc, po4_loc, l_chain_residues, prefix): 
  fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 8), gridspec_kw={'height_ratios': [.5, 6]})
  plt.subplots_adjust(hspace=0.01)
  #x = np.arange(0, len(l_chain_residues),1)
  xticks = [i for i in range(0,len(l_chain_residues))]
  #add cdr loop annotation 
  l_chain_annotate = [] 
  for i in range(len(l_chain_residues)):
      #add cdrl1 loop 
      if i in range(26, 31):
          l_chain_annotate.append(1)
      elif i in range(49,51):
          l_chain_annotate.append(1)
      elif i in range(88,93):
          l_chain_annotate.append(1)
      #add H chain loops 
      elif i in range(236, 245):
          l_chain_annotate.append(2)
      elif i in range(263,269):
          l_chain_annotate.append(2)
      elif i in range(310,327):
          l_chain_annotate.append(2)        
      else:
          l_chain_annotate.append(0)
  #light
  ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==1, 
                   alpha=1, color='#9BD47D') 
  #heavy
  ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==2, 
                  alpha=1, color='#385931') 
  #frameowrk - grey
  ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==0, 
                  alpha=.5, color='#BFBFBF')

  ax1.axhline(y=-10, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
  ax1.axhline(y=0, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
  #mark antibody chain start/stop 
  ax1.axvline(x=213, color='#000000', linestyle='-', ymin=0, linewidth=1.5)
  ax1.axvline(x=443, color='#000000', linestyle='-', ymin=0, linewidth=1.5)
  ax1.spines["right"].set_visible(False)
  ax1.spines["top"].set_visible(False)
  ax1.spines["bottom"].set_visible(False)
  ax1.set_xticks([])
  ax1.set_yticks([])

  width=1
  ax2.bar(xticks, acyl_loc, width, label='Acyl', color='blue')
  ax2.bar(xticks, glycerol_loc, width, bottom=acyl_loc, label='Glycerol', color='red')
  ax2.bar(xticks, po4_loc, width, bottom=glycerol_loc+acyl_loc, label='PO4', color='orange')
  
  ax2.set_xticks(np.arange(min(xticks), max(xticks)+1, 1.0))
  for i in xticks:
      ax2.axvline(x=i, color='#000000', linestyle='-', ymin=0, linewidth=0.05)
  #ax2.set_xticks() #xticks
  ax2.set_xticklabels([])#l_chain_residues
  ax2.set_yticks([0, 0.50,  1]) 
  ax2.set_yticklabels(['0',  '0.5',  '1.0'], fontsize=50) 
  #ax2.set_yticklabels(['0', '1', '2', '3', '4'])
  ax2.tick_params(axis='x', labelsize=50)
  ax2.tick_params(axis='y', labelsize=50)#, fontsize=16
  #ax2.set_ylabel(r"Time ($\rm \mu s$)", fontsize=54)
  #ax2.set_xlabel('Residue', fontsize54)
  ax2.spines["right"].set_visible(False)
  ax2.spines["top"].set_visible(False)
  
  
  ax1.set_xlim(0, len(xticks))
  ax2.set_xlim(0, len(xticks))
  
  fig_name = prefix + '_interaction_profile_ticks.png' 
  plt.savefig(fig_name, transparent=True, bbox_inches="tight")

  plt.show()
  return "Made figure: "#, prefix



# In[5]:


def dist2Plane(points, planeParams):
	cen, norm = planeParams[0], planeParams[1]
	d = np.dot(norm,cen)
	return ( np.dot(norm,points) - d )

def split_seq2list(seq_string):
    return [residue for residue in seq_string ]  


# In[7]:


def calc_res_depths(res_depths_list, input_pdb_fp, dcd_fp):
    input_pdb = parsePDB(input_pdb_fp)
    dcd = DCDFile(dcd_fp)
    dcd.setCoords(input_pdb)
    dcd.link(input_pdb)
    dcd.reset()
    fab_selection_str = 'protein'
    mem_selection_str = 'resname POPC POPS PSM POPE CHOL'#POPC POPA CHOL
    for i, frame in enumerate(dcd):
        moveAtoms(frame.getAtoms(), to=np.zeros(3))

        ft_vect = []
        ft_vect_short = [] 
        #set mass for atoms
        frame.getAtoms().select('name CA or name CB').setMasses(12)
        frame.getAtoms().select('name N').setMasses(14)
        #define fab and 
        frame_fab = frame.getAtoms().select( fab_selection_str )
        frame_fab_bb = frame.getAtoms().select( 'name CA' )[0:442]
        frame_fab = frame.getAtoms().select( fab_selection_str )
        frame_fab_res = frame.getAtoms().select( 'name CA' )[0:442]
        #define psudo central axis for alignemnt of frames 
        #axis will go through center of fab and center of membrane 
        #need this to set frames in same orientation & alieviate wrapping alignment issues ; 
        #otherwise membrane selection will not be accurate 
        pseudo_fab_cen = calcCenter(frame_fab_bb.select('resnum 41 or resnum 273'))
        membrane_cen = calcCenter(frame.getAtoms().select(mem_selection_str))
        psuedo_central_ax = np.array(pseudo_fab_cen-membrane_cen)
        
        #must normalize axis vector before transforming 
        psuedo_central_ax_norm = psuedo_central_ax / np.linalg.norm(psuedo_central_ax)
        #if psuedo_central_ax_norm[2]<1:
            
        #print("NEGATIVE: ", psuedo_central_ax_norm)
        #if psuedo_central_ax_norm[2]<1:
        rotation = VectorAlign(psuedo_central_ax_norm, np.array([0, 0, 1]))
        #else: 
            #rotation = VectorAlign(psuedo_central_ax_norm, np.array([0, 0, -1]))
        transformCentralAx = Transformation(rotation, np.zeros(3))
        #apply transofrmation of plane to Z axis to entire system 
        applyTransformation(transformCentralAx, frame.getAtoms())
        
        
        #select membrane after aligning frames 
        frame_mem = frame.getAtoms().select( mem_selection_str )

        #DEFINE MEMBRANE LAYERS - bilayer midpoint, top PO4 plane, bottom PO4 plane 
        avg_bilayer_mp = (int(sum(list(frame_mem.getCoords()[:,2]))/len(frame_mem.getResnames())))
        mem_top_sel_str  = 'resname POPC name P and z > '+ str(avg_bilayer_mp)
        mem_bot_sel_str  = 'resname POPC name P and z < '+ str(avg_bilayer_mp)#and z < '+ str(avg_bilayer_mp)

        bot_leaf = frame.getAtoms().select(mem_bot_sel_str)
        bot_leaf_points = bot_leaf.getCoords()
        bot_leaf_plane = planeFit(np.transpose(bot_leaf_points))
        bot_phos_layer_sel_str  = 'resname POPC name P and z < ' +str(avg_bilayer_mp)
        phos = frame.getAtoms().select(bot_phos_layer_sel_str)
        phos_points = phos.getCoords()
        phos_z = phos_points[:,2]
        #print("BOT: ", len(phos_z))
        top_leaf = frame.getAtoms().select(mem_top_sel_str)
        top_leaf_points = top_leaf.getCoords()
        top_leaf_plane = planeFit(np.transpose(top_leaf_points))
        top_phos_layer_sel_str  = 'resname POPC name P and z > ' +str(avg_bilayer_mp)
        phos = frame.getAtoms().select(top_phos_layer_sel_str)
        phos_points = phos.getCoords()
        phos_z = phos_points[:,2]
        #print("TOP: ", len(phos_z))
        
        
        avg_top_phos_z = sum(phos_z)/len(phos_z)
        top_phos_layer_dist2bilayer_cen = avg_top_phos_z - avg_bilayer_mp
    
        bot_leaf = frame.getAtoms().select(mem_bot_sel_str)
        bot_leaf_points = bot_leaf.getCoords()
        bot_leaf_plane = planeFit(np.transpose(bot_leaf_points))
        bot_phos_layer_sel_str  = 'resname POPC name P and z < ' +str(avg_bilayer_mp)
        phos = frame.getAtoms().select(bot_phos_layer_sel_str)
        phos_points = phos.getCoords()
        phos_z = phos_points[:,2]
        avg_bot_phos_z = sum(phos_z)/len(phos_z)
        bot_phos_layer_dist2bilayer_cen = avg_bot_phos_z - avg_bilayer_mp
    
        #align norm vector of membrane plane to Z axis for approx angle calculations  
        #system oriented so that fab is embeded in top bilayer 
        #print(top_leaf_plane[1])
        if top_leaf_plane[1][2]<0: 
            #print("SWITCHED\n\n")
            top_leaf_plane_FLIPPED = np.array([-top_leaf_plane[1][0], -top_leaf_plane[1][1], -top_leaf_plane[1][2]])
            #print(top_leaf_plane_FLIPPED)
            rotation = VectorAlign(top_leaf_plane_FLIPPED, np.array([0, 0, 1]))
            transformMemNorm = Transformation(rotation, np.zeros(3))
            #apply transofrmation of plane to Z axis to entire system 
            applyTransformation(transformMemNorm, frame.getAtoms())
        else:
            rotation = VectorAlign(top_leaf_plane[1], np.array([0, 0, 1]))
            transformMemNorm = Transformation(rotation, np.zeros(3))
            #apply transofrmation of plane to Z axis to entire system 
            applyTransformation(transformMemNorm, frame.getAtoms())
            
        for j in range(len(frame_fab_res)):
            res = frame_fab_res[j].getCoords()
            #depth to plane 
            if top_leaf_plane[1][2]<0:
                res_depth = dist2Plane(res, top_leaf_plane)*-1 #multiply by negative one to get depth values that make sense
            else:
                res_depth = dist2Plane(res, top_leaf_plane)
            res_loc=0 
            #acyl chain location : depth<-7 
            if res_depth<=-7:
                res_loc = 1
                res_depths_list[j].append(res_loc)
            #glycerol level  -7<depth<-3
            elif res_depth<=-3 and res_depth>-7:
                res_loc = 2
                res_depths_list[j].append(res_loc)
            #phosphate level  -3<depth<3
            elif res_depth<=3 and res_depth>-3:
                res_loc = 3
                res_depths_list[j].append(res_loc)
            #water level  -3<depth<3
            elif res_depth>3:
                res_loc = 4
                res_depths_list[j].append(res_loc)


# In[5]:


aa_resi_loc_4e10_ppm =[[] for _ in range(441)] #number of residues 
calc_res_depths(aa_resi_loc_4e10_ppm,
                '/home/bat-gpu/colleen/phos_interactions/final_traj/4e10_ppm/analysis_input.pdb',
                '/home/bat-gpu/colleen/phos_interactions/final_traj/4e10_ppm/final_md.dcd')    

aa_resi_loc_4e10_p15 =[[] for _ in range(441)] #number of residues 
calc_res_depths(aa_resi_loc_4e10_p15,
                '/home/bat-gpu/colleen/phos_interactions/final_traj/4e10_p15/analysis_input.pdb',
                '/home/bat-gpu/colleen/phos_interactions/final_traj/4e10_p15/final_md.dcd')    

aa_resi_loc_4e10_n15 =[[] for _ in range(441)] #number of residues 
calc_res_depths(aa_resi_loc_4e10_n15,
                '/home/bat-gpu/colleen/phos_interactions/final_traj/4e10_n15/analysis_input.pdb',
                '/home/bat-gpu/colleen/phos_interactions/final_traj/4e10_n15/final_md.dcd')    

aa_resi_loc_4e10_ppm_1us =[[] for _ in range(441)] #number of residues 
calc_res_depths(aa_resi_loc_4e10_ppm_1us,
                '/home/bat-gpu/colleen/phos_interactions/final_traj/4e10_ppm_1us/analysis_input.pdb',
                '/home/bat-gpu/colleen/phos_interactions/final_traj/4e10_ppm_1us/final_md.dcd')    


# In[8]:


#high chol contetnt analysis 
aa_resi_loc_4e10_hivLike =[[] for _ in range(441)] #number of residues 
calc_res_depths(aa_resi_loc_4e10_hivLike,
                '/Users/cmaillie/Dropbox (Scripps Research)/eLife_revisions_experiments/4e10_hivLike/gromacs/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/eLife_revisions_experiments/4e10_hivLike/gromacs/final_analysis_traj.dcd')    


# In[40]:


#high chol contetnt analysis 
aa_resi_loc_pgzl1_hivLike =[[] for _ in range(441)] #number of residues 
calc_res_depths(aa_resi_loc_pgzl1_hivLike,
                '/Users/cmaillie/Dropbox (Scripps Research)/eLife_revisions_experiments/pgzl1_hivLike/gromacs/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/eLife_revisions_experiments/pgzl1_hivLike/gromacs/final_analysis_traj.dcd')    


# In[49]:


#high chol contetnt analysis 
aa_resi_loc_10e8_hivLike =[[] for _ in range(444)] #number of residues 
calc_res_depths(aa_resi_loc_10e8_hivLike,
                '/Users/cmaillie/Dropbox (Scripps Research)/eLife_revisions_experiments/10e8_hivLike/gromacs/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/eLife_revisions_experiments/10e8_hivLike/gromacs/final_analysis_traj.dcd')    


# In[9]:


#store HivLike simulation interactino profiles for plotting
aa_resi_loc_4e10_hivLike_counts =[[] for _ in range(441)] 
for i in range(len(aa_resi_loc_4e10_hivLike)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_4e10_hivLike[i].count(1)
    aa_resi_loc_4e10_hivLike_counts[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_4e10_hivLike[i].count(2)
    aa_resi_loc_4e10_hivLike_counts[i].append(glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_4e10_hivLike[i].count(3)
    aa_resi_loc_4e10_hivLike_counts[i].append(po4_counts)
    #water
    water_counts = aa_resi_loc_4e10_hivLike[i].count(4)
    aa_resi_loc_4e10_hivLike_counts[i].append(water_counts)
#write
with open('4e10_hivLike_AA_interaction_profile.pkl', 'wb') as handle:
    pickle.dump(aa_resi_loc_4e10_hivLike_counts,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)
with open('4e10_hivLike_AA_interaction_profile.pkl', 'rb') as handle:
    interaction_prof_4e10_hivLike = pickle.load(handle)
    
interaction_prof_4e10_hivLike = np.array(interaction_prof_4e10_hivLike)


# In[24]:


aa_resi_loc_4e10_hivLike_counts_light =[] 
aa_resi_loc_4e10_hivLike_counts_heavy =[] 

for i in range(len(interaction_prof_4e10_hivLike)):
    if i<225:
        aa_resi_loc_4e10_hivLike_counts_heavy.append(interaction_prof_4e10_hivLike[i]) #add to heavy chain
        #print(interaction_prof_4e10_hivLike[i])
    else:
        aa_resi_loc_4e10_hivLike_counts_light.append(interaction_prof_4e10_hivLike[i])#add count for light chain 
    #print(i)


# In[30]:


#print(aa_resi_loc_4e10_hivLike_counts_heavy)
print(len(aa_resi_loc_4e10_hivLike_counts_heavy))
print(len(aa_resi_loc_4e10_hivLike_counts_light))

aa_resi_loc_4e10_hivLike_counts_total = []
for i in aa_resi_loc_4e10_hivLike_counts_light:
    aa_resi_loc_4e10_hivLike_counts_total.append(i)
for i in aa_resi_loc_4e10_hivLike_counts_heavy:
    aa_resi_loc_4e10_hivLike_counts_total.append(i)
print(len(aa_resi_loc_4e10_hivLike_counts_total))


#write
with open('4e10_hivLike_AA_interaction_profile_total.pkl', 'wb') as handle:
    pickle.dump(aa_resi_loc_4e10_hivLike_counts_total,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)
with open('4e10_hivLike_AA_interaction_profile_total.pkl', 'rb') as handle:
    interaction_prof_4e10_hivLike_total = pickle.load(handle)
    
interaction_prof_4e10_hivLike_total = np.array(interaction_prof_4e10_hivLike_total)


# In[39]:


#holder residue list - lenght of 4e10wawa
l_chain_pgzl1_germ_seq = 'VLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPAA'
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGECQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'

l_chain_residues = split_seq2list(l_chain_pgzl1_germ_seq)
print(len(l_chain_residues))
plot_interaction_profile(interaction_prof_4e10_hivLike_total[:,0]/2802,
                         interaction_prof_4e10_hivLike_total[:,1]/2802,
                         interaction_prof_4e10_hivLike_total[:,2]/2802,
                         l_chain_residues, '4e10_hivLike')


# In[41]:


#store HivLike simulation interactino profiles for plotting
aa_resi_loc_pgzl1_hivLike_counts =[[] for _ in range(441)] 
for i in range(len(aa_resi_loc_pgzl1_hivLike)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_pgzl1_hivLike[i].count(1)
    aa_resi_loc_pgzl1_hivLike_counts[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_pgzl1_hivLike[i].count(2)
    aa_resi_loc_pgzl1_hivLike_counts[i].append(glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_pgzl1_hivLike[i].count(3)
    aa_resi_loc_pgzl1_hivLike_counts[i].append(po4_counts)
    #water
    water_counts = aa_resi_loc_pgzl1_hivLike[i].count(4)
    aa_resi_loc_pgzl1_hivLike_counts[i].append(water_counts)
#write
with open('pgzl1_hivLike_AA_interaction_profile.pkl', 'wb') as handle:
    pickle.dump(aa_resi_loc_pgzl1_hivLike_counts,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)
with open('pgzl1_hivLike_AA_interaction_profile.pkl', 'rb') as handle:
    interaction_prof_pgzl1_hivLike = pickle.load(handle)
    
interaction_prof_pgzl1_hivLike = np.array(interaction_prof_pgzl1_hivLike)


# In[46]:


print(interaction_prof_pgzl1_hivLike[0])


# In[47]:


chain_pgzl1_seq = 'DVVMTQSPGTLSLSPGERATLSCRASQSVSGGALAWYQQKPGQAPRLLIYDTSSRPTGVPGRFSGSGSGTDFSLTISRLEPEDFAVYYCQQYGTSQSTFGQGTRLEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEEVQLVQSGGEVKRPGSSVTVSCKATGGTFSTLAFNWVRQAPGQGPEWMGGIVPLFSIVNYGQKFQGRLTIRADKSTTTVFLDLSGLTSADTATYYCAREGEGWFGKPLRAFEFWGQGTVITVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'

#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC'
chain_residues = split_seq2list(chain_pgzl1_seq)
print(len(chain_pgzl1_seq))
plot_interaction_profile_pgzl1(interaction_prof_pgzl1_hivLike[:,0]/3836,
                         interaction_prof_pgzl1_hivLike[:,1]/3836,
                         interaction_prof_pgzl1_hivLike[:,2]/3836,
                         chain_pgzl1_seq, 'pgzl1_hivLike')


# In[50]:


#store HivLike simulation interactino profiles for plotting
aa_resi_loc_10e8_hivLike_counts =[[] for _ in range(444)] 
for i in range(len(aa_resi_loc_10e8_hivLike)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_10e8_hivLike[i].count(1)
    aa_resi_loc_10e8_hivLike_counts[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_10e8_hivLike[i].count(2)
    aa_resi_loc_10e8_hivLike_counts[i].append(glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_10e8_hivLike[i].count(3)
    aa_resi_loc_10e8_hivLike_counts[i].append(po4_counts)
    #water
    water_counts = aa_resi_loc_10e8_hivLike[i].count(4)
    aa_resi_loc_10e8_hivLike_counts[i].append(water_counts)
#write
with open('10e8_hivLike_AA_interaction_profile.pkl', 'wb') as handle:
    pickle.dump(aa_resi_loc_10e8_hivLike_counts,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)
with open('10e8_hivLike_AA_interaction_profile.pkl', 'rb') as handle:
    interaction_prof_10e8_hivLike = pickle.load(handle)
    
interaction_prof_10e8_hivLike = np.array(interaction_prof_10e8_hivLike)


# In[55]:


print(interaction_prof_10e8_hivLike[0])


# In[59]:


aa_resi_loc_10e8_hivLike_counts_light =[] 
aa_resi_loc_10e8_hivLike_counts_heavy =[] 

for i in range(len(interaction_prof_10e8_hivLike)):
    if i<232:
        aa_resi_loc_10e8_hivLike_counts_heavy.append(interaction_prof_10e8_hivLike[i]) #add to heavy chain
        #print(interaction_prof_10e8_hivLike[i])
    else:
        aa_resi_loc_10e8_hivLike_counts_light.append(interaction_prof_10e8_hivLike[i])#add count for light chain 
    #print(i)


# In[60]:


#print(aa_resi_loc_10e8_hivLike_counts_heavy)
print(len(aa_resi_loc_10e8_hivLike_counts_heavy))
print(len(aa_resi_loc_10e8_hivLike_counts_light))

aa_resi_loc_10e8_hivLike_counts_total = []
for i in aa_resi_loc_10e8_hivLike_counts_light:
    aa_resi_loc_10e8_hivLike_counts_total.append(i)
for i in aa_resi_loc_10e8_hivLike_counts_heavy:
    aa_resi_loc_10e8_hivLike_counts_total.append(i)
print(len(aa_resi_loc_10e8_hivLike_counts_total))


#write
with open('10e8_hivLike_AA_interaction_profile_total.pkl', 'wb') as handle:
    pickle.dump(aa_resi_loc_10e8_hivLike_counts_total,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)
with open('10e8_hivLike_AA_interaction_profile_total.pkl', 'rb') as handle:
    interaction_prof_10e8_hivLike_total = pickle.load(handle)
    
interaction_prof_10e8_hivLike_total = np.array(interaction_prof_10e8_hivLike_total)


# In[61]:


#get list of residues in l chain 4e10 
chain_10e8_seq = "SYELTQETGVSVALGRTVTITCRGDSLRSHYASWYQKKPGQAPILLFYGKNNRPSGVPDRFSGSASGNRASLTISGAQAEDDAEYYCSSRDKSGSRLSVFGGGTKLTVLSQPKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPVKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTEVQLVESGGGLVKPGGSLRLSCSASGFDFDNAWMTWVRQPPGKGLEWVGRITGPGEGWSVDYAAPVEGRFTISRLNSINFLYLEMNNLRMEDSGLYFCARTGKYYDFWSGYPPGEEYFQDWGRGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPK"

#chain_10e8_seq ="SYELTQETGVSVALGRTVTITCRGDSLRSHYASWYQKKPGQAPILLFYGKNNRPSGVPDRFSGSASGNRASLTISGAQAEDDAEYYCSSRDKSGSRLSVFGGGTKLTVLSQPKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPVKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTECSEVQLVESGGGLVKPGGSLRLSCSASGFDFDNAWMTWVRQPPGKGLEWVGRITGPGEGWSVDYAAPVEGRFTISRLNSINFLYLEMNNLRMEDSGLYFCARTGKYYDFWSGYPPGEEYFQDWGRGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSCDK"


chain_residues = split_seq2list(chain_10e8_seq)
print(len(chain_10e8_seq))
plot_interaction_profile_10e8_v2(interaction_prof_10e8_hivLike_total[:,0]/3495,
                         interaction_prof_10e8_hivLike_total[:,1]/3495,
                         interaction_prof_10e8_hivLike_total[:,2]/3495,
                         chain_10e8_seq, '10e8_hivLike')


# In[8]:


pdb = parsePDB('/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/LN01_OPM_rep1/analysis_input.pdb')
print(len(pdb.select('name CA')))


# In[11]:


aa_resi_loc_lno1_opm_rep1 =[[] for _ in range(443)] #number of residues 
calc_res_depths(aa_resi_loc_lno1_opm_rep1,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/LN01_OPM_rep1/100ns_all.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/LN01_OPM_rep1/analysis.dcd')    


# In[12]:


aa_resi_loc_ln01_6snd_ppm_02 =[[] for _ in range(443)] #number of residues 
calc_res_depths(aa_resi_loc_ln01_6snd_ppm_02,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_6snd_ppm_02/gromacs/1000ns_out.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_6snd_ppm_02/gromacs/analysis.dcd') 


# In[26]:


aa_resi_loc_ln01_6sne_ppm_03=[[] for _ in range(443)] #number of residues 
calc_res_depths(aa_resi_loc_ln01_6sne_ppm_03,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_6sne_ppm_03/gromacs/500ns_all.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_6sne_ppm_03/gromacs/analysis.dcd') 


# In[14]:


aa_resi_loc_ln01_6sne_ppm_04=[[] for _ in range(443)] #number of residues 
calc_res_depths(aa_resi_loc_ln01_6sne_ppm_04,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_6sne_ppm_04/gromacs/1000ns_out.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_6sne_ppm_04/gromacs/analysis.dcd') 


# In[ ]:


aa_resi_loc_ln01_dops_mem=[[] for _ in range(443)] #number of residues 
calc_res_depths(aa_resi_loc_ln01_6sne_ppm_04,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_dops_mem/gromacs/500ns_out.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_dops_mem/gromacs/analysis.dcd') 


# In[10]:


#121522
aa_resi_loc_lno1_wyf_ppm_01 =[[] for _ in range(443)] #number of residues 
calc_res_depths(aa_resi_loc_lno1_wyf_ppm_01,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_ppm_6snd_wyf_rep01/gromacs/500ns_all.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_ppm_6snd_wyf_rep01/gromacs/analysis.dcd')    


# In[11]:


#121522
aa_resi_loc_lno1_wyf_ppm_02 =[[] for _ in range(443)] #number of residues 
calc_res_depths(aa_resi_loc_lno1_wyf_ppm_02,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_ppm_6snd_wyf_rep02/gromacs/500ns_all.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_ppm_6snd_wyf_rep02/gromacs/analysis.dcd')    


# In[12]:


#121522
aa_resi_loc_lno1_wyf_ppm_03 =[[] for _ in range(443)] #number of residues 
calc_res_depths(aa_resi_loc_lno1_wyf_ppm_03,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_ppm_6snd_wyf_rep03/gromacs/500ns_all.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_ppm_6snd_wyf_rep03/gromacs/analysis.dcd')    


# In[13]:


#aggregate LN01 interaction profile includes all systems with same mem composition (POPC/POPA/CHOl) (3 us total)


aa_resi_loc_lno1_wyf_ppm_aggregate =[[] for _ in range(443)] 
for i in range(len(aa_resi_loc_lno1_wyf_ppm_01)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_lno1_wyf_ppm_01[i].count(1)
    aa_resi_loc_lno1_wyf_ppm_aggregate[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_lno1_wyf_ppm_01[i].count(2)
    aa_resi_loc_lno1_wyf_ppm_aggregate[i].append(glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_lno1_wyf_ppm_01[i].count(3)
    aa_resi_loc_lno1_wyf_ppm_aggregate[i].append(po4_counts)
    #water
    water_counts = aa_resi_loc_lno1_wyf_ppm_01[i].count(4)
    aa_resi_loc_lno1_wyf_ppm_aggregate[i].append(water_counts)
    
    
for i in range(len(aa_resi_loc_lno1_wyf_ppm_02)):
    acyl_counts = aa_resi_loc_lno1_wyf_ppm_02[i].count(1)
    aa_resi_loc_lno1_wyf_ppm_aggregate[i][0] =aa_resi_loc_lno1_wyf_ppm_aggregate[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_lno1_wyf_ppm_02[i].count(2)
    aa_resi_loc_lno1_wyf_ppm_aggregate[i][1] =aa_resi_loc_lno1_wyf_ppm_aggregate[i][1] + (glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_lno1_wyf_ppm_02[i].count(3)
    aa_resi_loc_lno1_wyf_ppm_aggregate[i][2] =aa_resi_loc_lno1_wyf_ppm_aggregate[i][2]+(po4_counts)
    #water
    water_counts = aa_resi_loc_lno1_wyf_ppm_02[i].count(4)
    aa_resi_loc_lno1_wyf_ppm_aggregate[i][3] =aa_resi_loc_lno1_wyf_ppm_aggregate[i][3]+(water_counts)

    
    
for i in range(len(aa_resi_loc_lno1_wyf_ppm_03)):
    acyl_counts = aa_resi_loc_lno1_wyf_ppm_03[i].count(1)
    aa_resi_loc_lno1_wyf_ppm_aggregate[i][0] =aa_resi_loc_lno1_wyf_ppm_aggregate[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_lno1_wyf_ppm_03[i].count(2)
    aa_resi_loc_lno1_wyf_ppm_aggregate[i][1] =aa_resi_loc_lno1_wyf_ppm_aggregate[i][1] + (glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_lno1_wyf_ppm_03[i].count(3)
    aa_resi_loc_lno1_wyf_ppm_aggregate[i][2] =aa_resi_loc_lno1_wyf_ppm_aggregate[i][2]+(po4_counts)
    #water
    water_counts = aa_resi_loc_lno1_wyf_ppm_03[i].count(4)
    aa_resi_loc_lno1_wyf_ppm_aggregate[i][3] =aa_resi_loc_lno1_wyf_ppm_aggregate[i][3]+(water_counts)

    


# In[106]:


print(len(aa_resi_loc_lno1_wyf_ppm_01[0]))
print(len(aa_resi_loc_lno1_wyf_ppm_02[0]))
print(len(aa_resi_loc_lno1_wyf_ppm_03[0]))


# In[14]:


with open('ln01_aggregate_interaction_profile_121522.pkl', 'wb') as handle:
    pickle.dump(aa_resi_loc_lno1_wyf_ppm_aggregate,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)


# In[9]:


#write

with open('ln01_aggregate_interaction_profile_121522.pkl', 'rb') as handle:
    aa_resi_loc_lno1_wyf_ppm_aggregate = pickle.load(handle)
    
aa_resi_loc_lno1_wyf_ppm_aggregate = np.array(aa_resi_loc_lno1_wyf_ppm_aggregate)

#holder residue list - lenght of 4e10wawa
l_chain_pgzl1_int_seq = 'AAVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPAA'
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGECQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'

l_chain_residues = split_seq2list(l_chain_pgzl1_int_seq)

plot_interaction_profile_ln01(aa_resi_loc_lno1_wyf_ppm_aggregate[:,0]/(2503*3),
                         aa_resi_loc_lno1_wyf_ppm_aggregate[:,1]/(2503*3),
                         aa_resi_loc_lno1_wyf_ppm_aggregate[:,2]/(2503*3),
                         l_chain_residues, 'ln01_aggregate_011123')


# In[16]:


#121522
aa_resi_loc_lno1_tm_wyf_ppm_01 =[[] for _ in range(443)] #number of residues 
calc_res_depths(aa_resi_loc_lno1_tm_wyf_ppm_01,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/LN01_TM-MPER/500ns_all.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/LN01_TM-MPER/analysis.dcd')    


# In[57]:


#121522
aa_resi_loc_lno1_tm_wyf_ppm_02 =[[] for _ in range(443)] #number of residues 
calc_res_depths(aa_resi_loc_lno1_tm_wyf_ppm_02,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_TM_ppm_6snd_wyf_rep02/gromacs/500ns_all.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_TM_ppm_6snd_wyf_rep02/gromacs/analysis.dcd')    


# In[58]:


#121522
aa_resi_loc_lno1_tm_wyf_ppm_03 =[[] for _ in range(443)] #number of residues 
calc_res_depths(aa_resi_loc_lno1_tm_wyf_ppm_03,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_TM_ppm_6snd_wyf_rep03/gromacs/500ns_all.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_TM_ppm_6snd_wyf_rep03/gromacs/analysis.dcd')    


# In[85]:


#aggregate LN01 interaction profile includes all systems with same mem composition (POPC/POPA/CHOl) (3 us total)


aa_resi_loc_lno1_tm_wyf_ppm_aggregate =[[] for _ in range(443)] 
for i in range(len(aa_resi_loc_lno1_tm_wyf_ppm_01)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_lno1_tm_wyf_ppm_01[i].count(1)
    aa_resi_loc_lno1_tm_wyf_ppm_aggregate[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_lno1_tm_wyf_ppm_01[i].count(2)
    aa_resi_loc_lno1_tm_wyf_ppm_aggregate[i].append(glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_lno1_tm_wyf_ppm_01[i].count(3)
    aa_resi_loc_lno1_tm_wyf_ppm_aggregate[i].append(po4_counts)
    #water
    water_counts = aa_resi_loc_lno1_tm_wyf_ppm_01[i].count(4)
    aa_resi_loc_lno1_tm_wyf_ppm_aggregate[i].append(water_counts)
    
    
for i in range(len(aa_resi_loc_lno1_tm_wyf_ppm_02)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_lno1_tm_wyf_ppm_02[i].count(1)
    aa_resi_loc_lno1_tm_wyf_ppm_aggregate[i][0] =aa_resi_loc_lno1_tm_wyf_ppm_aggregate[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_lno1_tm_wyf_ppm_02[i].count(2)
    aa_resi_loc_lno1_tm_wyf_ppm_aggregate[i][1] =aa_resi_loc_lno1_tm_wyf_ppm_aggregate[i][1] + (glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_lno1_tm_wyf_ppm_02[i].count(3)
    aa_resi_loc_lno1_tm_wyf_ppm_aggregate[i][2] =aa_resi_loc_lno1_tm_wyf_ppm_aggregate[i][2]+(po4_counts)
    #water
    water_counts = aa_resi_loc_lno1_tm_wyf_ppm_02[i].count(4)
    aa_resi_loc_lno1_tm_wyf_ppm_aggregate[i][3] =aa_resi_loc_lno1_tm_wyf_ppm_aggregate[i][3]+(water_counts)

    
for i in range(len(aa_resi_loc_lno1_tm_wyf_ppm_03)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_lno1_tm_wyf_ppm_03[i].count(1)
    aa_resi_loc_lno1_tm_wyf_ppm_aggregate[i][0] =aa_resi_loc_lno1_tm_wyf_ppm_aggregate[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_lno1_tm_wyf_ppm_03[i].count(2)
    aa_resi_loc_lno1_tm_wyf_ppm_aggregate[i][1] =aa_resi_loc_lno1_tm_wyf_ppm_aggregate[i][1] + (glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_lno1_tm_wyf_ppm_03[i].count(3)
    aa_resi_loc_lno1_tm_wyf_ppm_aggregate[i][2] =aa_resi_loc_lno1_tm_wyf_ppm_aggregate[i][2]+(po4_counts)
    #water
    water_counts = aa_resi_loc_lno1_tm_wyf_ppm_03[i].count(4)
    aa_resi_loc_lno1_tm_wyf_ppm_aggregate[i][3] =aa_resi_loc_lno1_tm_wyf_ppm_aggregate[i][3]+(water_counts)


# In[90]:


# print(aa_resi_loc_lno1_tm_wyf_ppm_03[0])
#print(aa_resi_loc_lno1_tm_wyf_ppm_aggregate)


# In[80]:


# with open('ln01_tm_aggregate_interaction_profile_121522.pkl', 'wb') as handle:
#     pickle.dump(aa_resi_loc_lno1_tm_wyf_ppm_aggregate,
#                 handle,
#                 protocol=pickle.HIGHEST_PROTOCOL)
    
    


# In[88]:


print(len(aa_resi_loc_lno1_tm_wyf_ppm_03[0]))
print(len(aa_resi_loc_lno1_tm_wyf_ppm_02[0]))
print(len(aa_resi_loc_lno1_tm_wyf_ppm_01[0]))


# In[17]:


#write

with open('ln01_tm_aggregate_interaction_profile_121522.pkl', 'rb') as handle:
    aa_resi_loc_lno1_tm_wyf_ppm_aggregate = pickle.load(handle)
    
aa_resi_loc_lno1_tm_wyf_ppm_aggregate = np.array(aa_resi_loc_lno1_tm_wyf_ppm_aggregate)

#holder residue list - lenght of 4e10wawa
l_chain_pgzl1_int_seq = 'AAVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPAA'
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGECQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'

l_chain_residues = split_seq2list(l_chain_pgzl1_int_seq)

plot_interaction_profile_ln01(aa_resi_loc_lno1_tm_wyf_ppm_aggregate[:,0]/(2503*3),
                         aa_resi_loc_lno1_tm_wyf_ppm_aggregate[:,1]/(2503*3),
                         aa_resi_loc_lno1_tm_wyf_ppm_aggregate[:,2]/(2503*3),
                         l_chain_residues, 'ln01_tm_aggregate_121522')


# In[16]:


# #aggregate LN01 interaction profile includes all systems with same mem composition (POPC/POPA/CHOl) (3 us total)


# aa_resi_loc_ln01_aggregate_counts =[[] for _ in range(443)] 
# for i in range(len(aa_resi_loc_lno1_opm_rep1)):
#     #acyl location counts ]
#     acyl_counts = aa_resi_loc_lno1_opm_rep1[i].count(1)
#     aa_resi_loc_ln01_aggregate_counts[i].append(acyl_counts)
#     #glycerol_counts 
#     glycerol_counts = aa_resi_loc_lno1_opm_rep1[i].count(2)
#     aa_resi_loc_ln01_aggregate_counts[i].append(glycerol_counts)
#     #po4 
#     po4_counts = aa_resi_loc_lno1_opm_rep1[i].count(3)
#     aa_resi_loc_ln01_aggregate_counts[i].append(po4_counts)
#     #water
#     water_counts = aa_resi_loc_lno1_opm_rep1[i].count(4)
#     aa_resi_loc_ln01_aggregate_counts[i].append(water_counts)
    
    
# for i in range(len(aa_resi_loc_ln01_6snd_ppm_02)):
#     #acyl location counts ]
#     acyl_counts = aa_resi_loc_ln01_6snd_ppm_02[i].count(1)
#     aa_resi_loc_ln01_aggregate_counts[i].append(acyl_counts)
#     #glycerol_counts 
#     glycerol_counts = aa_resi_loc_ln01_6snd_ppm_02[i].count(2)
#     aa_resi_loc_ln01_aggregate_counts[i].append(glycerol_counts)
#     #po4 
#     po4_counts = aa_resi_loc_ln01_6snd_ppm_02[i].count(3)
#     aa_resi_loc_ln01_aggregate_counts[i].append(po4_counts)
#     #water
#     water_counts = aa_resi_loc_ln01_6snd_ppm_02[i].count(4)
#     aa_resi_loc_ln01_aggregate_counts[i].append(water_counts)
    
# # for i in range(len(aa_resi_loc_ln01_6sne_ppm_03)):
# #     #acyl location counts ]
# #     acyl_counts = aa_resi_loc_ln01_6sne_ppm_03[i].count(1)
# #     aa_resi_loc_ln01_aggregate_counts[i].append(acyl_counts)
# #     #glycerol_counts 
# #     glycerol_counts = aa_resi_loc_ln01_6sne_ppm_03[i].count(2)
# #     aa_resi_loc_ln01_aggregate_counts[i].append(glycerol_counts)
# #     #po4 
# #     po4_counts = aa_resi_loc_ln01_6sne_ppm_03[i].count(3)
# #     aa_resi_loc_ln01_aggregate_counts[i].append(po4_counts)
# #     #water
# #     water_counts = aa_resi_loc_ln01_6sne_ppm_03[i].count(4)
# #     aa_resi_loc_ln01_aggregate_counts[i].append(water_counts)
    
# for i in range(len(aa_resi_loc_ln01_6sne_ppm_04)):
#     #acyl location counts ]
#     acyl_counts = aa_resi_loc_ln01_6sne_ppm_04[i].count(1)
#     aa_resi_loc_ln01_aggregate_counts[i].append(acyl_counts)
#     #glycerol_counts 
#     glycerol_counts = aa_resi_loc_ln01_6sne_ppm_04[i].count(2)
#     aa_resi_loc_ln01_aggregate_counts[i].append(glycerol_counts)
#     #po4 
#     po4_counts = aa_resi_loc_ln01_6sne_ppm_04[i].count(3)
#     aa_resi_loc_ln01_aggregate_counts[i].append(po4_counts)
#     #water
#     water_counts = aa_resi_loc_ln01_6sne_ppm_04[i].count(4)
#     aa_resi_loc_ln01_aggregate_counts[i].append(water_counts)


# In[7]:


with open('ln01_tm_aggregate_interaction_profile_121522.pkl', 'rb') as handle:
    aa_resi_loc_lno1_tm_wyf_ppm_aggregate = pickle.load(handle) 
with open('ln01_aggregate_interaction_profile_121522.pkl', 'rb') as handle:
    aa_resi_loc_lno1_wyf_ppm_aggregate = pickle.load(handle)
    


# In[32]:


aa_resi_loc_lno1_wyf_ppm_aggregate = np.array(aa_resi_loc_lno1_wyf_ppm_aggregate)

with open('ln01_interaction_prof_annotated.csv', 'w', newline='') as csvfile:
    fieldnames = ['Residue', 'Hydrocarbon', 'Glycerol','PO4','Water'  ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(aa_resi_loc_lno1_wyf_ppm_aggregate)): 

        writer.writerow({'Residue':ln01_seq[i], 
                         'Hydrocarbon': aa_resi_loc_lno1_wyf_ppm_aggregate[i][0],
                         'Glycerol': aa_resi_loc_lno1_wyf_ppm_aggregate[i][1],
                         'PO4': aa_resi_loc_lno1_wyf_ppm_aggregate[i][2],
                         'Water': aa_resi_loc_lno1_wyf_ppm_aggregate[i][3]})


# In[33]:


aa_resi_loc_lno1_tm_wyf_ppm_aggregate = np.array(aa_resi_loc_lno1_tm_wyf_ppm_aggregate)

with open('ln01_TM_interaction_prof_annotated.csv', 'w', newline='') as csvfile:
    fieldnames = ['Residue', 'Hydrocarbon', 'Glycerol','PO4','Water'  ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(aa_resi_loc_lno1_tm_wyf_ppm_aggregate)): 

        writer.writerow({'Residue':ln01_seq[i], 
                         'Hydrocarbon': aa_resi_loc_lno1_tm_wyf_ppm_aggregate[i][0],
                         'Glycerol': aa_resi_loc_lno1_tm_wyf_ppm_aggregate[i][1],
                         'PO4': aa_resi_loc_lno1_tm_wyf_ppm_aggregate[i][2],
                         'Water': aa_resi_loc_lno1_tm_wyf_ppm_aggregate[i][3]})


# In[37]:


with open('4e10_AA_interaction_profile.pickle', 'rb') as handle:
    interaction_prof_4e10_AA = pickle.load(handle)
interaction_prof_4e10_AA = np.array(interaction_prof_4e10_AA)


with open('4e10_interaction_prof_annotated.csv', 'w', newline='') as csvfile:
    fieldnames = ['Residue', 'Hydrocarbon', 'Glycerol','PO4','Water'  ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(interaction_prof_4e10_AA)): 

        writer.writerow({'Residue':ab_4e10_seq[i], 
                         'Hydrocarbon': interaction_prof_4e10_AA[i][0],
                         'Glycerol': interaction_prof_4e10_AA[i][1],
                         'PO4': interaction_prof_4e10_AA[i][2],
                         'Water': interaction_prof_4e10_AA[i][3]})


# In[46]:


with open('pgzl1_AA_interaction_profile.pickle', 'rb') as handle:
    interaction_prof_pgzl1_AA = pickle.load(handle)
interaction_prof_pgzl1_AA = np.array(interaction_prof_pgzl1_AA)

# print(interaction_prof_pgzl1_AA[-5])

# print(len(ab_pgzl1_seq))
with open('pgzl1_interaction_prof_annotated.csv', 'w', newline='') as csvfile:
    fieldnames = ['Residue', 'Hydrocarbon', 'Glycerol','PO4','Water'  ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(ab_pgzl1_seq)): 

        writer.writerow({'Residue':ab_pgzl1_seq[i], 
                         'Hydrocarbon': interaction_prof_pgzl1_AA[i][0],
                         'Glycerol': interaction_prof_pgzl1_AA[i][1],
                         'PO4': interaction_prof_pgzl1_AA[i][2],
                         'Water': interaction_prof_pgzl1_AA[i][3]})


# In[16]:



aa_resi_loc_lno1_wyf_ppm_aggregate = np.array(aa_resi_loc_lno1_wyf_ppm_aggregate)

#holder residue list - lenght of 4e10wawa
l_chain_pgzl1_int_seq = 'AAVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPAA'
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGECQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'

l_chain_residues = split_seq2list(l_chain_pgzl1_int_seq)

plot_interaction_profile_ln01_ticks(aa_resi_loc_lno1_wyf_ppm_aggregate[:,0]/(2503*3),
                         aa_resi_loc_lno1_wyf_ppm_aggregate[:,1]/(2503*3),
                         aa_resi_loc_lno1_wyf_ppm_aggregate[:,2]/(2503*3),
                         l_chain_residues, 'ln01_aggregate_TICKS')


# In[32]:


#write
with open('ln01_aggregate_interaction_profile.pkl', 'wb') as handle:
    pickle.dump(aa_resi_loc_ln01_aggregate_counts,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)
with open('ln01_aggregate_interaction_profile.pkl', 'rb') as handle:
    interaction_prof_ln01_aggregate = pickle.load(handle)
    
interaction_prof_ln01_aggregate = np.array(interaction_prof_ln01_aggregate)

#holder residue list - lenght of 4e10wawa
l_chain_pgzl1_int_seq = 'AAVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPAA'
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGECQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'

l_chain_residues = split_seq2list(l_chain_pgzl1_int_seq)
print(len(l_chain_residues))
plot_interaction_profile_ln01(interaction_prof_ln01_aggregate[:,0],
                         interaction_prof_ln01_aggregate[:,1],
                         interaction_prof_ln01_aggregate[:,2],
                         l_chain_residues, 'ln01_aggregate')


# In[36]:


aa_resi_loc_lno1_tm_mper_02 =[[] for _ in range(443)] #number of residues 
calc_res_depths(aa_resi_loc_lno1_tm_mper_02,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_tm_mper_02/1000ns_out.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_tm_mper_02/analysis.dcd')    


# In[37]:


aa_resi_loc_lno1_tm_mper_04 =[[] for _ in range(443)] #number of residues 
calc_res_depths(aa_resi_loc_lno1_tm_mper_04,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_tm_mper_04/100ns.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_tm_mper_04/analysis.dcd')    


# In[39]:


aa_resi_loc_lno1_tm_mper_05 =[[] for _ in range(443)] #number of residues 
calc_res_depths(aa_resi_loc_lno1_tm_mper_05,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_tm_mper_05/gromacs/500ns_out.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_tm_mper_05/gromacs/analysis.dcd')    


# In[40]:


aa_resi_loc_lno1_tm_mper_aggregate_counts =[[] for _ in range(443)] 
for i in range(len(aa_resi_loc_lno1_tm_mper_02)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_lno1_tm_mper_02[i].count(1)
    aa_resi_loc_lno1_tm_mper_aggregate_counts[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_lno1_tm_mper_02[i].count(2)
    aa_resi_loc_lno1_tm_mper_aggregate_counts[i].append(glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_lno1_tm_mper_02[i].count(3)
    aa_resi_loc_lno1_tm_mper_aggregate_counts[i].append(po4_counts)
    #water
    water_counts = aa_resi_loc_lno1_tm_mper_02[i].count(4)
    aa_resi_loc_lno1_tm_mper_aggregate_counts[i].append(water_counts)
    
    
for i in range(len(aa_resi_loc_lno1_tm_mper_04)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_lno1_tm_mper_04[i].count(1)
    aa_resi_loc_lno1_tm_mper_aggregate_counts[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_lno1_tm_mper_04[i].count(2)
    aa_resi_loc_lno1_tm_mper_aggregate_counts[i].append(glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_lno1_tm_mper_04[i].count(3)
    aa_resi_loc_lno1_tm_mper_aggregate_counts[i].append(po4_counts)
    #water
    water_counts = aa_resi_loc_lno1_tm_mper_04[i].count(4)
    aa_resi_loc_lno1_tm_mper_aggregate_counts[i].append(water_counts)
    
for i in range(len(aa_resi_loc_lno1_tm_mper_05)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_lno1_tm_mper_05[i].count(1)
    aa_resi_loc_lno1_tm_mper_aggregate_counts[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_lno1_tm_mper_05[i].count(2)
    aa_resi_loc_lno1_tm_mper_aggregate_counts[i].append(glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_lno1_tm_mper_05[i].count(3)
    aa_resi_loc_lno1_tm_mper_aggregate_counts[i].append(po4_counts)
    #water
    water_counts = aa_resi_loc_lno1_tm_mper_05[i].count(4)
    aa_resi_loc_lno1_tm_mper_aggregate_counts[i].append(water_counts)


# In[47]:


pdb = parsePDB('/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_tm_mper_05/gromacs/500ns_out.pdb')
fab = pdb.select('name CA' )[0:442]
print(len(fab))
for i in range(len(fab)):
    print(i, fab[i].getResname(), fab[i].getChid())


# In[48]:


#write
with open('ln01_tm_aggregate_interaction_profile.pkl', 'wb') as handle:
    pickle.dump(aa_resi_loc_lno1_tm_mper_aggregate_counts,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)
with open('ln01_tm_aggregate_interaction_profile.pkl', 'rb') as handle:
    ln01_tm_aggregate_interaction_profile = pickle.load(handle)
    
ln01_tm_aggregate_interaction_profile = np.array(ln01_tm_aggregate_interaction_profile)

#holder residue list - lenght of 4e10wawa
l_chain_pgzl1_int_seq = 'VLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPAA'
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGECQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'

l_chain_residues = split_seq2list(l_chain_pgzl1_int_seq)
print(len(l_chain_residues))
plot_interaction_profile_ln01(ln01_tm_aggregate_interaction_profile[:441,0],
                         ln01_tm_aggregate_interaction_profile[:441,1],
                         ln01_tm_aggregate_interaction_profile[:441,2],
                         l_chain_residues, 'ln01_tm_aggregate')


# In[25]:



aa_resi_loc_pgzl1_int_ppm =[[] for _ in range(441)] #number of residues 
calc_res_depths(aa_resi_loc_pgzl1_int_ppm,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_int_ppm/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_int_ppm/final_analysis_traj.dcd')    

#test 500ns replicate 4E10 WAWA AA interaction profile - collect counts from each trajectory into one list 
aa_resi_loc_4e10_ppm_counts =[[] for _ in range(441)] 
for i in range(len(aa_resi_loc_pgzl1_int_ppm)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_pgzl1_int_ppm[i].count(1)
    aa_resi_loc_4e10_ppm_counts[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_pgzl1_int_ppm[i].count(2)
    aa_resi_loc_4e10_ppm_counts[i].append(glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_pgzl1_int_ppm[i].count(3)
    aa_resi_loc_4e10_ppm_counts[i].append(po4_counts)
    #water
    water_counts = aa_resi_loc_pgzl1_int_ppm[i].count(4)
    aa_resi_loc_4e10_ppm_counts[i].append(water_counts)
#write
with open('pgzl1_int_AA_interaction_profile.pkl', 'wb') as handle:
    pickle.dump(aa_resi_loc_4e10_ppm_counts,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)
with open('pgzl1_int_AA_interaction_profile.pkl', 'rb') as handle:
    interaction_prof_pgzl1_int_AA = pickle.load(handle)
    
interaction_prof_pgzl1_int_AA = np.array(interaction_prof_pgzl1_int_AA)


#holder residue list - lenght of 4e10wawa
l_chain_pgzl1_int_seq = 'VLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPAA'
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGECQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'

l_chain_residues = split_seq2list(l_chain_pgzl1_int_seq)
print(len(l_chain_residues))
plot_interaction_profile(interaction_prof_pgzl1_int_AA[:,0],
                         interaction_prof_pgzl1_int_AA[:,1],
                         interaction_prof_pgzl1_int_AA[:,2],
                         l_chain_residues, 'pgzl1_int_CHECK')


# In[26]:



aa_resi_loc_pgzl1_germ_ppm =[[] for _ in range(441)] #number of residues 
calc_res_depths(aa_resi_loc_pgzl1_germ_ppm,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_germ_ppm/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_germ_ppm/final_analysis_traj.dcd')    

#test 500ns replicate 4E10 WAWA AA interaction profile - collect counts from each trajectory into one list 
aa_resi_loc_4e10_ppm_counts =[[] for _ in range(441)] 
for i in range(len(aa_resi_loc_pgzl1_germ_ppm)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_pgzl1_germ_ppm[i].count(1)
    aa_resi_loc_4e10_ppm_counts[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_pgzl1_germ_ppm[i].count(2)
    aa_resi_loc_4e10_ppm_counts[i].append(glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_pgzl1_germ_ppm[i].count(3)
    aa_resi_loc_4e10_ppm_counts[i].append(po4_counts)
    #water
    water_counts = aa_resi_loc_pgzl1_germ_ppm[i].count(4)
    aa_resi_loc_4e10_ppm_counts[i].append(water_counts)
#write
with open('pgzl1_germ_AA_interaction_profile.pkl', 'wb') as handle:
    pickle.dump(aa_resi_loc_4e10_ppm_counts,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)
with open('pgzl1_germ_AA_interaction_profile.pkl', 'rb') as handle:
    interaction_prof_pgzl1_germ_AA = pickle.load(handle)
    
interaction_prof_pgzl1_germ_AA = np.array(interaction_prof_pgzl1_germ_AA)


#holder residue list - lenght of 4e10wawa
l_chain_pgzl1_germ_seq = 'VLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPAA'
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGECQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'

l_chain_residues = split_seq2list(l_chain_pgzl1_germ_seq)
print(len(l_chain_residues))
plot_interaction_profile(interaction_prof_pgzl1_germ_AA[:,0],
                         interaction_prof_pgzl1_germ_AA[:,1],
                         interaction_prof_pgzl1_germ_AA[:,2],
                         l_chain_residues, 'pgzl1_germ_CHECK')


# In[23]:



aa_resi_loc_4e10_wawa_ppm =[[] for _ in range(441)] #number of residues 
calc_res_depths(aa_resi_loc_4e10_wawa_ppm,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4E10_WAWA_mm/analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4E10_WAWA_mm/final_md.dcd')    

#test 500ns replicate 4E10 WAWA AA interaction profile - collect counts from each trajectory into one list 
aa_resi_loc_4e10_ppm_counts =[[] for _ in range(441)] 
for i in range(len(aa_resi_loc_4e10_wawa_ppm)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_4e10_wawa_ppm[i].count(1)
    aa_resi_loc_4e10_ppm_counts[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_4e10_wawa_ppm[i].count(2)
    aa_resi_loc_4e10_ppm_counts[i].append(glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_4e10_wawa_ppm[i].count(3)
    aa_resi_loc_4e10_ppm_counts[i].append(po4_counts)
    #water
    water_counts = aa_resi_loc_4e10_wawa_ppm[i].count(4)
    aa_resi_loc_4e10_ppm_counts[i].append(water_counts)
#write
with open('4e10_wawa_AA_interaction_profile.pkl', 'wb') as handle:
    pickle.dump(aa_resi_loc_4e10_ppm_counts,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)
with open('4e10_wawa_AA_interaction_profile.pkl', 'rb') as handle:
    interaction_prof_4e10_wawa_AA = pickle.load(handle)
    
interaction_prof_4e10_wawa_AA = np.array(interaction_prof_4e10_wawa_AA)


#holder residue list - lenght of 4e10wawa
l_chain_4e10_wawa_seq = 'VLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPAA'
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGECQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'

l_chain_residues = split_seq2list(l_chain_4e10_wawa_seq)
print(len(l_chain_residues))
plot_interaction_profile(interaction_prof_4e10_wawa_AA[:,0],
                         interaction_prof_4e10_wawa_AA[:,1],
                         interaction_prof_4e10_wawa_AA[:,2],
                         l_chain_residues, '4e10_wawa_CHECK')


# In[24]:





# In[17]:


#test 500ns replicate ln01 AA interaction profile - collect counts from each trajectory into one list 
aa_resi_loc_lno1_ppm_counts =[[] for _ in range(443)] 
for i in range(len(aa_resi_loc_lno1_ppm)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_lno1_ppm[i].count(1)
    aa_resi_loc_lno1_ppm_counts[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_lno1_ppm[i].count(2)
    aa_resi_loc_lno1_ppm_counts[i].append(glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_lno1_ppm[i].count(3)
    aa_resi_loc_lno1_ppm_counts[i].append(po4_counts)
    #water
    water_counts = aa_resi_loc_lno1_ppm[i].count(4)
    aa_resi_loc_lno1_ppm_counts[i].append(water_counts)
    
#print(aa_resi_loc_lno1_ppm_counts)


# In[15]:



#write
with open('ln01_AA_interaction_profile.pkl', 'wb') as handle:
    pickle.dump(aa_resi_loc_lno1_ppm_counts,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)
with open('ln01_AA_interaction_profile.pkl', 'rb') as handle:
    interaction_prof_ln01_AA = pickle.load(handle)
    
interaction_prof_ln01_AA = np.array(interaction_prof_ln01_AA)


# In[16]:



#holder residue list - lenght of ln01
l_chain_ln01_seq = 'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPAA'
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGECQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'

l_chain_residues = split_seq2list(l_chain_ln01_seq)
print(len(l_chain_residues))
plot_interaction_profile(interaction_prof_ln01_AA[:,0],
                         interaction_prof_ln01_AA[:,1],
                         interaction_prof_ln01_AA[:,2],
                         l_chain_residues, 'ln01_CHECK')


# In[6]:


#aggregate 4e10 AA interaction profile - collect counts from each trajectory into one list 
aa_resi_loc_counts_4e10_aggregate =[[] for _ in range(441)] 
for i in range(len(aa_resi_loc_4e10_ppm_1us)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_4e10_ppm_1us[i].count(1)
    aa_resi_loc_counts_4e10_aggregate[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_4e10_ppm_1us[i].count(2)
    aa_resi_loc_counts_4e10_aggregate[i].append(glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_4e10_ppm_1us[i].count(3)
    aa_resi_loc_counts_4e10_aggregate[i].append(po4_counts)
    #water
    water_counts = aa_resi_loc_4e10_ppm_1us[i].count(4)
    aa_resi_loc_counts_4e10_aggregate[i].append(water_counts)
print(aa_resi_loc_counts_4e10_aggregate[318])   
for i in range(len(aa_resi_loc_4e10_ppm)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_4e10_ppm[i].count(1)
    aa_resi_loc_counts_4e10_aggregate[i][0] =aa_resi_loc_counts_4e10_aggregate[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_4e10_ppm[i].count(2)
    aa_resi_loc_counts_4e10_aggregate[i][1] =aa_resi_loc_counts_4e10_aggregate[i][1] + (glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_4e10_ppm[i].count(3)
    aa_resi_loc_counts_4e10_aggregate[i][2] =aa_resi_loc_counts_4e10_aggregate[i][2] + (po4_counts)
    #water
    water_counts = aa_resi_loc_4e10_ppm[i].count(4)
    aa_resi_loc_counts_4e10_aggregate[i][3] =aa_resi_loc_counts_4e10_aggregate[i][3] + (water_counts)
for i in range(len(aa_resi_loc_4e10_p15)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_4e10_p15[i].count(1)
    aa_resi_loc_counts_4e10_aggregate[i][0] =aa_resi_loc_counts_4e10_aggregate[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_4e10_p15[i].count(2)
    aa_resi_loc_counts_4e10_aggregate[i][1] =aa_resi_loc_counts_4e10_aggregate[i][1] + (glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_4e10_p15[i].count(3)
    aa_resi_loc_counts_4e10_aggregate[i][2] =aa_resi_loc_counts_4e10_aggregate[i][2] + (po4_counts)
    #water
    water_counts = aa_resi_loc_4e10_p15[i].count(4)
    aa_resi_loc_counts_4e10_aggregate[i][3] =aa_resi_loc_counts_4e10_aggregate[i][3] + (water_counts)
    
for i in range(len(aa_resi_loc_4e10_n15)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_4e10_n15[i].count(1)
    aa_resi_loc_counts_4e10_aggregate[i][0] =aa_resi_loc_counts_4e10_aggregate[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_4e10_n15[i].count(2)
    aa_resi_loc_counts_4e10_aggregate[i][1] =aa_resi_loc_counts_4e10_aggregate[i][1] + (glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_4e10_n15[i].count(3)
    aa_resi_loc_counts_4e10_aggregate[i][2] =aa_resi_loc_counts_4e10_aggregate[i][2] + (po4_counts)
    #water
    water_counts = aa_resi_loc_4e10_n15[i].count(4)
    aa_resi_loc_counts_4e10_aggregate[i][3] =aa_resi_loc_counts_4e10_aggregate[i][3] + (water_counts)
    
  


# In[34]:


import pickle

#data: aa_resi_loc_counts_4e10_aggregate 

# # Store data (serialize)
# with open('4e10_AA_interaction_profile.pickle', 'wb') as handle:
#     pickle.dump(aa_resi_loc_counts_4e10_aggregate,
#                 handle,
#                 protocol=pickle.HIGHEST_PROTOCOL)
    
    # Load data (deserialize)
with open('4e10_AA_interaction_profile.pickle', 'rb') as handle:
    interaction_prof_4e10_AA = pickle.load(handle)
interaction_prof_4e10_AA = np.array(interaction_prof_4e10_AA)


# In[11]:



  
def plot_interaction_profile(acyl_loc, glycerol_loc, po4_loc, l_chain_residues, prefix): 
  fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 8), gridspec_kw={'height_ratios': [.5, 6]})
  plt.subplots_adjust(hspace=0.01)
  #x = np.arange(0, len(l_chain_residues),1)
  xticks = [i for i in range(0,len(l_chain_residues))]
  #add cdr loop annotation 
  l_chain_annotate = [] 
  for i in range(len(l_chain_residues)):
      #add cdrl1 loop 
      if i in range(27, 33):
          l_chain_annotate.append(1)
      elif i in range(50,53):
          l_chain_annotate.append(1)
      elif i in range(89,98):
          l_chain_annotate.append(1)
      #add H chain loops 
      elif i in range(239, 247):
          l_chain_annotate.append(2)
      elif i in range(263,271):
          l_chain_annotate.append(2)
      elif i in range(310,330):
          l_chain_annotate.append(2)        
      else:
          l_chain_annotate.append(0)

  ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==1, 
                   alpha=1, color='#55A3ff') 
  
  ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==2, 
                  alpha=1, color='#2A517F') 
  ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==0, 
                  alpha=.5, color='#BFBFBF')

  ax1.axhline(y=-10, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
  ax1.axhline(y=0, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
  #mark antibody chain start/stop 
  ax1.axvline(x=213, color='#000000', linestyle='-', ymin=0, linewidth=1.5)
  ax1.axvline(x=441, color='#000000', linestyle='-', ymin=0, linewidth=1.5)
  ax1.spines["right"].set_visible(False)
  ax1.spines["top"].set_visible(False)
  ax1.spines["bottom"].set_visible(False)
  ax1.set_xticks([])
  ax1.set_yticks([])

  width=1
  ax2.bar(xticks, acyl_loc, width, label='Acyl', color='blue')
  ax2.bar(xticks, glycerol_loc, width, bottom=acyl_loc, label='Glycerol', color='red')
  ax2.bar(xticks, po4_loc, width, bottom=glycerol_loc+acyl_loc, label='PO4', color='orange')
  
 
  ax2.set_xticks([]) #xticks
  ax2.set_xticklabels([])#l_chain_residues
  ax2.set_yticks([0, 1]) 
  #ax2.set_yticks([0, 5033, 10066, 15099, 20123]) 
  #ax2.set_yticklabels(['0', '1', '2', '3', '4'])
  ax2.tick_params(axis='x', labelsize=50)
  ax2.tick_params(axis='y', labelsize=50)#, fontsize=16
  ax2.set_ylabel(r"Time ($\rm \mu s$)", fontsize=54)
  #ax2.set_xlabel('Residue', fontsize54)
  ax2.spines["right"].set_visible(False)
  ax2.spines["top"].set_visible(False)
  
  
  ax1.set_xlim(0, len(xticks))
  ax2.set_xlim(0, len(xticks))
  
  fig_name = prefix + '_interaction_profile.png' 
  plt.savefig(fig_name, transparent=True, bbox_inches="tight")

  plt.show()
  return "Made figure: "#, prefix



# In[15]:



#get list of residues in l chain 4e10 
l_chain_4e10_seq = 'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEP'
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGECQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'

l_chain_residues = split_seq2list(l_chain_4e10_seq)
print(len(l_chain_residues))
plot_interaction_profile(interaction_prof_4e10_AA[:,0],
                         interaction_prof_4e10_AA[:,1],
                         interaction_prof_4e10_AA[:,2],
                         l_chain_residues, '4e10_AA')


# In[30]:


aa_resi_loc_4e10_cg_revert_med2 =[[] for _ in range(441)] #number of residues 
calc_res_depths(aa_resi_loc_4e10_cg_revert_med2,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/4e10/medoid2/analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/4e10/medoid2/traj_1000ns.dcd')  


# In[31]:


aa_resi_loc_4e10_cg_revert_med2_count =[[] for _ in range(441)] 
for i in range(len(aa_resi_loc_4e10_cg_revert_med2)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_4e10_cg_revert_med2[i].count(1)
    aa_resi_loc_4e10_cg_revert_med2_count[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_4e10_cg_revert_med2[i].count(2)
    aa_resi_loc_4e10_cg_revert_med2_count[i].append(glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_4e10_cg_revert_med2[i].count(3)
    aa_resi_loc_4e10_cg_revert_med2_count[i].append(po4_counts)
    #water
    water_counts = aa_resi_loc_4e10_cg_revert_med2[i].count(4)
    aa_resi_loc_4e10_cg_revert_med2_count[i].append(water_counts)

aa_resi_loc_4e10_cg_revert_med2_count = np.array(aa_resi_loc_4e10_cg_revert_med2_count)


# In[32]:



#get list of residues in l chain 4e10 
l_chain_4e10_seq = 'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEP'
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGECQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'
l_chain_residues = split_seq2list(l_chain_4e10_seq)
print(len(l_chain_residues))
plot_interaction_profile(aa_resi_loc_4e10_cg_revert_med2_count[:,0],
                         aa_resi_loc_4e10_cg_revert_med2_count[:,1],
                         aa_resi_loc_4e10_cg_revert_med2_count[:,2],
                         l_chain_residues, '4e10_cgMed2_revert')


# In[28]:


aa_resi_loc_4e10_cg_revert_med3 =[[] for _ in range(441)] #number of residues 
calc_res_depths(aa_resi_loc_4e10_cg_revert_med3,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/4e10/medoid3/analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/4e10/medoid3/traj_1000ns.dcd')  


# In[29]:


aa_resi_loc_4e10_cg_revert_med3_count =[[] for _ in range(441)] 
for i in range(len(aa_resi_loc_4e10_cg_revert_med3)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_4e10_cg_revert_med3[i].count(1)
    aa_resi_loc_4e10_cg_revert_med3_count[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_4e10_cg_revert_med3[i].count(2)
    aa_resi_loc_4e10_cg_revert_med3_count[i].append(glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_4e10_cg_revert_med3[i].count(3)
    aa_resi_loc_4e10_cg_revert_med3_count[i].append(po4_counts)
    #water
    water_counts = aa_resi_loc_4e10_cg_revert_med3[i].count(4)
    aa_resi_loc_4e10_cg_revert_med3_count[i].append(water_counts)

aa_resi_loc_4e10_cg_revert_med3_count = np.array(aa_resi_loc_4e10_cg_revert_med3_count)


#get list of residues in l chain 4e10 
l_chain_4e10_seq = 'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEP'
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGECQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'
l_chain_residues = split_seq2list(l_chain_4e10_seq)
print(len(l_chain_residues))
plot_interaction_profile(aa_resi_loc_4e10_cg_revert_med3_count[:,0],
                         aa_resi_loc_4e10_cg_revert_med3_count[:,1],
                         aa_resi_loc_4e10_cg_revert_med3_count[:,2],
                         l_chain_residues, '4e10_cgmed3_revert')


# In[23]:





# In[9]:


def split_seq2list(seq_string):
    return [residue for residue in seq_string ]


# In[ ]:


# #print(aa_resi_loc_counts_4e10_aggregate[318])      
    
# aa_resi_loc_counts_4e10_aggregate = np.array(aa_resi_loc_counts_4e10_aggregate)
# #convert to np array
# aa_resi_loc_counts_4e10_aggregate =np.array([np.array(loc_counts) for loc_counts in aa_resi_loc_counts_4e10_aggregate])

# labels = [i for i in range(1, 442)]
# xticks = [i for i in range(0, 441, 10)]
# #stack 4 lists 
# #print(aa_resi_loc_counts_4e10_ppm_1us[:,0])
# acyl_loc = aa_resi_loc_counts_4e10_aggregate[:,0]*.0002
# #print(acyl_loc)
# glycerol_loc = aa_resi_loc_counts_4e10_aggregate[:,1]*.0002
# po4_loc = aa_resi_loc_counts_4e10_aggregate[:,2]*.0002
# water_loc = aa_resi_loc_counts_4e10_aggregate[:,3]*.0002
# #print(len(po4_loc))
# width = 1     # the width of the bars: can also be len(x) sequence

# fig, ax = plt.subplots(figsize=(9, 4))

# ax.bar(labels, acyl_loc, width, label='Acyl', color='blue')
# ax.bar(labels, glycerol_loc, width, bottom=acyl_loc, label='Glycerol', color='red')
# ax.bar(labels, po4_loc, width, bottom=glycerol_loc, label='PO4', color='orange')
# #ax.bar(labels, water_loc, width,  bottom=po4_loc, label='water', color='lightgrey')
# #ax.set_ylim(0, 10)
# ax.set_ylabel(r"Time ($\rm \mu s$)", fontsize=20)
# ax.tick_params(axis='x', labelsize=16)
# ax.tick_params(axis='y', labelsize=16)
# ax.set_xlabel('Residue', fontsize=20)

# ax.set_xticks(xticks)
# ax.set_xlim(200,330)
# #ax.set_title('Scores by group and gender')
# #ax.legend()
# ax.set_title("4e10 aggregate")
# plt.show()


# In[36]:


aa_resi_loc_pgzl1_ppm =[[] for _ in range(441)] #number of residues 
calc_res_depths(aa_resi_loc_pgzl1_ppm,
                '/home/bat-gpu/colleen/phos_interactions/final_traj/pgzl1_ppm/analysis_input.pdb',
                '/home/bat-gpu/colleen/phos_interactions/final_traj/pgzl1_ppm/final_md.dcd')    

aa_resi_loc_pgzl1_p15 =[[] for _ in range(441)] #number of residues 
calc_res_depths(aa_resi_loc_pgzl1_p15,
                '/home/bat-gpu/colleen/phos_interactions/final_traj/pgzl1_p15/analysis_input.pdb',
                '/home/bat-gpu/colleen/phos_interactions/final_traj/pgzl1_p15/final_md.dcd')    

aa_resi_loc_pgzl1_n15 =[[] for _ in range(441)] #number of residues 
calc_res_depths(aa_resi_loc_pgzl1_n15,
                '/home/bat-gpu/colleen/phos_interactions/final_traj/pgzl1_n15/analysis_input.pdb',
                '/home/bat-gpu/colleen/phos_interactions/final_traj/pgzl1_n15/final_md.dcd')    

aa_resi_loc_pgzl1_ppm_1us =[[] for _ in range(441)] #number of residues 
calc_res_depths(aa_resi_loc_pgzl1_ppm_1us,
                '/home/bat-gpu/colleen/phos_interactions/final_traj/pgzl1_ppm_1us/analysis_input.pdb',
                '/home/bat-gpu/colleen/phos_interactions/final_traj/pgzl1_ppm_1us/final_md.dcd')    


# In[ ]:





# In[35]:


# aa_resi_loc_108e_ppm =[[] for _ in range(444)] #number of residues 
calc_res_depths(aa_resi_loc_108e_ppm,
                '/home/bat-gpu/colleen/phos_interactions/final_traj/108e_ppm/analysis_input.pdb',
                '/home/bat-gpu/colleen/phos_interactions/final_traj/108e_ppm/final_md.dcd')    

aa_resi_loc_108e_p15 =[[] for _ in range(444)] #number of residues 
calc_res_depths(aa_resi_loc_108e_p15,
                '/home/bat-gpu/colleen/phos_interactions/final_traj/108e_p15/analysis_input.pdb',
                '/home/bat-gpu/colleen/phos_interactions/final_traj/108e_p15/complete_md.dcd')    

aa_resi_loc_108e_n15 =[[] for _ in range(444)] #number of residues 
calc_res_depths(aa_resi_loc_108e_n15,
                '/home/bat-gpu/colleen/phos_interactions/final_traj/108e_n15/analysis_input.pdb',
                '/home/bat-gpu/colleen/phos_interactions/final_traj/108e_n15/final_md.dcd')    

aa_resi_loc_108e_ppm_1us =[[] for _ in range(444)] #number of residues 
calc_res_depths(aa_resi_loc_108e_ppm_1us,
                '/home/bat-gpu/colleen/phos_interactions/final_traj/108e_ppm_1us/analysis_input.pdb',
                '/home/bat-gpu/colleen/phos_interactions/final_traj/108e_ppm_1us/final_md.dcd')    


# In[27]:


#aggregate pgzl1 AA interaction profile - collect counts from each trajectory into one list 


# In[37]:


#aggregate pgzl1 AA interaction profile - collect counts from each trajectory into one list 
aa_resi_loc_counts_pgzl1_aggregate =[[] for _ in range(441)] 
for i in range(len(aa_resi_loc_pgzl1_ppm_1us)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_pgzl1_ppm_1us[i].count(1)
    aa_resi_loc_counts_pgzl1_aggregate[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_pgzl1_ppm_1us[i].count(2)
    aa_resi_loc_counts_pgzl1_aggregate[i].append(glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_pgzl1_ppm_1us[i].count(3)
    aa_resi_loc_counts_pgzl1_aggregate[i].append(po4_counts)
    #water
    water_counts = aa_resi_loc_pgzl1_ppm_1us[i].count(4)
    aa_resi_loc_counts_pgzl1_aggregate[i].append(water_counts)
print(aa_resi_loc_counts_pgzl1_aggregate[318])   
for i in range(len(aa_resi_loc_pgzl1_ppm)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_pgzl1_ppm[i].count(1)
    aa_resi_loc_counts_pgzl1_aggregate[i][0] =aa_resi_loc_counts_pgzl1_aggregate[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_pgzl1_ppm[i].count(2)
    aa_resi_loc_counts_pgzl1_aggregate[i][1] =aa_resi_loc_counts_pgzl1_aggregate[i][1] + (glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_pgzl1_ppm[i].count(3)
    aa_resi_loc_counts_pgzl1_aggregate[i][2] =aa_resi_loc_counts_pgzl1_aggregate[i][2] + (po4_counts)
    #water
    water_counts = aa_resi_loc_pgzl1_ppm[i].count(4)
    aa_resi_loc_counts_pgzl1_aggregate[i][3] =aa_resi_loc_counts_pgzl1_aggregate[i][3] + (water_counts)
for i in range(len(aa_resi_loc_pgzl1_p15)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_pgzl1_p15[i].count(1)
    aa_resi_loc_counts_pgzl1_aggregate[i][0] =aa_resi_loc_counts_pgzl1_aggregate[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_pgzl1_p15[i].count(2)
    aa_resi_loc_counts_pgzl1_aggregate[i][1] =aa_resi_loc_counts_pgzl1_aggregate[i][1] + (glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_pgzl1_p15[i].count(3)
    aa_resi_loc_counts_pgzl1_aggregate[i][2] =aa_resi_loc_counts_pgzl1_aggregate[i][2] + (po4_counts)
    #water
    water_counts = aa_resi_loc_pgzl1_p15[i].count(4)
    aa_resi_loc_counts_pgzl1_aggregate[i][3] =aa_resi_loc_counts_pgzl1_aggregate[i][3] + (water_counts)
    
for i in range(len(aa_resi_loc_pgzl1_n15)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_pgzl1_n15[i].count(1)
    aa_resi_loc_counts_pgzl1_aggregate[i][0] =aa_resi_loc_counts_pgzl1_aggregate[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_pgzl1_n15[i].count(2)
    aa_resi_loc_counts_pgzl1_aggregate[i][1] =aa_resi_loc_counts_pgzl1_aggregate[i][1] + (glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_pgzl1_n15[i].count(3)
    aa_resi_loc_counts_pgzl1_aggregate[i][2] =aa_resi_loc_counts_pgzl1_aggregate[i][2] + (po4_counts)
    #water
    water_counts = aa_resi_loc_pgzl1_n15[i].count(4)
    aa_resi_loc_counts_pgzl1_aggregate[i][3] =aa_resi_loc_counts_pgzl1_aggregate[i][3] + (water_counts)
    


# In[45]:


#aggregate 10e8 AA interaction profile - collect counts from each trajectory into one list 
aa_resi_loc_counts_10e8_aggregate =[[] for _ in range(444)] 
for i in range(len(aa_resi_loc_108e_ppm_1us)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_108e_ppm_1us[i].count(1)
    aa_resi_loc_counts_10e8_aggregate[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_108e_ppm_1us[i].count(2)
    aa_resi_loc_counts_10e8_aggregate[i].append(glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_108e_ppm_1us[i].count(3)
    aa_resi_loc_counts_10e8_aggregate[i].append(po4_counts)
    #water
    water_counts = aa_resi_loc_108e_ppm_1us[i].count(4)
    aa_resi_loc_counts_10e8_aggregate[i].append(water_counts)
print(aa_resi_loc_counts_10e8_aggregate[318])   
for i in range(len(aa_resi_loc_108e_ppm)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_108e_ppm[i].count(1)
    aa_resi_loc_counts_10e8_aggregate[i][0] =aa_resi_loc_counts_10e8_aggregate[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_108e_ppm[i].count(2)
    aa_resi_loc_counts_10e8_aggregate[i][1] =aa_resi_loc_counts_10e8_aggregate[i][1] + (glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_108e_ppm[i].count(3)
    aa_resi_loc_counts_10e8_aggregate[i][2] =aa_resi_loc_counts_10e8_aggregate[i][2] + (po4_counts)
    #water
    water_counts = aa_resi_loc_108e_ppm[i].count(4)
    aa_resi_loc_counts_10e8_aggregate[i][3] =aa_resi_loc_counts_10e8_aggregate[i][3] + (water_counts)
for i in range(len(aa_resi_loc_108e_p15)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_108e_p15[i].count(1)
    aa_resi_loc_counts_10e8_aggregate[i][0] =aa_resi_loc_counts_10e8_aggregate[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_108e_p15[i].count(2)
    aa_resi_loc_counts_10e8_aggregate[i][1] =aa_resi_loc_counts_10e8_aggregate[i][1] + (glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_108e_p15[i].count(3)
    aa_resi_loc_counts_10e8_aggregate[i][2] =aa_resi_loc_counts_10e8_aggregate[i][2] + (po4_counts)
    #water
    water_counts = aa_resi_loc_108e_p15[i].count(4)
    aa_resi_loc_counts_10e8_aggregate[i][3] =aa_resi_loc_counts_10e8_aggregate[i][3] + (water_counts)
    
for i in range(len(aa_resi_loc_108e_n15)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_108e_n15[i].count(1)
    aa_resi_loc_counts_10e8_aggregate[i][0] =aa_resi_loc_counts_10e8_aggregate[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_108e_n15[i].count(2)
    aa_resi_loc_counts_10e8_aggregate[i][1] =aa_resi_loc_counts_10e8_aggregate[i][1] + (glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_108e_n15[i].count(3)
    aa_resi_loc_counts_10e8_aggregate[i][2] =aa_resi_loc_counts_10e8_aggregate[i][2] + (po4_counts)
    #water
    water_counts = aa_resi_loc_108e_n15[i].count(4)
    aa_resi_loc_counts_10e8_aggregate[i][3] =aa_resi_loc_counts_10e8_aggregate[i][3] + (water_counts)
    


# In[17]:


# with open('pgzl1_AA_interaction_profile.pickle', 'wb') as handle:
#     pickle.dump(aa_resi_loc_counts_pgzl1_aggregate,
#                 handle,
#                 protocol=pickle.HIGHEST_PROTOCOL)
    
    # Load data (deserialize)
with open('pgzl1_AA_interaction_profile.pickle', 'rb') as handle:
    interaction_prof_pgzl1_AA = pickle.load(handle)
    
interaction_prof_pgzl1_AA = np.array(interaction_prof_pgzl1_AA)


# In[25]:


import pickle
# with open('10e8_AA_interaction_profile.pickle', 'wb') as handle:

#     pickle.dump(aa_resi_loc_counts_10e8_aggregate,
#                 handle,
#                 protocol=pickle.HIGHEST_PROTOCOL)
    
    # Load data (deserialize)
with open('10e8_AA_interaction_profile.pickle', 'rb') as handle:
    interaction_prof_10e8_AA = pickle.load(handle)
    
interaction_prof_10e8_AA = np.array(interaction_prof_10e8_AA)


# In[51]:



def plot_interaction_profile_10e8(acyl_loc, glycerol_loc, po4_loc, chain_residues, prefix): 
    fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 8), gridspec_kw={'height_ratios': [.5, 6]})
    plt.subplots_adjust(hspace=0.01)
    #x = np.arange(0, len(l_chain_residues),1)
    xticks = [i for i in range(0,len(chain_residues))]
    #add cdr loop annotation 
    l_chain_annotate = [] 
    for i in range(len(chain_residues)):
        #add cdrl1 loop 
        if i in range(25, 30):
            l_chain_annotate.append(1)
        elif i in range(48,50):
            l_chain_annotate.append(1)
        elif i in range(87,98):
            l_chain_annotate.append(1)
        #add H chain loops 
        elif i in range(237, 245):
            l_chain_annotate.append(2)
        elif i in range(262,272):
            l_chain_annotate.append(2)
        elif i in range(310,332):
            l_chain_annotate.append(2)        
        else:
            l_chain_annotate.append(0)

    ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==1, 
                     alpha=1, color='#c28ee8') 
    
    ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==2, 
                    alpha=1, color='#460273') 
    ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==0, 
                    alpha=.5, color='#BFBFBF')

    ax1.axhline(y=-10, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
    ax1.axhline(y=0, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
    #mark antibody chain start/stop 
    ax1.axvline(x=221, color='#000000', linestyle='-', ymin=0, linewidth=1.5)
    ax1.axvline(x=444, color='#000000', linestyle='-', ymin=0, linewidth=1.5)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.set_xticks([])
    ax1.set_yticks([])

    width=1
    ax2.bar(xticks, acyl_loc, width, label='Acyl', color='blue')
    ax2.bar(xticks, glycerol_loc, width, bottom=acyl_loc, label='Glycerol', color='red')
    ax2.bar(xticks, po4_loc, width, bottom=glycerol_loc, label='PO4', color='orange')
    
   
    ax2.set_xticks([]) #xticks
    ax2.set_xticklabels([])#l_chain_residues
    ax2.set_yticks([0, 1]) #5033, 10066, 15099, 20123
    #ax2.set_yticklabels(['0', '1', '2', '3', '4'])
    ax2.tick_params(axis='x', labelsize=50)
    ax2.tick_params(axis='y', labelsize=50)#, fontsize=16
    ax2.set_ylabel(r"Time ($\rm \mu s$)", fontsize=54)
    #ax2.set_xlabel('Residue', fontsize=54)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    
    
    ax1.set_xlim(0, len(xticks))
    ax2.set_xlim(0, len(xticks))
    
    fig_name = prefix + '_interaction_profile.png' 
    plt.savefig(fig_name, transparent=True, bbox_inches="tight")

    plt.show()
    return "Made figure: "#, prefix


# In[57]:



def plot_interaction_profile_10e8_v2(acyl_loc, glycerol_loc, po4_loc, chain_residues, prefix): 
    fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 8), gridspec_kw={'height_ratios': [.5, 6]})
    plt.subplots_adjust(hspace=0.01)
    #x = np.arange(0, len(l_chain_residues),1)
    xticks = [i for i in range(0,len(chain_residues))]
    #add cdr loop annotation 
    l_chain_annotate = [] 
    for i in range(len(chain_residues)):
        #add cdrl1 loop 
        if i in range(25, 33):
            l_chain_annotate.append(1)
        elif i in range(48,50):
            l_chain_annotate.append(1)
        elif i in range(86,98):
            l_chain_annotate.append(1)
        #add H chain loops 
        elif i in range(238, 246):
            l_chain_annotate.append(2)
        elif i in range(262,273):
            l_chain_annotate.append(2)
        elif i in range(310,331):
            l_chain_annotate.append(2)        
        else:
            l_chain_annotate.append(0)

    ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==1, 
                     alpha=1, color='#c28ee8') 
    
    ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==2, 
                    alpha=1, color='#460273') 
    ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==0, 
                    alpha=.5, color='#BFBFBF')

    ax1.axhline(y=-10, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
    ax1.axhline(y=0, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
    #mark antibody chain start/stop 
    ax1.axvline(x=221, color='#000000', linestyle='-', ymin=0, linewidth=1.5)
    ax1.axvline(x=444, color='#000000', linestyle='-', ymin=0, linewidth=1.5)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.set_xticks([])
    ax1.set_yticks([])

    width=1
    ax2.bar(xticks, acyl_loc, width, label='Acyl', color='blue')
    ax2.bar(xticks, glycerol_loc, width, bottom=acyl_loc, label='Glycerol', color='red')
    ax2.bar(xticks, po4_loc, width, bottom=glycerol_loc, label='PO4', color='orange')
    
   
    ax2.set_xticks([]) #xticks
    ax2.set_xticklabels([])#l_chain_residues
    ax2.set_yticks([0, 1]) #5033, 10066, 15099, 20123
    #ax2.set_yticklabels(['0', '1', '2', '3', '4'])
    ax2.tick_params(axis='x', labelsize=50)
    ax2.tick_params(axis='y', labelsize=50)#, fontsize=16
    ax2.set_ylabel(r"Time ($\rm \mu s$)", fontsize=54)
    #ax2.set_xlabel('Residue', fontsize=54)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    
    
    ax1.set_xlim(0, len(xticks))
    ax2.set_xlim(0, len(xticks))
    
    fig_name = prefix + '_interaction_profile.png' 
    plt.savefig(fig_name, transparent=True, bbox_inches="tight")

    plt.show()
    return "Made figure: "#, prefix


# In[26]:


l_chain = interaction_prof_10e8_AA[233:445]
h_chain = interaction_prof_10e8_AA[0:233]
interaction_prof_10e8_AA_final = [] 
for i in l_chain:
    interaction_prof_10e8_AA_final.append(i)
for i in h_chain:
    interaction_prof_10e8_AA_final.append(i)
print(len(interaction_prof_10e8_AA))    
print(len(interaction_prof_10e8_AA_final))
interaction_prof_10e8_AA_final = np.array(interaction_prof_10e8_AA_final)


# In[29]:


#get list of residues in l chain 4e10 
chain_10e8_seq = "SYELTQETGVSVALGRTVTITCRGDSLRSHYASWYQKKPGQAPILLFYGKNNRPSGVPDRFSGSASGNRASLTISGAQAEDDAEYYCSSRDKSGSRLSVFGGGTKLTVLSQPKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPVKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTEVQLVESGGGLVKPGGSLRLSCSASGFDFDNAWMTWVRQPPGKGLEWVGRITGPGEGWSVDYAAPVEGRFTISRLNSINFLYLEMNNLRMEDSGLYFCARTGKYYDFWSGYPPGEEYFQDWGRGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPK"

#chain_10e8_seq ="SYELTQETGVSVALGRTVTITCRGDSLRSHYASWYQKKPGQAPILLFYGKNNRPSGVPDRFSGSASGNRASLTISGAQAEDDAEYYCSSRDKSGSRLSVFGGGTKLTVLSQPKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPVKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTECSEVQLVESGGGLVKPGGSLRLSCSASGFDFDNAWMTWVRQPPGKGLEWVGRITGPGEGWSVDYAAPVEGRFTISRLNSINFLYLEMNNLRMEDSGLYFCARTGKYYDFWSGYPPGEEYFQDWGRGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSCDK"


chain_residues = split_seq2list(chain_10e8_seq)
print(len(chain_10e8_seq))
plot_interaction_profile_10e8(interaction_prof_10e8_AA_final[:,0],
                         interaction_prof_10e8_AA_final[:,1],
                         interaction_prof_10e8_AA_final[:,2],
                         chain_10e8_seq, '10e8_AA')



# In[44]:



def plot_interaction_profile_pgzl1(acyl_loc, glycerol_loc, po4_loc, chain_residues, prefix): 
fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 8), gridspec_kw={'height_ratios': [.5, 6]})
plt.subplots_adjust(hspace=0.01)
#x = np.arange(0, len(l_chain_residues),1)
xticks = [i for i in range(0,len(chain_residues))]
#add cdr loop annotation 
l_chain_annotate = [] 
for i in range(len(chain_residues)):
    #add cdrl1 loop 
    if i in range(26, 33):
        l_chain_annotate.append(1)
    elif i in range(49,53):
        l_chain_annotate.append(1)
    elif i in range(89,98):
        l_chain_annotate.append(1)
    #add H chain loops 
    elif i in range(239, 249):
        l_chain_annotate.append(2)
    elif i in range(264,272):
        l_chain_annotate.append(2)
    elif i in range(310,325):
        l_chain_annotate.append(2)        
    else:
        l_chain_annotate.append(0)

ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==1, 
                 alpha=1, color='#f79a99') 

ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==2, 
                alpha=1, color='#a33634') 
ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==0, 
                alpha=.5, color='#BFBFBF')

ax1.axhline(y=-10, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
ax1.axhline(y=0, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
#mark antibody chain start/stop 
ax1.axvline(x=214, color='#000000', linestyle='-', ymin=0, linewidth=1.5)
ax1.axvline(x=441, color='#000000', linestyle='-', ymin=0, linewidth=1.5)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax1.set_xticks([])
ax1.set_yticks([])

width=1
ax2.bar(xticks, acyl_loc, width, label='Acyl', color='blue')
ax2.bar(xticks, glycerol_loc, width, bottom=acyl_loc, label='Glycerol', color='red')
ax2.bar(xticks, po4_loc, width, bottom=glycerol_loc+acyl_loc, label='PO4', color='orange')

   
ax2.set_xticks([]) #xticks
ax2.set_xticklabels([])#l_chain_residues
ax2.set_yticks([0, 1]) #5033, 10066, 15099, 20123
ax2.set_yticklabels(['0', '1']) #'2', '3', '4'
ax2.tick_params(axis='x', labelsize=50)
ax2.tick_params(axis='y', labelsize=50)#, fontsize=16
ax2.set_ylabel(r"Time ($\rm \mu s$)", fontsize=54)
#ax2.set_xlabel('Residue', fontsize=40)
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)


ax1.set_xlim(0, len(xticks))
ax2.set_xlim(0, len(xticks))

fig_name = prefix + '_interaction_profile.png' 
plt.savefig(fig_name, transparent=True, bbox_inches="tight")

plt.show()
return "Made figure: "#, prefix


# In[7]:



def plot_interaction_profile_pgzl1(acyl_loc, glycerol_loc, po4_loc, chain_residues, prefix): 
fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 8), gridspec_kw={'height_ratios': [.5, 6]})
plt.subplots_adjust(hspace=0.01)
#x = np.arange(0, len(l_chain_residues),1)
xticks = [i for i in range(0,len(chain_residues))]
#add cdr loop annotation 
l_chain_annotate = [] 
for i in range(len(chain_residues)):
    #add cdrl1 loop 
    if i in range(26, 33):
        l_chain_annotate.append(1)
    elif i in range(49,53):
        l_chain_annotate.append(1)
    elif i in range(89,98):
        l_chain_annotate.append(1)
    #add H chain loops 
    elif i in range(239, 249):
        l_chain_annotate.append(2)
    elif i in range(264,272):
        l_chain_annotate.append(2)
    elif i in range(310,325):
        l_chain_annotate.append(2)        
    else:
        l_chain_annotate.append(0)

ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==1, 
                 alpha=1, color='#f79a99') 

ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==2, 
                alpha=1, color='#a33634') 
ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==0, 
                alpha=.5, color='#BFBFBF')

ax1.axhline(y=-10, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
ax1.axhline(y=0, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
#mark antibody chain start/stop 
ax1.axvline(x=214, color='#000000', linestyle='-', ymin=0, linewidth=1.5)
ax1.axvline(x=441, color='#000000', linestyle='-', ymin=0, linewidth=1.5)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax1.set_xticks([])
ax1.set_yticks([])

width=1
ax2.bar(xticks, acyl_loc, width, label='Acyl', color='blue')
ax2.bar(xticks, glycerol_loc, width, bottom=acyl_loc, label='Glycerol', color='red')
ax2.bar(xticks, po4_loc, width, bottom=glycerol_loc+acyl_loc, label='PO4', color='orange')

   
ax2.set_xticks([]) #xticks
ax2.set_xticklabels([])#l_chain_residues
ax2.set_yticks([0, 1]) #5033, 10066, 15099, 20123
ax2.set_yticklabels(['0', '1']) #'2', '3', '4'
ax2.tick_params(axis='x', labelsize=50)
ax2.tick_params(axis='y', labelsize=50)#, fontsize=16
ax2.set_ylabel(r"Time ($\rm \mu s$)", fontsize=54)
#ax2.set_xlabel('Residue', fontsize=40)
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)


ax1.set_xlim(0, len(xticks))
ax2.set_xlim(0, len(xticks))

fig_name = prefix + '_interaction_profile.png' 
plt.savefig(fig_name, transparent=True, bbox_inches="tight")

plt.show()
return "Made figure: "#, prefix


# In[17]:



def plot_interaction_profile_pgzl1_variant(acyl_loc, glycerol_loc, po4_loc, chain_residues, prefix): 
fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 8), gridspec_kw={'height_ratios': [.5, 6]})
plt.subplots_adjust(hspace=0.01)
#x = np.arange(0, len(l_chain_residues),1)
xticks = [i for i in range(0,len(chain_residues))]
#add cdr loop annotation 
l_chain_annotate = [] 
for i in range(len(chain_residues)):
    #add cdrl1 loop 
    if i in range(25, 31):
        l_chain_annotate.append(1)
    elif i in range(49,52):
        l_chain_annotate.append(1)
    elif i in range(88,96):
        l_chain_annotate.append(1)
    #add H chain loops 
    elif i in range(253, 260): #chain H starts at 227 #26-33
        l_chain_annotate.append(2)
    elif i in range(276,285): #49-58
        l_chain_annotate.append(2)
    elif i in range(324,339): #97-112 
        l_chain_annotate.append(2)        
    else:
        l_chain_annotate.append(0)

ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==1, 
                 alpha=1, color='#f79a99') 

ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==2, 
                alpha=1, color='#a33634') 
ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==0, 
                alpha=.5, color='#BFBFBF')

ax1.axhline(y=-10, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
ax1.axhline(y=0, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
#mark antibody chain start/stop 
ax1.axvline(x=214, color='#000000', linestyle='-', ymin=0, linewidth=1.5)
ax1.axvline(x=439, color='#000000', linestyle='-', ymin=0, linewidth=1.5)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax1.set_xticks([])
ax1.set_yticks([])

width=1
ax2.bar(xticks, acyl_loc, width, label='Acyl', color='blue')
ax2.bar(xticks, glycerol_loc, width, bottom=acyl_loc, label='Glycerol', color='red')
ax2.bar(xticks, po4_loc, width, bottom=glycerol_loc+acyl_loc, label='PO4', color='orange')

   
ax2.set_xticks([]) #xticks
ax2.set_xticklabels([])#l_chain_residues
ax2.set_yticks([0, 1]) #5033, 10066, 15099, 20123
ax2.set_yticklabels(['0', '1']) #'2', '3', '4'
ax2.tick_params(axis='x', labelsize=50)
ax2.tick_params(axis='y', labelsize=50)#, fontsize=16
ax2.set_ylabel(r"Time ($\rm \mu s$)", fontsize=54)
#ax2.set_xlabel('Residue', fontsize=40)
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)


ax1.set_xlim(0, len(xticks))
ax2.set_xlim(0, len(xticks))

fig_name = prefix + '_interaction_profile.png' 
plt.savefig(fig_name, transparent=True, bbox_inches="tight")

plt.show()
return "Made figure: "#, prefix


# In[17]:



def plot_interaction_profile_pgzl1_test(acyl_loc, glycerol_loc, po4_loc, chain_residues, prefix): 
fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 8), gridspec_kw={'height_ratios': [.5, 6]})
plt.subplots_adjust(hspace=0.01)
#x = np.arange(0, len(l_chain_residues),1)
xticks = [i for i in range(0,len(chain_residues))]
#add cdr loop annotation 
l_chain_annotate = [] 
for i in range(len(chain_residues)):
    #add cdrl1 loop 
    if i in range(26, 33):
        l_chain_annotate.append(1)
    elif i in range(49,53):
        l_chain_annotate.append(1)
    elif i in range(89,98):
        l_chain_annotate.append(1)
    #add H chain loops 
    elif i in range(239, 249):
        l_chain_annotate.append(2)
    elif i in range(264,272):
        l_chain_annotate.append(2)
    elif i in range(310,325):
        l_chain_annotate.append(2)        
    else:
        l_chain_annotate.append(0)

ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==1, 
                 alpha=1, color='#f79a99') 

ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==2, 
                alpha=1, color='#a33634') 
ax1.fill_between(xticks, -10, 0, where=np.array(l_chain_annotate)==0, 
                alpha=.5, color='#BFBFBF')

ax1.axhline(y=-10, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
ax1.axhline(y=0, color='#000000', linestyle='-', xmin=0, linewidth=1.5)
#mark antibody chain start/stop 
ax1.axvline(x=214, color='#000000', linestyle='-', ymin=0, linewidth=1.5)
ax1.axvline(x=441, color='#000000', linestyle='-', ymin=0, linewidth=1.5)
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax1.set_xticks([])
ax1.set_yticks([])

width=1
#ax2.bar(xticks, acyl_loc, width, label='Acyl', color='blue')
ax2.bar(xticks, glycerol_loc, width,  label='Glycerol', color='red')
ax2.bar(xticks, po4_loc, width, bottom=glycerol_loc, label='PO4', color='orange')

   
ax2.set_xticks([]) #xticks
ax2.set_xticklabels([])#l_chain_residues
ax2.set_yticks([0, 1]) #5033, 10066, 15099, 20123
ax2.set_yticklabels(['0', '1']) #'2', '3', '4'
ax2.tick_params(axis='x', labelsize=50)
ax2.tick_params(axis='y', labelsize=50)#, fontsize=16
ax2.set_ylabel(r"Time ($\rm \mu s$)", fontsize=54)
#ax2.set_xlabel('Residue', fontsize=40)
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)


ax1.set_xlim(0, len(xticks))
ax2.set_xlim(0, len(xticks))

fig_name = prefix + '_interaction_profile.png' 
plt.savefig(fig_name, transparent=True, bbox_inches="tight")

plt.show()
return "Made figure: "#, prefix


# In[ ]:





# In[23]:


#get list of residues in l chain 4e10 
chain_pgzl1_seq = 'DVVMTQSPGTLSLSPGERATLSCRASQSVSGGALAWYQQKPGQAPRLLIYDTSSRPTGVPGRFSGSGSGTDFSLTISRLEPEDFAVYYCQQYGTSQSTFGQGTRLEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEEVQLVQSGGEVKRPGSSVTVSCKATGGTFSTLAFNWVRQAPGQGPEWMGGIVPLFSIVNYGQKFQGRLTIRADKSTTTVFLDLSGLTSADTATYYCAREGEGWFGKPLRAFEFWGQGTVITVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'


#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC'
chain_residues = split_seq2list(chain_pgzl1_seq)
print(len(chain_pgzl1_seq))
plot_interaction_profile_pgzl1(interaction_prof_pgzl1_AA[:,0],
                         interaction_prof_pgzl1_AA[:,1],
                         interaction_prof_pgzl1_AA[:,2],
                         chain_pgzl1_seq, 'pgzl1_AA')


# In[13]:


aa_resi_loc_pgzl1_cg_revert_med1 =[[] for _ in range(437)] #number of residues 
calc_res_depths(aa_resi_loc_pgzl1_cg_revert_med1,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/pgzl1/medoid1/analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/pgzl1/medoid1/traj_500ns.dcd')  


# In[21]:


aa_resi_loc_pgzl1_cg_revert_med2 =[[] for _ in range(437)] #number of residues 
calc_res_depths(aa_resi_loc_pgzl1_cg_revert_med2,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/pgzl1/medoid2/analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/pgzl1/medoid2/traj_1000ns.dcd')  


# In[36]:


aa_resi_loc_pgzl1_int_ppm =[[] for _ in range(439)] #number of residues 
calc_res_depths(aa_resi_loc_pgzl1_int_ppm,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_germ_ppm/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_germ_ppm/final_analysis_traj.dcd')    

aa_resi_loc_counts_pgzl1_int_aggregate =[[] for _ in range(439)] 
for i in range(len(aa_resi_loc_pgzl1_int_ppm)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_pgzl1_int_ppm[i].count(1)
    aa_resi_loc_counts_pgzl1_int_aggregate[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_pgzl1_int_ppm[i].count(2)
    aa_resi_loc_counts_pgzl1_int_aggregate[i].append(glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_pgzl1_int_ppm[i].count(3)
    aa_resi_loc_counts_pgzl1_int_aggregate[i].append(po4_counts)
    #water
    water_counts = aa_resi_loc_pgzl1_int_ppm[i].count(4)
    aa_resi_loc_counts_pgzl1_int_aggregate[i].append(water_counts)

aa_resi_loc_counts_pgzl1_int_aggregate = np.array(aa_resi_loc_counts_pgzl1_int_aggregate)

l_chain = aa_resi_loc_counts_pgzl1_int_aggregate[226:445]
h_chain = aa_resi_loc_counts_pgzl1_int_aggregate[0:226]
interaction_prof_pgzl1_int_final = [] 
for i in l_chain:
    interaction_prof_pgzl1_int_final.append(i)
for i in h_chain:
    interaction_prof_pgzl1_int_final.append(i)
print(len(aa_resi_loc_counts_pgzl1_int_aggregate))    
print(len(interaction_prof_pgzl1_int_final))
interaction_prof_pgzl1_int_final = np.array(interaction_prof_pgzl1_int_final)





#get list of residues in l chain pgzl1 
l_chain_pgzl1_seq = 'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKAA'
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGECQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'
l_chain_residues = split_seq2list(l_chain_pgzl1_seq)
print(len(l_chain_residues))
plot_interaction_profile_pgzl1_variant(interaction_prof_pgzl1_int_final[:,0],
                         interaction_prof_pgzl1_int_final[:,1],
                         interaction_prof_pgzl1_int_final[:,2],
                         l_chain_residues, 'pgzl1_int')


# In[38]:


counter = 0 
for i in interaction_prof_pgzl1_int_final:
    counter +=1 
    print(counter, i)


# In[15]:


aa_resi_loc_pgzl1_cg_revert_med1_count =[[] for _ in range(437)] 
for i in range(len(aa_resi_loc_pgzl1_cg_revert_med1)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_pgzl1_cg_revert_med1[i].count(1)
    aa_resi_loc_pgzl1_cg_revert_med1_count[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_pgzl1_cg_revert_med1[i].count(2)
    aa_resi_loc_pgzl1_cg_revert_med1_count[i].append(glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_pgzl1_cg_revert_med1[i].count(3)
    aa_resi_loc_pgzl1_cg_revert_med1_count[i].append(po4_counts)
    #water
    water_counts = aa_resi_loc_pgzl1_cg_revert_med1[i].count(4)
    aa_resi_loc_pgzl1_cg_revert_med1_count[i].append(water_counts)

aa_resi_loc_pgzl1_cg_revert_med1_count = np.array(aa_resi_loc_pgzl1_cg_revert_med1_count)

l_chain = aa_resi_loc_pgzl1_cg_revert_med1_count[223:445]
h_chain = aa_resi_loc_pgzl1_cg_revert_med1_count[0:223]
interaction_prof_pgzl1_med1_final = [] 
for i in l_chain:
    interaction_prof_pgzl1_med1_final.append(i)
for i in h_chain:
    interaction_prof_pgzl1_med1_final.append(i)
print(len(aa_resi_loc_pgzl1_cg_revert_med1))    
print(len(interaction_prof_pgzl1_med1_final))
interaction_prof_pgzl1_med1_final = np.array(interaction_prof_pgzl1_med1_final)





#get list of residues in l chain pgzl1 
l_chain_pgzl1_seq = 'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDK'
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGECQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'
l_chain_residues = split_seq2list(l_chain_pgzl1_seq)
print(len(l_chain_residues))
plot_interaction_profile_pgzl1(interaction_prof_pgzl1_med1_final[:,0],
                         interaction_prof_pgzl1_med1_final[:,1],
                         interaction_prof_pgzl1_med1_final[:,2],
                         l_chain_residues, 'pgzl1_cgmed1_revert')


# In[22]:


aa_resi_loc_pgzl1_cg_revert_med2_count =[[] for _ in range(437)] 
for i in range(len(aa_resi_loc_pgzl1_cg_revert_med2)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_pgzl1_cg_revert_med2[i].count(1)
    aa_resi_loc_pgzl1_cg_revert_med2_count[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_pgzl1_cg_revert_med2[i].count(2)
    aa_resi_loc_pgzl1_cg_revert_med2_count[i].append(glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_pgzl1_cg_revert_med2[i].count(3)
    aa_resi_loc_pgzl1_cg_revert_med2_count[i].append(po4_counts)
    #water
    water_counts = aa_resi_loc_pgzl1_cg_revert_med2[i].count(4)
    aa_resi_loc_pgzl1_cg_revert_med2_count[i].append(water_counts)

aa_resi_loc_pgzl1_cg_revert_med2_count = np.array(aa_resi_loc_pgzl1_cg_revert_med2_count)

l_chain = aa_resi_loc_pgzl1_cg_revert_med2_count[223:445]
h_chain = aa_resi_loc_pgzl1_cg_revert_med2_count[0:223]
interaction_prof_pgzl1_med2_final = [] 
for i in l_chain:
    interaction_prof_pgzl1_med2_final.append(i)
for i in h_chain:
    interaction_prof_pgzl1_med2_final.append(i)
print(len(aa_resi_loc_pgzl1_cg_revert_med2))    
print(len(interaction_prof_pgzl1_med2_final))
interaction_prof_pgzl1_med2_final = np.array(interaction_prof_pgzl1_med2_final)


#get list of residues in l chain pgzl1 
l_chain_pgzl1_seq = 'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDK'
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGECQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'
l_chain_residues = split_seq2list(l_chain_pgzl1_seq)
print(len(l_chain_residues))
plot_interaction_profile_pgzl1(interaction_prof_pgzl1_med2_final[:,0],
                         interaction_prof_pgzl1_med2_final[:,1],
                         interaction_prof_pgzl1_med2_final[:,2],
                         l_chain_residues, 'pgzl1_cgmed2_revert')


# In[36]:


aa_resi_loc_10e8_cg_revert_med1 =[[] for _ in range(444)] #number of residues 
calc_res_depths(aa_resi_loc_10e8_cg_revert_med1,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/10e8/medoid1/analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/10e8/medoid1/traj_500ns.dcd')  


# In[43]:


aa_resi_loc_10e8_cg_revert_med2 =[[] for _ in range(444)] #number of residues 
calc_res_depths(aa_resi_loc_10e8_cg_revert_med2,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/10e8/medoid2/analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/10e8/medoid2/traj_500ns.dcd')  


# In[35]:


aa_resi_loc_10e8_cg_revert_med3 =[[] for _ in range(444)] #number of residues 
calc_res_depths(aa_resi_loc_10e8_cg_revert_med3,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/10e8/medoid3/analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/10e8/medoid3/traj_460ns.dcd')  


# In[39]:


aa_resi_loc_10e8_cg_revert_med3_count =[[] for _ in range(444)] 
for i in range(len(aa_resi_loc_10e8_cg_revert_med3)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_10e8_cg_revert_med3[i].count(1)
    aa_resi_loc_10e8_cg_revert_med3_count[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_10e8_cg_revert_med3[i].count(2)
    aa_resi_loc_10e8_cg_revert_med3_count[i].append(glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_10e8_cg_revert_med3[i].count(3)
    aa_resi_loc_10e8_cg_revert_med3_count[i].append(po4_counts)
    #water
    water_counts = aa_resi_loc_10e8_cg_revert_med3[i].count(4)
    aa_resi_loc_10e8_cg_revert_med3_count[i].append(water_counts)

aa_resi_loc_10e8_cg_revert_med3_count = np.array(aa_resi_loc_10e8_cg_revert_med3_count)

l_chain = aa_resi_loc_10e8_cg_revert_med3_count[233:445]
h_chain = aa_resi_loc_10e8_cg_revert_med3_count[0:233]
interaction_prof_10e8_med3_final = [] 
for i in l_chain:
    interaction_prof_10e8_med3_final.append(i)
for i in h_chain:
    interaction_prof_10e8_med3_final.append(i)
print(len(aa_resi_loc_10e8_cg_revert_med3))    
print(len(interaction_prof_10e8_med3_final))
interaction_prof_10e8_med3_final = np.array(interaction_prof_10e8_med3_final)



#get list of residues in l chain 10e8 
l_chain_10e8_seq = 'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDEASBDDDD'
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGECQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'
l_chain_residues = split_seq2list(l_chain_10e8_seq)
print(len(l_chain_residues))
plot_interaction_profile_10e8_v2(interaction_prof_10e8_med3_final[:,0],
                         interaction_prof_10e8_med3_final[:,1],
                         interaction_prof_10e8_med3_final[:,2],
                         l_chain_residues, '10e8_cgmed3_revert')


# In[52]:


aa_resi_loc_10e8_cg_revert_med1_count =[[] for _ in range(444)] 
for i in range(len(aa_resi_loc_10e8_cg_revert_med1)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_10e8_cg_revert_med1[i].count(1)
    aa_resi_loc_10e8_cg_revert_med1_count[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_10e8_cg_revert_med1[i].count(2)
    aa_resi_loc_10e8_cg_revert_med1_count[i].append(glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_10e8_cg_revert_med1[i].count(3)
    aa_resi_loc_10e8_cg_revert_med1_count[i].append(po4_counts)
    #water
    water_counts = aa_resi_loc_10e8_cg_revert_med1[i].count(4)
    aa_resi_loc_10e8_cg_revert_med1_count[i].append(water_counts)

aa_resi_loc_10e8_cg_revert_med1_count = np.array(aa_resi_loc_10e8_cg_revert_med1_count)

l_chain = aa_resi_loc_10e8_cg_revert_med1_count[233:445]
h_chain = aa_resi_loc_10e8_cg_revert_med1_count[0:233]
interaction_prof_10e8_med1_final = [] 
for i in l_chain:
    interaction_prof_10e8_med1_final.append(i)
for i in h_chain:
    interaction_prof_10e8_med1_final.append(i)
print(len(aa_resi_loc_10e8_cg_revert_med1))    
print(len(interaction_prof_10e8_med1_final))
interaction_prof_10e8_med1_final = np.array(interaction_prof_10e8_med1_final)



#get list of residues in l chain 10e8 
l_chain_10e8_seq = 'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDEASBDDDD'
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGECQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'
l_chain_residues = split_seq2list(l_chain_10e8_seq)
print(len(l_chain_residues))
plot_interaction_profile_10e8_v2(interaction_prof_10e8_med1_final[:,0],
                         interaction_prof_10e8_med1_final[:,1],
                         interaction_prof_10e8_med1_final[:,2],
                         l_chain_residues, '10e8_cgmed1_revert')


# In[53]:


aa_resi_loc_10e8_cg_revert_med2_count =[[] for _ in range(444)] 
for i in range(len(aa_resi_loc_10e8_cg_revert_med2)):
    #acyl location counts ]
    acyl_counts = aa_resi_loc_10e8_cg_revert_med2[i].count(1)
    aa_resi_loc_10e8_cg_revert_med2_count[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = aa_resi_loc_10e8_cg_revert_med2[i].count(2)
    aa_resi_loc_10e8_cg_revert_med2_count[i].append(glycerol_counts)
    #po4 
    po4_counts = aa_resi_loc_10e8_cg_revert_med2[i].count(3)
    aa_resi_loc_10e8_cg_revert_med2_count[i].append(po4_counts)
    #water
    water_counts = aa_resi_loc_10e8_cg_revert_med2[i].count(4)
    aa_resi_loc_10e8_cg_revert_med2_count[i].append(water_counts)

aa_resi_loc_10e8_cg_revert_med2_count = np.array(aa_resi_loc_10e8_cg_revert_med2_count)

l_chain = aa_resi_loc_10e8_cg_revert_med2_count[233:445]
h_chain = aa_resi_loc_10e8_cg_revert_med2_count[0:233]
interaction_prof_10e8_med2_final = [] 
for i in l_chain:
    interaction_prof_10e8_med2_final.append(i)
for i in h_chain:
    interaction_prof_10e8_med2_final.append(i)
print(len(aa_resi_loc_10e8_cg_revert_med2))    
print(len(interaction_prof_10e8_med2_final))
interaction_prof_10e8_med2_final = np.array(interaction_prof_10e8_med2_final)



#get list of residues in l chain 10e8 
l_chain_10e8_seq = 'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDEASBDDDD'
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGECQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'
l_chain_residues = split_seq2list(l_chain_10e8_seq)
print(len(l_chain_residues))
plot_interaction_profile_10e8_v2(interaction_prof_10e8_med2_final[:,0],
                         interaction_prof_10e8_med2_final[:,1],
                         interaction_prof_10e8_med2_final[:,2],
                         l_chain_residues, '10e8_cgmed2_revert')


# In[ ]:





# In[ ]:





# In[6]:


#READ IN 4E10 Cluster 1 

clust1_4e10_ppm =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust1_4e10_ppm,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_ppm/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/4e10_clust1_ppm.dcd')    

clust1_4e10_p15 =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust1_4e10_p15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_p15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/4e10_clust1_p15.dcd')    

clust1_4e10_n15 =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust1_4e10_n15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_n15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/4e10_clust1_n15.dcd')    

clust1_4e10_ppm_rep =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust1_4e10_ppm_rep,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_ppm_rep/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/4e10_clust1_ppm_rep.dcd')


# In[ ]:





# In[7]:


#combine to 4e10_clust1_total 
#aggregate 4e10 AA interaction profile - collect counts from each trajectory into one list 
clust1_total_4e10 =[[] for _ in range(441)] 
for i in range(len(clust1_4e10_ppm)):
    #acyl location counts ]
    acyl_counts = clust1_4e10_ppm[i].count(1)
    clust1_total_4e10[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust1_4e10_ppm[i].count(2)
    clust1_total_4e10[i].append(glycerol_counts)
    #po4 
    po4_counts = clust1_4e10_ppm[i].count(3)
    clust1_total_4e10[i].append(po4_counts)
    #water
    water_counts = clust1_4e10_ppm[i].count(4)
    clust1_total_4e10[i].append(water_counts)
#print(aa_resi_loc_counts_4e10_aggregate[318])   
for i in range(len(clust1_4e10_p15)):
    #acyl location counts ]
    acyl_counts = clust1_4e10_p15[i].count(1)
    clust1_total_4e10[i][0] =clust1_total_4e10[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust1_4e10_p15[i].count(2)
    clust1_total_4e10[i][1] =clust1_total_4e10[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust1_4e10_p15[i].count(3)
    clust1_total_4e10[i][2] =clust1_total_4e10[i][2] + (po4_counts)
    #water
    water_counts = clust1_4e10_p15[i].count(4)
    clust1_total_4e10[i][3] =clust1_total_4e10[i][3] + (water_counts)
for i in range(len(clust1_4e10_n15)):
    #acyl location counts ]
    acyl_counts = clust1_4e10_n15[i].count(1)
    clust1_total_4e10[i][0] =clust1_total_4e10[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust1_4e10_n15[i].count(2)
    clust1_total_4e10[i][1] =clust1_total_4e10[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust1_4e10_n15[i].count(3)
    clust1_total_4e10[i][2] =clust1_total_4e10[i][2] + (po4_counts)
    #water
    water_counts = clust1_4e10_n15[i].count(4)
    clust1_total_4e10[i][3] =clust1_total_4e10[i][3] + (water_counts)
    
for i in range(len(clust1_4e10_ppm_rep)):
    #acyl location counts ]
    acyl_counts = clust1_4e10_ppm_rep[i].count(1)
    clust1_total_4e10[i][0] =clust1_total_4e10[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust1_4e10_ppm_rep[i].count(2)
    clust1_total_4e10[i][1] =clust1_total_4e10[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust1_4e10_ppm_rep[i].count(3)
    clust1_total_4e10[i][2] =clust1_total_4e10[i][2] + (po4_counts)
    #water
    water_counts = clust1_4e10_ppm_rep[i].count(4)
    clust1_total_4e10[i][3] =clust1_total_4e10[i][3] + (water_counts)
    
  


# In[23]:


print(len(clust1_total_4e10[]))


# In[26]:


l_chain_4e10_seq = 'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEP'
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGECQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'
clust1_total_4e10 = np.array(clust1_total_4e10)
l_chain_residues = split_seq2list(l_chain_4e10_seq)
print(len(l_chain_residues))
#normalize to time in cluster (divide by total number of cluster frames from  aggregate trajectories )
plot_interaction_profile(clust1_total_4e10[:,0]/3069,
                         clust1_total_4e10[:,1]/3069,
                         clust1_total_4e10[:,2]/3069,
                         l_chain_residues, '4e10_AA_clust1')


# In[12]:


#READ IN 4E10 Cluster 1 

clust2_4e10_ppm =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust2_4e10_ppm,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_ppm/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/4e10_clust2_ppm.dcd')    

clust2_4e10_p15 =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust2_4e10_p15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_p15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/4e10_clust2_p15.dcd')    

clust2_4e10_n15 =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust2_4e10_n15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_n15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/4e10_clust2_n15.dcd')    

clust2_4e10_ppm_rep =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust2_4e10_ppm_rep,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_ppm_rep/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/4e10_clust2_ppm_rep.dcd')


#combine to 4e10_clust2_total 
#aggregate 4e10 AA interaction profile - collect counts from each trajectory into one list 
clust2_total_4e10 =[[] for _ in range(441)] 
for i in range(len(clust2_4e10_ppm)):
    #acyl location counts ]
    acyl_counts = clust2_4e10_ppm[i].count(1)
    clust2_total_4e10[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust2_4e10_ppm[i].count(2)
    clust2_total_4e10[i].append(glycerol_counts)
    #po4 
    po4_counts = clust2_4e10_ppm[i].count(3)
    clust2_total_4e10[i].append(po4_counts)
    #water
    water_counts = clust2_4e10_ppm[i].count(4)
    clust2_total_4e10[i].append(water_counts)
#print(aa_resi_loc_counts_4e10_aggregate[318])   
for i in range(len(clust2_4e10_p15)):
    #acyl location counts ]
    acyl_counts = clust2_4e10_p15[i].count(1)
    clust2_total_4e10[i][0] =clust2_total_4e10[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust2_4e10_p15[i].count(2)
    clust2_total_4e10[i][1] =clust2_total_4e10[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust2_4e10_p15[i].count(3)
    clust2_total_4e10[i][2] =clust2_total_4e10[i][2] + (po4_counts)
    #water
    water_counts = clust2_4e10_p15[i].count(4)
    clust2_total_4e10[i][3] =clust2_total_4e10[i][3] + (water_counts)
for i in range(len(clust2_4e10_n15)):
    #acyl location counts ]
    acyl_counts = clust2_4e10_n15[i].count(1)
    clust2_total_4e10[i][0] =clust2_total_4e10[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust2_4e10_n15[i].count(2)
    clust2_total_4e10[i][1] =clust2_total_4e10[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust2_4e10_n15[i].count(3)
    clust2_total_4e10[i][2] =clust2_total_4e10[i][2] + (po4_counts)
    #water
    water_counts = clust2_4e10_n15[i].count(4)
    clust2_total_4e10[i][3] =clust2_total_4e10[i][3] + (water_counts)
    
for i in range(len(clust2_4e10_ppm_rep)):
    #acyl location counts ]
    acyl_counts = clust2_4e10_ppm_rep[i].count(1)
    clust2_total_4e10[i][0] =clust2_total_4e10[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust2_4e10_ppm_rep[i].count(2)
    clust2_total_4e10[i][1] =clust2_total_4e10[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust2_4e10_ppm_rep[i].count(3)
    clust2_total_4e10[i][2] =clust2_total_4e10[i][2] + (po4_counts)
    #water
    water_counts = clust2_4e10_ppm_rep[i].count(4)
    clust2_total_4e10[i][3] =clust2_total_4e10[i][3] + (water_counts)
    
  


# In[ ]:





# In[27]:


l_chain_4e10_seq = 'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEP'
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGECQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'
clust2_total_4e10 = np.array(clust2_total_4e10)
l_chain_residues = split_seq2list(l_chain_4e10_seq)
print(len(l_chain_residues))
plot_interaction_profile(clust2_total_4e10[:,0]/5583,
                         clust2_total_4e10[:,1]/5583,
                         clust2_total_4e10[:,2]/5583,
                         l_chain_residues, '4e10_AA_clust2')


# In[ ]:





# In[14]:


#READ IN 4E10 Cluster 1 

clust3_4e10_ppm =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust3_4e10_ppm,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_ppm/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/4e10_clust3_ppm.dcd')    

clust3_4e10_p15 =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust3_4e10_p15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_p15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/4e10_clust3_p15.dcd')    

clust3_4e10_n15 =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust3_4e10_n15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_n15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/4e10_clust3_n15.dcd')    

clust3_4e10_ppm_rep =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust3_4e10_ppm_rep,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_ppm_rep/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/4e10_clust3_ppm_rep.dcd')


#combine to 4e10_clust3_total 
#aggregate 4e10 AA interaction profile - collect counts from each trajectory into one list 
clust3_total_4e10 =[[] for _ in range(441)] 
for i in range(len(clust3_4e10_ppm)):
    #acyl location counts ]
    acyl_counts = clust3_4e10_ppm[i].count(1)
    clust3_total_4e10[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust3_4e10_ppm[i].count(2)
    clust3_total_4e10[i].append(glycerol_counts)
    #po4 
    po4_counts = clust3_4e10_ppm[i].count(3)
    clust3_total_4e10[i].append(po4_counts)
    #water
    water_counts = clust3_4e10_ppm[i].count(4)
    clust3_total_4e10[i].append(water_counts)
#print(aa_resi_loc_counts_4e10_aggregate[318])   
for i in range(len(clust3_4e10_p15)):
    #acyl location counts ]
    acyl_counts = clust3_4e10_p15[i].count(1)
    clust3_total_4e10[i][0] =clust3_total_4e10[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust3_4e10_p15[i].count(2)
    clust3_total_4e10[i][1] =clust3_total_4e10[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust3_4e10_p15[i].count(3)
    clust3_total_4e10[i][2] =clust3_total_4e10[i][2] + (po4_counts)
    #water
    water_counts = clust3_4e10_p15[i].count(4)
    clust3_total_4e10[i][3] =clust3_total_4e10[i][3] + (water_counts)
for i in range(len(clust3_4e10_n15)):
    #acyl location counts ]
    acyl_counts = clust3_4e10_n15[i].count(1)
    clust3_total_4e10[i][0] =clust3_total_4e10[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust3_4e10_n15[i].count(2)
    clust3_total_4e10[i][1] =clust3_total_4e10[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust3_4e10_n15[i].count(3)
    clust3_total_4e10[i][2] =clust3_total_4e10[i][2] + (po4_counts)
    #water
    water_counts = clust3_4e10_n15[i].count(4)
    clust3_total_4e10[i][3] =clust3_total_4e10[i][3] + (water_counts)
    
for i in range(len(clust3_4e10_ppm_rep)):
    #acyl location counts ]
    acyl_counts = clust3_4e10_ppm_rep[i].count(1)
    clust3_total_4e10[i][0] =clust3_total_4e10[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust3_4e10_ppm_rep[i].count(2)
    clust3_total_4e10[i][1] =clust3_total_4e10[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust3_4e10_ppm_rep[i].count(3)
    clust3_total_4e10[i][2] =clust3_total_4e10[i][2] + (po4_counts)
    #water
    water_counts = clust3_4e10_ppm_rep[i].count(4)
    clust3_total_4e10[i][3] =clust3_total_4e10[i][3] + (water_counts)
    
  


# In[28]:


l_chain_4e10_seq = 'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEP'
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGECQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'
clust3_total_4e10 = np.array(clust3_total_4e10)
l_chain_residues = split_seq2list(l_chain_4e10_seq)
print(len(l_chain_residues))
plot_interaction_profile(clust3_total_4e10[:,0]/3035,
                         clust3_total_4e10[:,1]/3035,
                         clust3_total_4e10[:,2]/3035,
                         l_chain_residues, '4e10_AA_clust3')


# In[29]:


#READ IN 4E10 Cluster 1 

clust4_4e10_ppm =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust4_4e10_ppm,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_ppm/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/4e10_clust4_ppm.dcd')    

clust4_4e10_p15 =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust4_4e10_p15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_p15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/4e10_clust4_p15.dcd')    

clust4_4e10_n15 =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust4_4e10_n15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_n15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/4e10_clust4_n15.dcd')    

clust4_4e10_ppm_rep =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust4_4e10_ppm_rep,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_ppm_rep/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/4e10_clust4_ppm_re.dcd')


#combine to 4e10_clust4_total 
#aggregate 4e10 AA interaction profile - collect counts from each trajectory into one list 
clust4_total_4e10 =[[] for _ in range(441)] 
for i in range(len(clust4_4e10_ppm)):
    #acyl location counts ]
    acyl_counts = clust4_4e10_ppm[i].count(1)
    clust4_total_4e10[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust4_4e10_ppm[i].count(2)
    clust4_total_4e10[i].append(glycerol_counts)
    #po4 
    po4_counts = clust4_4e10_ppm[i].count(3)
    clust4_total_4e10[i].append(po4_counts)
    #water
    water_counts = clust4_4e10_ppm[i].count(4)
    clust4_total_4e10[i].append(water_counts)
#print(aa_resi_loc_counts_4e10_aggregate[318])   
for i in range(len(clust4_4e10_p15)):
    #acyl location counts ]
    acyl_counts = clust4_4e10_p15[i].count(1)
    clust4_total_4e10[i][0] =clust4_total_4e10[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust4_4e10_p15[i].count(2)
    clust4_total_4e10[i][1] =clust4_total_4e10[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust4_4e10_p15[i].count(3)
    clust4_total_4e10[i][2] =clust4_total_4e10[i][2] + (po4_counts)
    #water
    water_counts = clust4_4e10_p15[i].count(4)
    clust4_total_4e10[i][3] =clust4_total_4e10[i][3] + (water_counts)
for i in range(len(clust4_4e10_n15)):
    #acyl location counts ]
    acyl_counts = clust4_4e10_n15[i].count(1)
    clust4_total_4e10[i][0] =clust4_total_4e10[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust4_4e10_n15[i].count(2)
    clust4_total_4e10[i][1] =clust4_total_4e10[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust4_4e10_n15[i].count(3)
    clust4_total_4e10[i][2] =clust4_total_4e10[i][2] + (po4_counts)
    #water
    water_counts = clust4_4e10_n15[i].count(4)
    clust4_total_4e10[i][3] =clust4_total_4e10[i][3] + (water_counts)
    
for i in range(len(clust4_4e10_ppm_rep)):
    #acyl location counts ]
    acyl_counts = clust4_4e10_ppm_rep[i].count(1)
    clust4_total_4e10[i][0] =clust4_total_4e10[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust4_4e10_ppm_rep[i].count(2)
    clust4_total_4e10[i][1] =clust4_total_4e10[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust4_4e10_ppm_rep[i].count(3)
    clust4_total_4e10[i][2] =clust4_total_4e10[i][2] + (po4_counts)
    #water
    water_counts = clust4_4e10_ppm_rep[i].count(4)
    clust4_total_4e10[i][3] =clust4_total_4e10[i][3] + (water_counts)
    


# In[ ]:


l_chain_4e10_seq = 'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEP'
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGECQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'
clust4_total_4e10 = np.array(clust4_total_4e10)
l_chain_residues = split_seq2list(l_chain_4e10_seq)
print(len(l_chain_residues))
plot_interaction_profile(clust4_total_4e10[:,0]/8449,
                         clust4_total_4e10[:,1]/8449,
                         clust4_total_4e10[:,2]/8449,
                         l_chain_residues, '4e10_AA_clust4')


# In[105]:


#approach, rotation 
plt.rcParams["figure.figsize"] = [7.00, 7]
vectors = np.array(([1, 72.33], [0, 0]))
origin = np.array([[0,0], [0,0]])
fig, ax = plt.subplots(1)
plt.axhline(y=0, color='grey', linestyle='dashed')
ax.quiver(*origin,
           vectors[:, 0],
           vectors[:, 1],
           scale=1,
           scale_units='xy',
           angles = 'xy',
           color=['black'], width=0.02 )

ax.set_xlim((-.01, 1))
ax.set_xticks(())
ax.set_ylim((-5, 95))
ax.set_yticks([0, 30, 60, 90])
plt.tick_params(axis='y', labelsize=50)


# In[103]:


#approach, rotation 
plt.rcParams["figure.figsize"] = [7.00, 7]
vectors = np.array(([1, -3.23], [0,0]))
origin = np.array([[0,0], [0,0]])
fig, ax = plt.subplots(1)
plt.axhline(y=0, color='grey', linestyle='dashed')
ax.quiver(*origin,
           vectors[:, 0],
           vectors[:, 1],
           scale=1,
           scale_units='xy',
           angles = 'xy',
           color=['red'], width=0.02 )

ax.set_xlim((-.01, 1))
ax.set_xticks(())
ax.set_ylim((-95, 95))
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
plt.tick_params(axis='y', labelsize=50)


# In[68]:


plt.rcParams["figure.figsize"] = [7.00, 7]
vectors = np.array(([1, 75.65], [1, -8.1]))
origin = np.array([[0,0], [0,0]])
fig, ax = plt.subplots(1)
plt.axhline(y=0, color='grey', linestyle='dashed')
ax.quiver(*origin,
           vectors[:, 0],
           vectors[:, 1],
           scale=1,
           scale_units='xy',
           angles = 'xy',
           color=['black', 'red'], width=0.02 )

ax.set_xlim((-.01, 1))
ax.set_xticks(())
ax.set_ylim((-95, 95))
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
plt.tick_params(axis='y', labelsize=50)


# In[69]:


plt.rcParams["figure.figsize"] = [7.00, 7]
vectors = np.array(([1, 83.92], [1, -2.94]))
origin = np.array([[0,0], [0,0]])
fig, ax = plt.subplots(1)
plt.axhline(y=0, color='grey', linestyle='dashed')
ax.quiver(*origin,
           vectors[:, 0],
           vectors[:, 1],
           scale=1,
           scale_units='xy',
           angles = 'xy',
           color=['black', 'red'], width=0.02 )

ax.set_xlim((-.01, 1))
ax.set_xticks(())
ax.set_ylim((-95, 95))
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
plt.tick_params(axis='y', labelsize=50)


# In[70]:


plt.rcParams["figure.figsize"] = [7.00, 7]
vectors = np.array(([1, 78.6], [1, 2.93]))
origin = np.array([[0,0], [0,0]])
fig, ax = plt.subplots(1)
plt.axhline(y=0, color='grey', linestyle='dashed')
ax.quiver(*origin,
           vectors[:, 0],
           vectors[:, 1],
           scale=1,
           scale_units='xy',
           angles = 'xy',
           color=['black', 'red'], width=0.02 )

ax.set_xlim((-.01, 1))
ax.set_xticks(())
ax.set_ylim((-95, 95))
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
plt.tick_params(axis='y', labelsize=50)


# In[ ]:





# In[73]:


#combine to 10e8_clust1_total 
clust1_10e8_ppm =[[] for _ in range(444)] #number of residues 
calc_res_depths(clust1_10e8_ppm,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_ppm/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/10e8_clust1_ppm.dcd')    

clust1_10e8_p15 =[[] for _ in range(444)] #number of residues 
calc_res_depths(clust1_10e8_p15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_p15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/10e8_clust1_p15.dcd')    

clust1_10e8_n15 =[[] for _ in range(444)] #number of residues 
calc_res_depths(clust1_10e8_n15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_n15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/10e8_clust1_n15.dcd')    

clust1_10e8_ppm_rep =[[] for _ in range(444)] #number of residues 
calc_res_depths(clust1_10e8_ppm_rep,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_ppm_rep/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/10e8_clust1_ppm_rep.dcd')

#aggregate 10e8 AA interaction profile - collect counts from each trajectory into one list 
clust1_total_10e8 =[[] for _ in range(444)] 
for i in range(len(clust1_10e8_ppm)):
    #acyl location counts ]
    acyl_counts = clust1_10e8_ppm[i].count(1)
    clust1_total_10e8[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust1_10e8_ppm[i].count(2)
    clust1_total_10e8[i].append(glycerol_counts)
    #po4 
    po4_counts = clust1_10e8_ppm[i].count(3)
    clust1_total_10e8[i].append(po4_counts)
    #water
    water_counts = clust1_10e8_ppm[i].count(4)
    clust1_total_10e8[i].append(water_counts)
#print(aa_resi_loc_counts_10e8_aggregate[318])   
for i in range(len(clust1_10e8_p15)):
    #acyl location counts ]
    acyl_counts = clust1_10e8_p15[i].count(1)
    clust1_total_10e8[i][0] =clust1_total_10e8[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust1_10e8_p15[i].count(2)
    clust1_total_10e8[i][1] =clust1_total_10e8[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust1_10e8_p15[i].count(3)
    clust1_total_10e8[i][2] =clust1_total_10e8[i][2] + (po4_counts)
    #water
    water_counts = clust1_10e8_p15[i].count(4)
    clust1_total_10e8[i][3] =clust1_total_10e8[i][3] + (water_counts)
for i in range(len(clust1_10e8_n15)):
    #acyl location counts ]
    acyl_counts = clust1_10e8_n15[i].count(1)
    clust1_total_10e8[i][0] =clust1_total_10e8[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust1_10e8_n15[i].count(2)
    clust1_total_10e8[i][1] =clust1_total_10e8[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust1_10e8_n15[i].count(3)
    clust1_total_10e8[i][2] =clust1_total_10e8[i][2] + (po4_counts)
    #water
    water_counts = clust1_10e8_n15[i].count(4)
    clust1_total_10e8[i][3] =clust1_total_10e8[i][3] + (water_counts)
    
for i in range(len(clust1_10e8_ppm_rep)):
    #acyl location counts ]
    acyl_counts = clust1_10e8_ppm_rep[i].count(1)
    clust1_total_10e8[i][0] =clust1_total_10e8[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust1_10e8_ppm_rep[i].count(2)
    clust1_total_10e8[i][1] =clust1_total_10e8[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust1_10e8_ppm_rep[i].count(3)
    clust1_total_10e8[i][2] =clust1_total_10e8[i][2] + (po4_counts)
    #water
    water_counts = clust1_10e8_ppm_rep[i].count(4)
    clust1_total_10e8[i][3] =clust1_total_10e8[i][3] + (water_counts)
    
  


# In[76]:


l_chain = clust1_total_10e8[233:445]
h_chain = clust1_total_10e8[0:233]
clust1_total_10e8_fin = [] 
for i in l_chain:
    clust1_total_10e8_fin.append(i)
for i in h_chain:
    clust1_total_10e8_fin.append(i)
#print(len(interaction_prof_10e8_AA))    
#print(len(interaction_prof_10e8_AA_final))
clust1_total_10e8_fin = np.array(clust1_total_10e8_fin)

chain_10e8_seq = "SYELTQETGVSVALGRTVTITCRGDSLRSHYASWYQKKPGQAPILLFYGKNNRPSGVPDRFSGSASGNRASLTISGAQAEDDAEYYCSSRDKSGSRLSVFGGGTKLTVLSQPKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPVKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTEVQLVESGGGLVKPGGSLRLSCSASGFDFDNAWMTWVRQPPGKGLEWVGRITGPGEGWSVDYAAPVEGRFTISRLNSINFLYLEMNNLRMEDSGLYFCARTGKYYDFWSGYPPGEEYFQDWGRGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPK"
#chain_10e8_seq ="SYELTQETGVSVALGRTVTITCRGDSLRSHYASWYQKKPGQAPILLFYGKNNRPSGVPDRFSGSASGNRASLTISGAQAEDDAEYYCSSRDKSGSRLSVFGGGTKLTVLSQPKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPVKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTECSEVQLVESGGGLVKPGGSLRLSCSASGFDFDNAWMTWVRQPPGKGLEWVGRITGPGEGWSVDYAAPVEGRFTISRLNSINFLYLEMNNLRMEDSGLYFCARTGKYYDFWSGYPPGEEYFQDWGRGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSCDK"
chain_residues = split_seq2list(chain_10e8_seq)
print(len(chain_10e8_seq))
plot_interaction_profile_10e8(clust1_total_10e8_fin[:,0]/4044,
                         clust1_total_10e8_fin[:,1]/4044,
                         clust1_total_10e8_fin[:,2]/4044,
                         chain_10e8_seq, '10e8_AA_clust1')


# In[77]:


#combine to 10e8_clust2_total 
clust2_10e8_ppm =[[] for _ in range(444)] #number of residues 
calc_res_depths(clust2_10e8_ppm,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_ppm/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/10e8_clust2_ppm.dcd')    

clust2_10e8_p15 =[[] for _ in range(444)] #number of residues 
calc_res_depths(clust2_10e8_p15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_p15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/10e8_clust2_p15.dcd')    

clust2_10e8_n15 =[[] for _ in range(444)] #number of residues 
calc_res_depths(clust2_10e8_n15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_n15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/10e8_clust2_n15.dcd')    

clust2_10e8_ppm_rep =[[] for _ in range(444)] #number of residues 
calc_res_depths(clust2_10e8_ppm_rep,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_ppm_rep/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/10e8_clust2_ppm_rep.dcd')

#aggregate 10e8 AA interaction profile - collect counts from each trajectory into one list 
clust2_total_10e8 =[[] for _ in range(444)] 
for i in range(len(clust2_10e8_ppm)):
    #acyl location counts ]
    acyl_counts = clust2_10e8_ppm[i].count(1)
    clust2_total_10e8[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust2_10e8_ppm[i].count(2)
    clust2_total_10e8[i].append(glycerol_counts)
    #po4 
    po4_counts = clust2_10e8_ppm[i].count(3)
    clust2_total_10e8[i].append(po4_counts)
    #water
    water_counts = clust2_10e8_ppm[i].count(4)
    clust2_total_10e8[i].append(water_counts)
#print(aa_resi_loc_counts_10e8_aggregate[318])   
for i in range(len(clust2_10e8_p15)):
    #acyl location counts ]
    acyl_counts = clust2_10e8_p15[i].count(1)
    clust2_total_10e8[i][0] =clust2_total_10e8[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust2_10e8_p15[i].count(2)
    clust2_total_10e8[i][1] =clust2_total_10e8[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust2_10e8_p15[i].count(3)
    clust2_total_10e8[i][2] =clust2_total_10e8[i][2] + (po4_counts)
    #water
    water_counts = clust2_10e8_p15[i].count(4)
    clust2_total_10e8[i][3] =clust2_total_10e8[i][3] + (water_counts)
for i in range(len(clust2_10e8_n15)):
    #acyl location counts ]
    acyl_counts = clust2_10e8_n15[i].count(1)
    clust2_total_10e8[i][0] =clust2_total_10e8[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust2_10e8_n15[i].count(2)
    clust2_total_10e8[i][1] =clust2_total_10e8[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust2_10e8_n15[i].count(3)
    clust2_total_10e8[i][2] =clust2_total_10e8[i][2] + (po4_counts)
    #water
    water_counts = clust2_10e8_n15[i].count(4)
    clust2_total_10e8[i][3] =clust2_total_10e8[i][3] + (water_counts)
    
for i in range(len(clust2_10e8_ppm_rep)):
    #acyl location counts ]
    acyl_counts = clust2_10e8_ppm_rep[i].count(1)
    clust2_total_10e8[i][0] =clust2_total_10e8[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust2_10e8_ppm_rep[i].count(2)
    clust2_total_10e8[i][1] =clust2_total_10e8[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust2_10e8_ppm_rep[i].count(3)
    clust2_total_10e8[i][2] =clust2_total_10e8[i][2] + (po4_counts)
    #water
    water_counts = clust2_10e8_ppm_rep[i].count(4)
    clust2_total_10e8[i][3] =clust2_total_10e8[i][3] + (water_counts)
    
  


# In[90]:


l_chain = clust2_total_10e8[233:445]
h_chain = clust2_total_10e8[0:233]
clust2_total_10e8_fin = [] 
for i in l_chain:
    clust2_total_10e8_fin.append(i)
for i in h_chain:
    clust2_total_10e8_fin.append(i)
#print(len(interaction_prof_10e8_AA))    
#print(len(interaction_prof_10e8_AA_final))
clust2_total_10e8_fin = np.array(clust2_total_10e8_fin)

chain_10e8_seq = "SYELTQETGVSVALGRTVTITCRGDSLRSHYASWYQKKPGQAPILLFYGKNNRPSGVPDRFSGSASGNRASLTISGAQAEDDAEYYCSSRDKSGSRLSVFGGGTKLTVLSQPKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPVKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTEVQLVESGGGLVKPGGSLRLSCSASGFDFDNAWMTWVRQPPGKGLEWVGRITGPGEGWSVDYAAPVEGRFTISRLNSINFLYLEMNNLRMEDSGLYFCARTGKYYDFWSGYPPGEEYFQDWGRGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPK"
#chain_10e8_seq ="SYELTQETGVSVALGRTVTITCRGDSLRSHYASWYQKKPGQAPILLFYGKNNRPSGVPDRFSGSASGNRASLTISGAQAEDDAEYYCSSRDKSGSRLSVFGGGTKLTVLSQPKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPVKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTECSEVQLVESGGGLVKPGGSLRLSCSASGFDFDNAWMTWVRQPPGKGLEWVGRITGPGEGWSVDYAAPVEGRFTISRLNSINFLYLEMNNLRMEDSGLYFCARTGKYYDFWSGYPPGEEYFQDWGRGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSCDK"
chain_residues = split_seq2list(chain_10e8_seq)
print(len(chain_10e8_seq))
plot_interaction_profile_10e8(clust2_total_10e8_fin[:,0]/3335,
                         clust2_total_10e8_fin[:,1]/3335,
                         clust2_total_10e8_fin[:,2]/3335,
                         chain_10e8_seq, '10e8_AA_clust2')


# In[79]:


#combine to 10e8_clust3_total 
clust3_10e8_ppm =[[] for _ in range(444)] #number of residues 
calc_res_depths(clust3_10e8_ppm,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_ppm/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/10e8_clust3_ppm.dcd')    

clust3_10e8_p15 =[[] for _ in range(444)] #number of residues 
calc_res_depths(clust3_10e8_p15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_p15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/10e8_clust3_p15.dcd')    

clust3_10e8_n15 =[[] for _ in range(444)] #number of residues 
calc_res_depths(clust3_10e8_n15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_n15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/10e8_clust3_n15.dcd')    

clust3_10e8_ppm_rep =[[] for _ in range(444)] #number of residues 
calc_res_depths(clust3_10e8_ppm_rep,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_ppm_rep/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/10e8_clust3_ppm_rep.dcd')

#aggregate 10e8 AA interaction profile - collect counts from each trajectory into one list 
clust3_total_10e8 =[[] for _ in range(444)] 
for i in range(len(clust3_10e8_ppm)):
    #acyl location counts ]
    acyl_counts = clust3_10e8_ppm[i].count(1)
    clust3_total_10e8[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust3_10e8_ppm[i].count(2)
    clust3_total_10e8[i].append(glycerol_counts)
    #po4 
    po4_counts = clust3_10e8_ppm[i].count(3)
    clust3_total_10e8[i].append(po4_counts)
    #water
    water_counts = clust3_10e8_ppm[i].count(4)
    clust3_total_10e8[i].append(water_counts)
#print(aa_resi_loc_counts_10e8_aggregate[318])   
for i in range(len(clust3_10e8_p15)):
    #acyl location counts ]
    acyl_counts = clust3_10e8_p15[i].count(1)
    clust3_total_10e8[i][0] =clust3_total_10e8[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust3_10e8_p15[i].count(2)
    clust3_total_10e8[i][1] =clust3_total_10e8[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust3_10e8_p15[i].count(3)
    clust3_total_10e8[i][2] =clust3_total_10e8[i][2] + (po4_counts)
    #water
    water_counts = clust3_10e8_p15[i].count(4)
    clust3_total_10e8[i][3] =clust3_total_10e8[i][3] + (water_counts)
for i in range(len(clust3_10e8_n15)):
    #acyl location counts ]
    acyl_counts = clust3_10e8_n15[i].count(1)
    clust3_total_10e8[i][0] =clust3_total_10e8[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust3_10e8_n15[i].count(2)
    clust3_total_10e8[i][1] =clust3_total_10e8[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust3_10e8_n15[i].count(3)
    clust3_total_10e8[i][2] =clust3_total_10e8[i][2] + (po4_counts)
    #water
    water_counts = clust3_10e8_n15[i].count(4)
    clust3_total_10e8[i][3] =clust3_total_10e8[i][3] + (water_counts)
    
for i in range(len(clust3_10e8_ppm_rep)):
    #acyl location counts ]
    acyl_counts = clust3_10e8_ppm_rep[i].count(1)
    clust3_total_10e8[i][0] =clust3_total_10e8[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust3_10e8_ppm_rep[i].count(2)
    clust3_total_10e8[i][1] =clust3_total_10e8[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust3_10e8_ppm_rep[i].count(3)
    clust3_total_10e8[i][2] =clust3_total_10e8[i][2] + (po4_counts)
    #water
    water_counts = clust3_10e8_ppm_rep[i].count(4)
    clust3_total_10e8[i][3] =clust3_total_10e8[i][3] + (water_counts)
    
  


# In[80]:


#combine to 10e8_clust4_total 
clust4_10e8_ppm =[[] for _ in range(444)] #number of residues 
calc_res_depths(clust4_10e8_ppm,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_ppm/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/10e8_clust4_ppm.dcd')    

clust4_10e8_p15 =[[] for _ in range(444)] #number of residues 
calc_res_depths(clust4_10e8_p15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_p15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/10e8_clust4_p15.dcd')    

clust4_10e8_n15 =[[] for _ in range(444)] #number of residues 
calc_res_depths(clust4_10e8_n15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_n15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/10e8_clust4_n15.dcd')    

clust4_10e8_ppm_rep =[[] for _ in range(444)] #number of residues 
calc_res_depths(clust4_10e8_ppm_rep,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_ppm_rep/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/10e8_clust4_ppm_rep.dcd')

#aggregate 10e8 AA interaction profile - collect counts from each trajectory into one list 
clust4_total_10e8 =[[] for _ in range(444)] 
for i in range(len(clust4_10e8_ppm)):
    #acyl location counts ]
    acyl_counts = clust4_10e8_ppm[i].count(1)
    clust4_total_10e8[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust4_10e8_ppm[i].count(2)
    clust4_total_10e8[i].append(glycerol_counts)
    #po4 
    po4_counts = clust4_10e8_ppm[i].count(3)
    clust4_total_10e8[i].append(po4_counts)
    #water
    water_counts = clust4_10e8_ppm[i].count(4)
    clust4_total_10e8[i].append(water_counts)
#print(aa_resi_loc_counts_10e8_aggregate[318])   
for i in range(len(clust4_10e8_p15)):
    #acyl location counts ]
    acyl_counts = clust4_10e8_p15[i].count(1)
    clust4_total_10e8[i][0] =clust4_total_10e8[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust4_10e8_p15[i].count(2)
    clust4_total_10e8[i][1] =clust4_total_10e8[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust4_10e8_p15[i].count(3)
    clust4_total_10e8[i][2] =clust4_total_10e8[i][2] + (po4_counts)
    #water
    water_counts = clust4_10e8_p15[i].count(4)
    clust4_total_10e8[i][3] =clust4_total_10e8[i][3] + (water_counts)
for i in range(len(clust4_10e8_n15)):
    #acyl location counts ]
    acyl_counts = clust4_10e8_n15[i].count(1)
    clust4_total_10e8[i][0] =clust4_total_10e8[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust4_10e8_n15[i].count(2)
    clust4_total_10e8[i][1] =clust4_total_10e8[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust4_10e8_n15[i].count(3)
    clust4_total_10e8[i][2] =clust4_total_10e8[i][2] + (po4_counts)
    #water
    water_counts = clust4_10e8_n15[i].count(4)
    clust4_total_10e8[i][3] =clust4_total_10e8[i][3] + (water_counts)
    
for i in range(len(clust4_10e8_ppm_rep)):
    #acyl location counts ]
    acyl_counts = clust4_10e8_ppm_rep[i].count(1)
    clust4_total_10e8[i][0] =clust4_total_10e8[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust4_10e8_ppm_rep[i].count(2)
    clust4_total_10e8[i][1] =clust4_total_10e8[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust4_10e8_ppm_rep[i].count(3)
    clust4_total_10e8[i][2] =clust4_total_10e8[i][2] + (po4_counts)
    #water
    water_counts = clust4_10e8_ppm_rep[i].count(4)
    clust4_total_10e8[i][3] =clust4_total_10e8[i][3] + (water_counts)
    
  


# In[81]:


#combine to 10e8_clust5_total 
clust5_10e8_ppm =[[] for _ in range(444)] #number of residues 
calc_res_depths(clust5_10e8_ppm,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_ppm/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/10e8_clust5_ppm.dcd')    

clust5_10e8_p15 =[[] for _ in range(444)] #number of residues 
calc_res_depths(clust5_10e8_p15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_p15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/10e8_clust5_p15.dcd')    

clust5_10e8_n15 =[[] for _ in range(444)] #number of residues 
calc_res_depths(clust5_10e8_n15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_n15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/10e8_clust5_n15.dcd')    

clust5_10e8_ppm_rep =[[] for _ in range(444)] #number of residues 
calc_res_depths(clust5_10e8_ppm_rep,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_ppm_rep/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/10e8_clust5_ppm_rep.dcd')

#aggregate 10e8 AA interaction profile - collect counts from each trajectory into one list 
clust5_total_10e8 =[[] for _ in range(444)] 
for i in range(len(clust5_10e8_ppm)):
    #acyl location counts ]
    acyl_counts = clust5_10e8_ppm[i].count(1)
    clust5_total_10e8[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust5_10e8_ppm[i].count(2)
    clust5_total_10e8[i].append(glycerol_counts)
    #po4 
    po4_counts = clust5_10e8_ppm[i].count(3)
    clust5_total_10e8[i].append(po4_counts)
    #water
    water_counts = clust5_10e8_ppm[i].count(4)
    clust5_total_10e8[i].append(water_counts)
#print(aa_resi_loc_counts_10e8_aggregate[318])   
for i in range(len(clust5_10e8_p15)):
    #acyl location counts ]
    acyl_counts = clust5_10e8_p15[i].count(1)
    clust5_total_10e8[i][0] =clust5_total_10e8[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust5_10e8_p15[i].count(2)
    clust5_total_10e8[i][1] =clust5_total_10e8[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust5_10e8_p15[i].count(3)
    clust5_total_10e8[i][2] =clust5_total_10e8[i][2] + (po4_counts)
    #water
    water_counts = clust5_10e8_p15[i].count(4)
    clust5_total_10e8[i][3] =clust5_total_10e8[i][3] + (water_counts)
for i in range(len(clust5_10e8_n15)):
    #acyl location counts ]
    acyl_counts = clust5_10e8_n15[i].count(1)
    clust5_total_10e8[i][0] =clust5_total_10e8[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust5_10e8_n15[i].count(2)
    clust5_total_10e8[i][1] =clust5_total_10e8[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust5_10e8_n15[i].count(3)
    clust5_total_10e8[i][2] =clust5_total_10e8[i][2] + (po4_counts)
    #water
    water_counts = clust5_10e8_n15[i].count(4)
    clust5_total_10e8[i][3] =clust5_total_10e8[i][3] + (water_counts)
    
for i in range(len(clust5_10e8_ppm_rep)):
    #acyl location counts ]
    acyl_counts = clust5_10e8_ppm_rep[i].count(1)
    clust5_total_10e8[i][0] =clust5_total_10e8[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust5_10e8_ppm_rep[i].count(2)
    clust5_total_10e8[i][1] =clust5_total_10e8[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust5_10e8_ppm_rep[i].count(3)
    clust5_total_10e8[i][2] =clust5_total_10e8[i][2] + (po4_counts)
    #water
    water_counts = clust5_10e8_ppm_rep[i].count(4)
    clust5_total_10e8[i][3] =clust5_total_10e8[i][3] + (water_counts)
    
  


# In[89]:


l_chain = clust3_total_10e8[233:445]
h_chain = clust3_total_10e8[0:233]
clust3_total_10e8_fin = [] 
for i in l_chain:
    clust3_total_10e8_fin.append(i)
for i in h_chain:
    clust3_total_10e8_fin.append(i)
#print(len(interaction_prof_10e8_AA))    
#print(len(interaction_prof_10e8_AA_final))
clust3_total_10e8_fin = np.array(clust3_total_10e8_fin)

chain_10e8_seq = "SYELTQETGVSVALGRTVTITCRGDSLRSHYASWYQKKPGQAPILLFYGKNNRPSGVPDRFSGSASGNRASLTISGAQAEDDAEYYCSSRDKSGSRLSVFGGGTKLTVLSQPKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPVKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTEVQLVESGGGLVKPGGSLRLSCSASGFDFDNAWMTWVRQPPGKGLEWVGRITGPGEGWSVDYAAPVEGRFTISRLNSINFLYLEMNNLRMEDSGLYFCARTGKYYDFWSGYPPGEEYFQDWGRGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPK"
#chain_10e8_seq ="SYELTQETGVSVALGRTVTITCRGDSLRSHYASWYQKKPGQAPILLFYGKNNRPSGVPDRFSGSASGNRASLTISGAQAEDDAEYYCSSRDKSGSRLSVFGGGTKLTVLSQPKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPVKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTECSEVQLVESGGGLVKPGGSLRLSCSASGFDFDNAWMTWVRQPPGKGLEWVGRITGPGEGWSVDYAAPVEGRFTISRLNSINFLYLEMNNLRMEDSGLYFCARTGKYYDFWSGYPPGEEYFQDWGRGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSCDK"
chain_residues = split_seq2list(chain_10e8_seq)
print(len(chain_10e8_seq))
plot_interaction_profile_10e8(clust3_total_10e8_fin[:,0]/2694,
                         clust3_total_10e8_fin[:,1]/2694,
                         clust3_total_10e8_fin[:,2]/2694,
                         chain_10e8_seq, '10e8_AA_clust3')


# In[88]:


l_chain = clust4_total_10e8[233:445]
h_chain = clust4_total_10e8[0:233]
clust4_total_10e8_fin = [] 
for i in l_chain:
    clust4_total_10e8_fin.append(i)
for i in h_chain:
    clust4_total_10e8_fin.append(i)
#print(len(interaction_prof_10e8_AA))    
#print(len(interaction_prof_10e8_AA_final))
clust4_total_10e8_fin = np.array(clust4_total_10e8_fin)

chain_10e8_seq = "SYELTQETGVSVALGRTVTITCRGDSLRSHYASWYQKKPGQAPILLFYGKNNRPSGVPDRFSGSASGNRASLTISGAQAEDDAEYYCSSRDKSGSRLSVFGGGTKLTVLSQPKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPVKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTEVQLVESGGGLVKPGGSLRLSCSASGFDFDNAWMTWVRQPPGKGLEWVGRITGPGEGWSVDYAAPVEGRFTISRLNSINFLYLEMNNLRMEDSGLYFCARTGKYYDFWSGYPPGEEYFQDWGRGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPK"
#chain_10e8_seq ="SYELTQETGVSVALGRTVTITCRGDSLRSHYASWYQKKPGQAPILLFYGKNNRPSGVPDRFSGSASGNRASLTISGAQAEDDAEYYCSSRDKSGSRLSVFGGGTKLTVLSQPKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPVKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTECSEVQLVESGGGLVKPGGSLRLSCSASGFDFDNAWMTWVRQPPGKGLEWVGRITGPGEGWSVDYAAPVEGRFTISRLNSINFLYLEMNNLRMEDSGLYFCARTGKYYDFWSGYPPGEEYFQDWGRGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSCDK"
chain_residues = split_seq2list(chain_10e8_seq)
print(len(chain_10e8_seq))
plot_interaction_profile_10e8(clust4_total_10e8_fin[:,0]/3913,
                         clust4_total_10e8_fin[:,1]/3913,
                         clust4_total_10e8_fin[:,2]/3913,
                         chain_10e8_seq, '10e8_AA_clust4')


# In[87]:


l_chain = clust5_total_10e8[233:445]
h_chain = clust5_total_10e8[0:233]
clust5_total_10e8_fin = [] 
for i in l_chain:
    clust5_total_10e8_fin.append(i)
for i in h_chain:
    clust5_total_10e8_fin.append(i)
#print(len(interaction_prof_10e8_AA))    
#print(len(interaction_prof_10e8_AA_final))
clust5_total_10e8_fin = np.array(clust5_total_10e8_fin)

chain_10e8_seq = "SYELTQETGVSVALGRTVTITCRGDSLRSHYASWYQKKPGQAPILLFYGKNNRPSGVPDRFSGSASGNRASLTISGAQAEDDAEYYCSSRDKSGSRLSVFGGGTKLTVLSQPKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPVKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTEVQLVESGGGLVKPGGSLRLSCSASGFDFDNAWMTWVRQPPGKGLEWVGRITGPGEGWSVDYAAPVEGRFTISRLNSINFLYLEMNNLRMEDSGLYFCARTGKYYDFWSGYPPGEEYFQDWGRGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPK"
#chain_10e8_seq ="SYELTQETGVSVALGRTVTITCRGDSLRSHYASWYQKKPGQAPILLFYGKNNRPSGVPDRFSGSASGNRASLTISGAQAEDDAEYYCSSRDKSGSRLSVFGGGTKLTVLSQPKAAPSVTLFPPSSEELQANKATLVCLISDFYPGAVTVAWKADSSPVKAGVETTTPSKQSNNKYAASSYLSLTPEQWKSHRSYSCQVTHEGSTVEKTVAPTECSEVQLVESGGGLVKPGGSLRLSCSASGFDFDNAWMTWVRQPPGKGLEWVGRITGPGEGWSVDYAAPVEGRFTISRLNSINFLYLEMNNLRMEDSGLYFCARTGKYYDFWSGYPPGEEYFQDWGRGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSCDK"
chain_residues = split_seq2list(chain_10e8_seq)
print(len(chain_10e8_seq))
plot_interaction_profile_10e8(clust5_total_10e8_fin[:,0]/6150,
                         clust5_total_10e8_fin[:,1]/6150,
                         clust5_total_10e8_fin[:,2]/6150,
                         chain_10e8_seq, '10e8_AA_clust5')


# In[91]:


plt.rcParams["figure.figsize"] = [7.00, 7]
vectors = np.array(([1, 38.35], [1, -24.69]))
origin = np.array([[0,0], [0,0]])
fig, ax = plt.subplots(1)
plt.axhline(y=0, color='grey', linestyle='dashed')
ax.quiver(*origin,
           vectors[:, 0],
           vectors[:, 1],
           scale=1,
           scale_units='xy',
           angles = 'xy',
           color=['black', 'red'], width=0.02 )

ax.set_xlim((-.01, 1))
ax.set_xticks(())
ax.set_ylim((-95, 95))
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
plt.tick_params(axis='y', labelsize=50)


# In[106]:


plt.rcParams["figure.figsize"] = [7.00, 7]
vectors = np.array(([1, 31.9], [1, -32]))
origin = np.array([[0,0], [0,0]])
fig, ax = plt.subplots(1)
plt.axhline(y=0, color='grey', linestyle='dashed')
ax.quiver(*origin,
           vectors[:, 0],
           vectors[:, 1],
           scale=1,
           scale_units='xy',
           angles = 'xy',
           color=['black', 'red'], width=0.02 )

ax.set_xlim((-.01, 1))
ax.set_xticks(())
ax.set_ylim((-95, 95))
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
plt.tick_params(axis='y', labelsize=50)


# In[107]:


plt.rcParams["figure.figsize"] = [7.00, 7]
vectors = np.array(([1, 44], [1, 11]))
origin = np.array([[0,0], [0,0]])
fig, ax = plt.subplots(1)
plt.axhline(y=0, color='grey', linestyle='dashed')
ax.quiver(*origin,
           vectors[:, 0],
           vectors[:, 1],
           scale=1,
           scale_units='xy',
           angles = 'xy',
           color=['black', 'red'], width=0.02 )

ax.set_xlim((-.01, 1))
ax.set_xticks(())
ax.set_ylim((-95, 95))
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
plt.tick_params(axis='y', labelsize=50)


# In[108]:


plt.rcParams["figure.figsize"] = [7.00, 7]
vectors = np.array(([1, 44], [1, -16]))
origin = np.array([[0,0], [0,0]])
fig, ax = plt.subplots(1)
plt.axhline(y=0, color='grey', linestyle='dashed')
ax.quiver(*origin,
           vectors[:, 0],
           vectors[:, 1],
           scale=1,
           scale_units='xy',
           angles = 'xy',
           color=['black', 'red'], width=0.02 )

ax.set_xlim((-.01, 1))
ax.set_xticks(())
ax.set_ylim((-95, 95))
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
plt.tick_params(axis='y', labelsize=50)


# In[109]:


plt.rcParams["figure.figsize"] = [7.00, 7]
vectors = np.array(([1, 45], [1, -11]))
origin = np.array([[0,0], [0,0]])
fig, ax = plt.subplots(1)
plt.axhline(y=0, color='grey', linestyle='dashed')
ax.quiver(*origin,
           vectors[:, 0],
           vectors[:, 1],
           scale=1,
           scale_units='xy',
           angles = 'xy',
           color=['black', 'red'], width=0.02 )

ax.set_xlim((-.01, 1))
ax.set_xticks(())
ax.set_ylim((-95, 95))
ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
plt.tick_params(axis='y', labelsize=50)


# In[115]:


#combine to pgzl1_clust1_total 
clust1_pgzl1_ppm =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust1_pgzl1_ppm,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_ppm/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/pgzl1_clust1_ppm.dcd')    

clust1_pgzl1_p15 =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust1_pgzl1_p15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_p15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/pgzl1_clust1_p15.dcd')    

clust1_pgzl1_n15 =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust1_pgzl1_n15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_n15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/pgzl1_clust1_n15.dcd')    

clust1_pgzl1_ppm_rep =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust1_pgzl1_ppm_rep,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_ppm_rep/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/pgzl1_clust1_ppm_rep.dcd')

#aggregate pgzl1 AA interaction profile - collect counts from each trajectory into one list 
clust1_total_pgzl1 =[[] for _ in range(441)] 
for i in range(len(clust1_pgzl1_ppm)):
    #acyl location counts ]
    acyl_counts = clust1_pgzl1_ppm[i].count(1)
    clust1_total_pgzl1[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust1_pgzl1_ppm[i].count(2)
    clust1_total_pgzl1[i].append(glycerol_counts)
    #po4 
    po4_counts = clust1_pgzl1_ppm[i].count(3)
    clust1_total_pgzl1[i].append(po4_counts)
    #water
    water_counts = clust1_pgzl1_ppm[i].count(4)
    clust1_total_pgzl1[i].append(water_counts)
#print(aa_resi_loc_counts_pgzl1_aggregate[318])   
for i in range(len(clust1_pgzl1_p15)):
    #acyl location counts ]
    acyl_counts = clust1_pgzl1_p15[i].count(1)
    clust1_total_pgzl1[i][0] =clust1_total_pgzl1[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust1_pgzl1_p15[i].count(2)
    clust1_total_pgzl1[i][1] =clust1_total_pgzl1[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust1_pgzl1_p15[i].count(3)
    clust1_total_pgzl1[i][2] =clust1_total_pgzl1[i][2] + (po4_counts)
    #water
    water_counts = clust1_pgzl1_p15[i].count(4)
    clust1_total_pgzl1[i][3] =clust1_total_pgzl1[i][3] + (water_counts)
for i in range(len(clust1_pgzl1_n15)):
    #acyl location counts ]
    acyl_counts = clust1_pgzl1_n15[i].count(1)
    clust1_total_pgzl1[i][0] =clust1_total_pgzl1[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust1_pgzl1_n15[i].count(2)
    clust1_total_pgzl1[i][1] =clust1_total_pgzl1[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust1_pgzl1_n15[i].count(3)
    clust1_total_pgzl1[i][2] =clust1_total_pgzl1[i][2] + (po4_counts)
    #water
    water_counts = clust1_pgzl1_n15[i].count(4)
    clust1_total_pgzl1[i][3] =clust1_total_pgzl1[i][3] + (water_counts)
    
for i in range(len(clust1_pgzl1_ppm_rep)):
    #acyl location counts ]
    acyl_counts = clust1_pgzl1_ppm_rep[i].count(1)
    clust1_total_pgzl1[i][0] =clust1_total_pgzl1[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust1_pgzl1_ppm_rep[i].count(2)
    clust1_total_pgzl1[i][1] =clust1_total_pgzl1[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust1_pgzl1_ppm_rep[i].count(3)
    clust1_total_pgzl1[i][2] =clust1_total_pgzl1[i][2] + (po4_counts)
    #water
    water_counts = clust1_pgzl1_ppm_rep[i].count(4)
    clust1_total_pgzl1[i][3] =clust1_total_pgzl1[i][3] + (water_counts)
    
  


# In[116]:


#get list of residues in l chain [gzl1] 
chain_pgzl1_seq = 'DVVMTQSPGTLSLSPGERATLSCRASQSVSGGALAWYQQKPGQAPRLLIYDTSSRPTGVPGRFSGSGSGTDFSLTISRLEPEDFAVYYCQQYGTSQSTFGQGTRLEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEEVQLVQSGGEVKRPGSSVTVSCKATGGTFSTLAFNWVRQAPGQGPEWMGGIVPLFSIVNYGQKFQGRLTIRADKSTTTVFLDLSGLTSADTATYYCAREGEGWFGKPLRAFEFWGQGTVITVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'
clust1_total_pgzl1 = np.array(clust1_total_pgzl1)

#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC'
chain_residues = split_seq2list(chain_pgzl1_seq)
print(len(chain_pgzl1_seq))
plot_interaction_profile_pgzl1(clust1_total_pgzl1[:,0],
                         clust1_total_pgzl1[:,1],
                         clust1_total_pgzl1[:,2],
                         chain_pgzl1_seq, 'pgzl1_AA_clust1')


# In[119]:


#combine to pgzl1_clust2_total 
clust2_pgzl1_ppm =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust2_pgzl1_ppm,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_ppm/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/pgzl1_clust2_ppm.dcd')    

clust2_pgzl1_p15 =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust2_pgzl1_p15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_p15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/pgzl1_clust2_p15.dcd')    

clust2_pgzl1_n15 =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust2_pgzl1_n15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_n15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/pgzl1_clust2_n15.dcd')    

clust2_pgzl1_ppm_rep =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust2_pgzl1_ppm_rep,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_ppm_rep/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/pgzl1_clust2_ppm_rep.dcd')

#aggregate pgzl1 AA interaction profile - collect counts from each trajectory into one list 
clust2_total_pgzl1 =[[] for _ in range(441)] 
for i in range(len(clust2_pgzl1_ppm)):
    #acyl location counts ]
    acyl_counts = clust2_pgzl1_ppm[i].count(1)
    clust2_total_pgzl1[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust2_pgzl1_ppm[i].count(2)
    clust2_total_pgzl1[i].append(glycerol_counts)
    #po4 
    po4_counts = clust2_pgzl1_ppm[i].count(3)
    clust2_total_pgzl1[i].append(po4_counts)
    #water
    water_counts = clust2_pgzl1_ppm[i].count(4)
    clust2_total_pgzl1[i].append(water_counts)
#print(aa_resi_loc_counts_pgzl1_aggregate[318])   
for i in range(len(clust2_pgzl1_p15)):
    #acyl location counts ]
    acyl_counts = clust2_pgzl1_p15[i].count(1)
    clust2_total_pgzl1[i][0] =clust2_total_pgzl1[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust2_pgzl1_p15[i].count(2)
    clust2_total_pgzl1[i][1] =clust2_total_pgzl1[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust2_pgzl1_p15[i].count(3)
    clust2_total_pgzl1[i][2] =clust2_total_pgzl1[i][2] + (po4_counts)
    #water
    water_counts = clust2_pgzl1_p15[i].count(4)
    clust2_total_pgzl1[i][3] =clust2_total_pgzl1[i][3] + (water_counts)
for i in range(len(clust2_pgzl1_n15)):
    #acyl location counts ]
    acyl_counts = clust2_pgzl1_n15[i].count(1)
    clust2_total_pgzl1[i][0] =clust2_total_pgzl1[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust2_pgzl1_n15[i].count(2)
    clust2_total_pgzl1[i][1] =clust2_total_pgzl1[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust2_pgzl1_n15[i].count(3)
    clust2_total_pgzl1[i][2] =clust2_total_pgzl1[i][2] + (po4_counts)
    #water
    water_counts = clust2_pgzl1_n15[i].count(4)
    clust2_total_pgzl1[i][3] =clust2_total_pgzl1[i][3] + (water_counts)
    
for i in range(len(clust2_pgzl1_ppm_rep)):
    #acyl location counts ]
    acyl_counts = clust2_pgzl1_ppm_rep[i].count(1)
    clust2_total_pgzl1[i][0] =clust2_total_pgzl1[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust2_pgzl1_ppm_rep[i].count(2)
    clust2_total_pgzl1[i][1] =clust2_total_pgzl1[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust2_pgzl1_ppm_rep[i].count(3)
    clust2_total_pgzl1[i][2] =clust2_total_pgzl1[i][2] + (po4_counts)
    #water
    water_counts = clust2_pgzl1_ppm_rep[i].count(4)
    clust2_total_pgzl1[i][3] =clust2_total_pgzl1[i][3] + (water_counts)
    
  


# In[120]:


#get list of residues in l chain [gzl1] 
chain_pgzl1_seq = 'DVVMTQSPGTLSLSPGERATLSCRASQSVSGGALAWYQQKPGQAPRLLIYDTSSRPTGVPGRFSGSGSGTDFSLTISRLEPEDFAVYYCQQYGTSQSTFGQGTRLEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEEVQLVQSGGEVKRPGSSVTVSCKATGGTFSTLAFNWVRQAPGQGPEWMGGIVPLFSIVNYGQKFQGRLTIRADKSTTTVFLDLSGLTSADTATYYCAREGEGWFGKPLRAFEFWGQGTVITVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'
clust2_total_pgzl1 = np.array(clust2_total_pgzl1)

#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC'
chain_residues = split_seq2list(chain_pgzl1_seq)
print(len(chain_pgzl1_seq))
plot_interaction_profile_pgzl1(clust2_total_pgzl1[:,0],
                         clust2_total_pgzl1[:,1],
                         clust2_total_pgzl1[:,2],
                         chain_pgzl1_seq, 'pgzl1_AA_clust2')


# In[121]:


#combine to pgzl1_clust3_total 
clust3_pgzl1_ppm =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust3_pgzl1_ppm,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_ppm/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/pgzl1_clust3_ppm.dcd')    

clust3_pgzl1_p15 =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust3_pgzl1_p15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_p15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/pgzl1_clust3_p15.dcd')    

clust3_pgzl1_n15 =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust3_pgzl1_n15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_n15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/pgzl1_clust3_n15.dcd')    

clust3_pgzl1_ppm_rep =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust3_pgzl1_ppm_rep,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_ppm_rep/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/pgzl1_clust3_ppm_rep.dcd')

#aggregate pgzl1 AA interaction profile - collect counts from each trajectory into one list 
clust3_total_pgzl1 =[[] for _ in range(441)] 
for i in range(len(clust3_pgzl1_ppm)):
    #acyl location counts ]
    acyl_counts = clust3_pgzl1_ppm[i].count(1)
    clust3_total_pgzl1[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust3_pgzl1_ppm[i].count(2)
    clust3_total_pgzl1[i].append(glycerol_counts)
    #po4 
    po4_counts = clust3_pgzl1_ppm[i].count(3)
    clust3_total_pgzl1[i].append(po4_counts)
    #water
    water_counts = clust3_pgzl1_ppm[i].count(4)
    clust3_total_pgzl1[i].append(water_counts)
#print(aa_resi_loc_counts_pgzl1_aggregate[318])   
for i in range(len(clust3_pgzl1_p15)):
    #acyl location counts ]
    acyl_counts = clust3_pgzl1_p15[i].count(1)
    clust3_total_pgzl1[i][0] =clust3_total_pgzl1[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust3_pgzl1_p15[i].count(2)
    clust3_total_pgzl1[i][1] =clust3_total_pgzl1[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust3_pgzl1_p15[i].count(3)
    clust3_total_pgzl1[i][2] =clust3_total_pgzl1[i][2] + (po4_counts)
    #water
    water_counts = clust3_pgzl1_p15[i].count(4)
    clust3_total_pgzl1[i][3] =clust3_total_pgzl1[i][3] + (water_counts)
for i in range(len(clust3_pgzl1_n15)):
    #acyl location counts ]
    acyl_counts = clust3_pgzl1_n15[i].count(1)
    clust3_total_pgzl1[i][0] =clust3_total_pgzl1[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust3_pgzl1_n15[i].count(2)
    clust3_total_pgzl1[i][1] =clust3_total_pgzl1[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust3_pgzl1_n15[i].count(3)
    clust3_total_pgzl1[i][2] =clust3_total_pgzl1[i][2] + (po4_counts)
    #water
    water_counts = clust3_pgzl1_n15[i].count(4)
    clust3_total_pgzl1[i][3] =clust3_total_pgzl1[i][3] + (water_counts)
    
for i in range(len(clust3_pgzl1_ppm_rep)):
    #acyl location counts ]
    acyl_counts = clust3_pgzl1_ppm_rep[i].count(1)
    clust3_total_pgzl1[i][0] =clust3_total_pgzl1[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust3_pgzl1_ppm_rep[i].count(2)
    clust3_total_pgzl1[i][1] =clust3_total_pgzl1[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust3_pgzl1_ppm_rep[i].count(3)
    clust3_total_pgzl1[i][2] =clust3_total_pgzl1[i][2] + (po4_counts)
    #water
    water_counts = clust3_pgzl1_ppm_rep[i].count(4)
    clust3_total_pgzl1[i][3] =clust3_total_pgzl1[i][3] + (water_counts)
    
  


# In[122]:


#get list of residues in l chain [gzl1] 
chain_pgzl1_seq = 'DVVMTQSPGTLSLSPGERATLSCRASQSVSGGALAWYQQKPGQAPRLLIYDTSSRPTGVPGRFSGSGSGTDFSLTISRLEPEDFAVYYCQQYGTSQSTFGQGTRLEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEEVQLVQSGGEVKRPGSSVTVSCKATGGTFSTLAFNWVRQAPGQGPEWMGGIVPLFSIVNYGQKFQGRLTIRADKSTTTVFLDLSGLTSADTATYYCAREGEGWFGKPLRAFEFWGQGTVITVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'
clust3_total_pgzl1 = np.array(clust3_total_pgzl1)

#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC'
chain_residues = split_seq2list(chain_pgzl1_seq)
print(len(chain_pgzl1_seq))
plot_interaction_profile_pgzl1(clust3_total_pgzl1[:,0],
                         clust3_total_pgzl1[:,1],
                         clust3_total_pgzl1[:,2],
                         chain_pgzl1_seq, 'pgzl1_AA_clust3')


# In[129]:


#combine to pgzl1_clust4_total 
clust4_pgzl1_ppm =[[] for _ in range(441)] #number of residues 
# calc_res_depths(clust4_pgzl1_ppm,
#                 '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_ppm/final_analysis_input.pdb',
#                 '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/pgzl1_clust4_ppm.dcd')    

# clust4_pgzl1_p15 =[[] for _ in range(441)] #number of residues 
# calc_res_depths(clust4_pgzl1_p15,
#                 '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_p15/final_analysis_input.pdb',
#                 '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/pgzl1_clust4_p15.dcd')    

clust4_pgzl1_n15 =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust4_pgzl1_n15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_n15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/pgzl1_clust4_n15.dcd')    

# clust4_pgzl1_ppm_rep =[[] for _ in range(441)] #number of residues 
# calc_res_depths(clust4_pgzl1_ppm_rep,
#                 '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_ppm_rep/final_analysis_input.pdb',
#                 '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/pgzl1_clust4_ppm_rep.dcd')

#aggregate pgzl1 AA interaction profile - collect counts from each trajectory into one list 
clust4_total_pgzl1 =[[] for _ in range(441)] 
for i in range(len(clust4_pgzl1_ppm)):
    #acyl location counts ]
    acyl_counts = clust4_pgzl1_ppm[i].count(1)
    clust4_total_pgzl1[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust4_pgzl1_ppm[i].count(2)
    clust4_total_pgzl1[i].append(glycerol_counts)
    #po4 
    po4_counts = clust4_pgzl1_ppm[i].count(3)
    clust4_total_pgzl1[i].append(po4_counts)
    #water
    water_counts = clust4_pgzl1_ppm[i].count(4)
    clust4_total_pgzl1[i].append(water_counts)
#print(aa_resi_loc_counts_pgzl1_aggregate[318])   
for i in range(len(clust4_pgzl1_p15)):
    #acyl location counts ]
    acyl_counts = clust4_pgzl1_p15[i].count(1)
    clust4_total_pgzl1[i][0] =clust4_total_pgzl1[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust4_pgzl1_p15[i].count(2)
    clust4_total_pgzl1[i][1] =clust4_total_pgzl1[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust4_pgzl1_p15[i].count(3)
    clust4_total_pgzl1[i][2] =clust4_total_pgzl1[i][2] + (po4_counts)
    #water
    water_counts = clust4_pgzl1_p15[i].count(4)
    clust4_total_pgzl1[i][3] =clust4_total_pgzl1[i][3] + (water_counts)
for i in range(len(clust4_pgzl1_n15)):
    #acyl location counts ]
    acyl_counts = clust4_pgzl1_n15[i].count(1)
    clust4_total_pgzl1[i][0] =clust4_total_pgzl1[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust4_pgzl1_n15[i].count(2)
    clust4_total_pgzl1[i][1] =clust4_total_pgzl1[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust4_pgzl1_n15[i].count(3)
    clust4_total_pgzl1[i][2] =clust4_total_pgzl1[i][2] + (po4_counts)
    #water
    water_counts = clust4_pgzl1_n15[i].count(4)
    clust4_total_pgzl1[i][3] =clust4_total_pgzl1[i][3] + (water_counts)
    
for i in range(len(clust4_pgzl1_ppm_rep)):
    #acyl location counts ]
    acyl_counts = clust4_pgzl1_ppm_rep[i].count(1)
    clust4_total_pgzl1[i][0] =clust4_total_pgzl1[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust4_pgzl1_ppm_rep[i].count(2)
    clust4_total_pgzl1[i][1] =clust4_total_pgzl1[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust4_pgzl1_ppm_rep[i].count(3)
    clust4_total_pgzl1[i][2] =clust4_total_pgzl1[i][2] + (po4_counts)
    #water
    water_counts = clust4_pgzl1_ppm_rep[i].count(4)
    clust4_total_pgzl1[i][3] =clust4_total_pgzl1[i][3] + (water_counts)
    
  


# In[130]:


#get list of residues in l chain [gzl1] 
chain_pgzl1_seq = 'DVVMTQSPGTLSLSPGERATLSCRASQSVSGGALAWYQQKPGQAPRLLIYDTSSRPTGVPGRFSGSGSGTDFSLTISRLEPEDFAVYYCQQYGTSQSTFGQGTRLEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEEVQLVQSGGEVKRPGSSVTVSCKATGGTFSTLAFNWVRQAPGQGPEWMGGIVPLFSIVNYGQKFQGRLTIRADKSTTTVFLDLSGLTSADTATYYCAREGEGWFGKPLRAFEFWGQGTVITVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'
clust4_total_pgzl1 = np.array(clust4_total_pgzl1)

#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC'
chain_residues = split_seq2list(chain_pgzl1_seq)
print(len(chain_pgzl1_seq))
plot_interaction_profile_pgzl1(clust4_total_pgzl1[:,0],
                         clust4_total_pgzl1[:,1],
                         clust4_total_pgzl1[:,2],
                         chain_pgzl1_seq, 'pgzl1_AA_clust4')


# In[111]:


#combine to pgzl1_clust5_total 
clust5_pgzl1_ppm =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust5_pgzl1_ppm,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_ppm/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/pgzl1_clust5_ppm.dcd')    

clust5_pgzl1_p15 =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust5_pgzl1_p15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_p15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/pgzl1_clust5_p15.dcd')    

clust5_pgzl1_n15 =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust5_pgzl1_n15,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_n15/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/pgzl1_clust5_n15.dcd')    

clust5_pgzl1_ppm_rep =[[] for _ in range(441)] #number of residues 
calc_res_depths(clust5_pgzl1_ppm_rep,
                '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_ppm_rep/final_analysis_input.pdb',
                '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/pgzl1_clust5_ppm_rep.dcd')

#aggregate pgzl1 AA interaction profile - collect counts from each trajectory into one list 
clust5_total_pgzl1 =[[] for _ in range(441)] 
for i in range(len(clust5_pgzl1_ppm)):
    #acyl location counts ]
    acyl_counts = clust5_pgzl1_ppm[i].count(1)
    clust5_total_pgzl1[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust5_pgzl1_ppm[i].count(2)
    clust5_total_pgzl1[i].append(glycerol_counts)
    #po4 
    po4_counts = clust5_pgzl1_ppm[i].count(3)
    clust5_total_pgzl1[i].append(po4_counts)
    #water
    water_counts = clust5_pgzl1_ppm[i].count(4)
    clust5_total_pgzl1[i].append(water_counts)
#print(aa_resi_loc_counts_pgzl1_aggregate[318])   
for i in range(len(clust5_pgzl1_p15)):
    #acyl location counts ]
    acyl_counts = clust5_pgzl1_p15[i].count(1)
    clust5_total_pgzl1[i][0] =clust5_total_pgzl1[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust5_pgzl1_p15[i].count(2) 
    clust5_total_pgzl1[i][1] =clust5_total_pgzl1[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust5_pgzl1_p15[i].count(3)
    clust5_total_pgzl1[i][2] =clust5_total_pgzl1[i][2] + (po4_counts)
    #water
    water_counts = clust5_pgzl1_p15[i].count(4)
    clust5_total_pgzl1[i][3] =clust5_total_pgzl1[i][3] + (water_counts)
for i in range(len(clust5_pgzl1_n15)):
    #acyl location counts ]
    acyl_counts = clust5_pgzl1_n15[i].count(1)
    clust5_total_pgzl1[i][0] =clust5_total_pgzl1[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust5_pgzl1_n15[i].count(2)
    clust5_total_pgzl1[i][1] =clust5_total_pgzl1[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust5_pgzl1_n15[i].count(3)
    clust5_total_pgzl1[i][2] =clust5_total_pgzl1[i][2] + (po4_counts)
    #water
    water_counts = clust5_pgzl1_n15[i].count(4)
    clust5_total_pgzl1[i][3] =clust5_total_pgzl1[i][3] + (water_counts)
    
for i in range(len(clust5_pgzl1_ppm_rep)):
    #acyl location counts ]
    acyl_counts = clust5_pgzl1_ppm_rep[i].count(1)
    clust5_total_pgzl1[i][0] =clust5_total_pgzl1[i][0] + (acyl_counts)
    #glycerol_counts 
    glycerol_counts = clust5_pgzl1_ppm_rep[i].count(2)
    clust5_total_pgzl1[i][1] =clust5_total_pgzl1[i][1] + (glycerol_counts)
    #po4 
    po4_counts = clust5_pgzl1_ppm_rep[i].count(3)
    clust5_total_pgzl1[i][2] =clust5_total_pgzl1[i][2] + (po4_counts)
    #water
    water_counts = clust5_pgzl1_ppm_rep[i].count(4)
    clust5_total_pgzl1[i][3] =clust5_total_pgzl1[i][3] + (water_counts)
    
  


# In[ ]:





# In[114]:


#get list of residues in l chain [gzl1] 
chain_pgzl1_seq = 'DVVMTQSPGTLSLSPGERATLSCRASQSVSGGALAWYQQKPGQAPRLLIYDTSSRPTGVPGRFSGSGSGTDFSLTISRLEPEDFAVYYCQQYGTSQSTFGQGTRLEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEEVQLVQSGGEVKRPGSSVTVSCKATGGTFSTLAFNWVRQAPGQGPEWMGGIVPLFSIVNYGQKFQGRLTIRADKSTTTVFLDLSGLTSADTATYYCAREGEGWFGKPLRAFEFWGQGTVITVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'
clust5_total_pgzl1 = np.array(clust5_total_pgzl1)

#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC'
chain_residues = split_seq2list(chain_pgzl1_seq)
print(len(chain_pgzl1_seq))
plot_interaction_profile_pgzl1(clust5_total_pgzl1[:,0],
                         clust5_total_pgzl1[:,1],
                         clust5_total_pgzl1[:,2],
                         chain_pgzl1_seq, 'pgzl1_AA_clust5')


# In[8]:


wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/pgzl1/medoid1/"

pgzl1_backmap_med1 =[[] for _ in range(441)] #number of residues 
calc_res_depths(pgzl1_backmap_med1,
                wd+'500ns_all.pdb',
                wd+'analysis.dcd')
pgzl1_backmap_med1_totals =[[] for _ in range(441)] 
print(len(pgzl1_backmap_med1))
for i in range(len(pgzl1_backmap_med1)):
    #acyl location counts ]
    acyl_counts = pgzl1_backmap_med1[i].count(1)
    pgzl1_backmap_med1_totals[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = pgzl1_backmap_med1[i].count(2)
    pgzl1_backmap_med1_totals[i].append(glycerol_counts)
    #po4 
    po4_counts = pgzl1_backmap_med1[i].count(3)
    pgzl1_backmap_med1_totals[i].append(po4_counts)
    #water
    water_counts = pgzl1_backmap_med1[i].count(4)
    pgzl1_backmap_med1_totals[i].append(water_counts)


# In[12]:





# In[38]:


chain_pgzl1_seq = 'DVVMTQSPGTLSLSPGERATLSCRASQSVSGGALAWYQQKPGQAPRLLIYDTSSRPTGVPGRFSGSGSGTDFSLTISRLEPEDFAVYYCQQYGTSQSTFGQGTRLEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEEVQLVQSGGEVKRPGSSVTVSCKATGGTFSTLAFNWVRQAPGQGPEWMGGIVPLFSIVNYGQKFQGRLTIRADKSTTTVFLDLSGLTSADTATYYCAREGEGWFGKPLRAFEFWGQGTVITVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'
pgzl1_backmap_med1_totals = np.array(pgzl1_backmap_med1_totals)
wd='/Users/cmaillie/Dropbox (Scripps Research)/mper_ab_manuscript_y3/pngs/'
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC'
chain_residues = split_seq2list(chain_pgzl1_seq)
print(len(chain_pgzl1_seq))
plot_interaction_profile_pgzl1(pgzl1_backmap_med1_totals[:,0]/len(pgzl1_backmap_med1[0]),
                         pgzl1_backmap_med1_totals[:,1]/len(pgzl1_backmap_med1[0]),
                         pgzl1_backmap_med1_totals[:,2]/len(pgzl1_backmap_med2[0]),
                         pgzl1_backmap_med1_totals/len(pgzl1_backmap_med2[0]), wd+'pgzl1_backmap_med1_interaction_profile')


# In[14]:


wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/pgzl1/medoid2/"

pgzl1_backmap_med2 =[[] for _ in range(441)] #number of residues 
calc_res_depths(pgzl1_backmap_med2,
                wd+'500ns_all.pdb',
                wd+'analysis.dcd')
pgzl1_backmap_med2_totals =[[] for _ in range(441)] 
print(len(pgzl1_backmap_med2))
for i in range(len(pgzl1_backmap_med2)):
    #acyl location counts ]
    acyl_counts = pgzl1_backmap_med2[i].count(1)
    pgzl1_backmap_med2_totals[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = pgzl1_backmap_med2[i].count(2)
    pgzl1_backmap_med2_totals[i].append(glycerol_counts)
    #po4 
    po4_counts = pgzl1_backmap_med2[i].count(3)
    pgzl1_backmap_med2_totals[i].append(po4_counts)
    #water
    water_counts = pgzl1_backmap_med2[i].count(4)
    pgzl1_backmap_med2_totals[i].append(water_counts)


# In[37]:


chain_pgzl1_seq = 'DVVMTQSPGTLSLSPGERATLSCRASQSVSGGALAWYQQKPGQAPRLLIYDTSSRPTGVPGRFSGSGSGTDFSLTISRLEPEDFAVYYCQQYGTSQSTFGQGTRLEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEEVQLVQSGGEVKRPGSSVTVSCKATGGTFSTLAFNWVRQAPGQGPEWMGGIVPLFSIVNYGQKFQGRLTIRADKSTTTVFLDLSGLTSADTATYYCAREGEGWFGKPLRAFEFWGQGTVITVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'
pgzl1_backmap_med2_totals = np.array(pgzl1_backmap_med2_totals)
wd='/Users/cmaillie/Dropbox (Scripps Research)/mper_ab_manuscript_y3/pngs/'
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC'
chain_residues = split_seq2list(chain_pgzl1_seq)
print(len(chain_pgzl1_seq))
plot_interaction_profile_pgzl1(pgzl1_backmap_med2_totals[:,0]/len(pgzl1_backmap_med2[0]),
                         pgzl1_backmap_med2_totals[:,1]/len(pgzl1_backmap_med2[0]),
                         pgzl1_backmap_med2_totals[:,2]/len(pgzl1_backmap_med2[0]),
                         pgzl1_backmap_med2_totals/len(pgzl1_backmap_med2[0]), wd+'pgzl1_backmap_med2_interaction_profile')


# In[16]:


wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/pgzl1/medoid3/"

pgzl1_backmap_med3 =[[] for _ in range(441)] #number of residues 
calc_res_depths(pgzl1_backmap_med3,
                wd+'500ns_all.pdb',
                wd+'analysis.dcd')
pgzl1_backmap_med3_totals =[[] for _ in range(441)] 
print(len(pgzl1_backmap_med3))
for i in range(len(pgzl1_backmap_med3)):
    #acyl location counts ]
    acyl_counts = pgzl1_backmap_med3[i].count(1)
    pgzl1_backmap_med3_totals[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = pgzl1_backmap_med3[i].count(2)
    pgzl1_backmap_med3_totals[i].append(glycerol_counts)
    #po4 
    po4_counts = pgzl1_backmap_med3[i].count(3)
    pgzl1_backmap_med3_totals[i].append(po4_counts)
    #water
    water_counts = pgzl1_backmap_med3[i].count(4)
    pgzl1_backmap_med3_totals[i].append(water_counts)


# In[36]:


chain_pgzl1_seq = 'DVVMTQSPGTLSLSPGERATLSCRASQSVSGGALAWYQQKPGQAPRLLIYDTSSRPTGVPGRFSGSGSGTDFSLTISRLEPEDFAVYYCQQYGTSQSTFGQGTRLEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEEVQLVQSGGEVKRPGSSVTVSCKATGGTFSTLAFNWVRQAPGQGPEWMGGIVPLFSIVNYGQKFQGRLTIRADKSTTTVFLDLSGLTSADTATYYCAREGEGWFGKPLRAFEFWGQGTVITVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'
pgzl1_backmap_med3_totals = np.array(pgzl1_backmap_med3_totals)
wd='/Users/cmaillie/Dropbox (Scripps Research)/mper_ab_manuscript_y3/pngs/'
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC'
chain_residues = split_seq2list(chain_pgzl1_seq)
print(len(chain_pgzl1_seq))
plot_interaction_profile_pgzl1(pgzl1_backmap_med3_totals[:,0]/len(pgzl1_backmap_med3[0]),
                         pgzl1_backmap_med3_totals[:,1]/len(pgzl1_backmap_med3[0]),
                         pgzl1_backmap_med3_totals[:,2]/len(pgzl1_backmap_med3[0]),
                         pgzl1_backmap_med3_totals/len(pgzl1_backmap_med3[0]), wd+'pgzl1_backmap_med3_interaction_profile')


# In[35]:


print(pgzl1_backmap_med3_totals[:,0])
print(len(pgzl1_backmap_med3[0]))


# In[44]:


wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/4e10/medoid1/"

backmap_med1_4e10 =[[] for _ in range(441)] #number of residues 
calc_res_depths(backmap_med1_4e10 ,
                wd+'500ns_all.pdb',
                wd+'analysis.dcd')
backmap_med1_totals_4e10  =[[] for _ in range(441)] 
print(len(backmap_med1_4e10 ))


# In[45]:


for i in range(len(backmap_med1_4e10 )):
    #acyl location counts ]
    acyl_counts = backmap_med1_4e10[i].count(1)
    backmap_med1_totals_4e10[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = backmap_med1_4e10[i].count(2)
    backmap_med1_totals_4e10[i].append(glycerol_counts)
    #po4 
    po4_counts = backmap_med1_4e10[i].count(3)
    backmap_med1_totals_4e10[i].append(po4_counts)
    #water
    water_counts = backmap_med1_4e10 [i].count(4)
    backmap_med1_totals_4e10[i].append(water_counts)


# In[22]:


wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/4e10/medoid2/"

backmap_med2_4e10 =[[] for _ in range(441)] #number of residues 
calc_res_depths(backmap_med2_4e10 ,
                wd+'500ns_all.pdb',
                wd+'analysis.dcd')
backmap_med2_totals_4e10  =[[] for _ in range(441)] 
print(len(backmap_med2_4e10 ))
for i in range(len(backmap_med2_4e10 )):
    #acyl location counts ]
    acyl_counts = backmap_med2_4e10[i].count(1)
    backmap_med2_totals_4e10[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = backmap_med2_4e10[i].count(2)
    backmap_med2_totals_4e10[i].append(glycerol_counts)
    #po4 
    po4_counts = backmap_med2_4e10[i].count(3)
    backmap_med2_totals_4e10[i].append(po4_counts)
    #water
    water_counts = backmap_med2_4e10 [i].count(4)
    backmap_med2_totals_4e10[i].append(water_counts)


# In[23]:


wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/4e10/medoid3/"

backmap_med3_4e10 =[[] for _ in range(441)] #number of residues 
calc_res_depths(backmap_med3_4e10 ,
                wd+'500ns_all.pdb',
                wd+'analysis.dcd')
backmap_med3_totals_4e10  =[[] for _ in range(441)] 
print(len(backmap_med3_4e10 ))
for i in range(len(backmap_med3_4e10 )):
    #acyl location counts ]
    acyl_counts = backmap_med3_4e10[i].count(1)
    backmap_med3_totals_4e10[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = backmap_med3_4e10[i].count(2)
    backmap_med3_totals_4e10[i].append(glycerol_counts)
    #po4 
    po4_counts = backmap_med3_4e10[i].count(3)
    backmap_med3_totals_4e10[i].append(po4_counts)
    #water
    water_counts = backmap_med3_4e10 [i].count(4)
    backmap_med3_totals_4e10[i].append(water_counts)


# In[58]:


chain_pgzl1_seq = 'DVVMTQSPGTLSLSPGERATLSCRASQSVSGGALAWYQQKPGQAPRLLIYDTSSRPTGVPGRFSGSGSGTDFSLTISRLEPEDFAVYYCQQYGTSQSTFGQGTRLEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEEVQLVQSGGEVKRPGSSVTVSCKATGGTFSTLAFNWVRQAPGQGPEWMGGIVPLFSIVNYGQKFQGRLTIRADKSTTTVFLDLSGLTSADTATYYCAREGEGWFGKPLRAFEFWGQGTVITVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'

wd='/Users/cmaillie/Dropbox (Scripps Research)/mper_ab_manuscript_y3/pngs/'


l_chain_4e10_seq = 'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEP'
l_chain_residues = split_seq2list(l_chain_4e10_seq)
 
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC'
chain_residues = split_seq2list(chain_pgzl1_seq)
print(len(chain_pgzl1_seq))
plot_interaction_profile(backmap_med1_totals_4e10[:,0]/len(backmap_med1_4e10[0]),
                         backmap_med1_totals_4e10[:,1]/len(backmap_med1_4e10[0]),
                         backmap_med1_totals_4e10[:,2]/len(backmap_med1_4e10[0]),
                         l_chain_residues, wd+'4e10_backmap_med1_interaction_profile')


# In[61]:


chain_pgzl1_seq = 'DVVMTQSPGTLSLSPGERATLSCRASQSVSGGALAWYQQKPGQAPRLLIYDTSSRPTGVPGRFSGSGSGTDFSLTISRLEPEDFAVYYCQQYGTSQSTFGQGTRLEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEEVQLVQSGGEVKRPGSSVTVSCKATGGTFSTLAFNWVRQAPGQGPEWMGGIVPLFSIVNYGQKFQGRLTIRADKSTTTVFLDLSGLTSADTATYYCAREGEGWFGKPLRAFEFWGQGTVITVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'

wd='/Users/cmaillie/Dropbox (Scripps Research)/mper_ab_manuscript_y3/pngs/'


l_chain_4e10_seq = 'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEP'
l_chain_residues = split_seq2list(l_chain_4e10_seq)
 
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC'
chain_residues = split_seq2list(chain_pgzl1_seq)
print(len(chain_pgzl1_seq))
plot_interaction_profile(backmap_med2_totals_4e10[:,0]/len(backmap_med2_4e10[0]),
                         backmap_med2_totals_4e10[:,1]/len(backmap_med2_4e10[0]),
                         backmap_med2_totals_4e10[:,2]/len(backmap_med2_4e10[0]),
                         l_chain_residues, wd+'4e10_backmap_med2_interaction_profile')


# In[62]:


chain_pgzl1_seq = 'DVVMTQSPGTLSLSPGERATLSCRASQSVSGGALAWYQQKPGQAPRLLIYDTSSRPTGVPGRFSGSGSGTDFSLTISRLEPEDFAVYYCQQYGTSQSTFGQGTRLEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEEVQLVQSGGEVKRPGSSVTVSCKATGGTFSTLAFNWVRQAPGQGPEWMGGIVPLFSIVNYGQKFQGRLTIRADKSTTTVFLDLSGLTSADTATYYCAREGEGWFGKPLRAFEFWGQGTVITVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSC'

wd='/Users/cmaillie/Dropbox (Scripps Research)/mper_ab_manuscript_y3/pngs/'


l_chain_4e10_seq = 'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEP'
l_chain_residues = split_seq2list(l_chain_4e10_seq)
 
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC'
chain_residues = split_seq2list(chain_pgzl1_seq)
print(len(chain_pgzl1_seq))
plot_interaction_profile(backmap_med3_totals_4e10[:,0]/len(backmap_med3_4e10[0]),
                         backmap_med3_totals_4e10[:,1]/len(backmap_med3_4e10[0]),
                         backmap_med3_totals_4e10[:,2]/len(backmap_med3_4e10[0]),
                         l_chain_residues, wd+'4e10_backmap_med3_interaction_profile')


# In[60]:


backmap_med1_totals_4e10 = np.array(backmap_med1_totals_4e10)
backmap_med2_totals_4e10 = np.array(backmap_med2_totals_4e10)
backmap_med3_totals_4e10 = np.array(backmap_med3_totals_4e10)


# In[75]:


wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/10e8/med1_v2/charmm-gui-6906372682/gromacs/"

backmap_med1_10e8 =[[] for _ in range(444)] #number of residues 
calc_res_depths(backmap_med1_10e8 ,
                wd+'400ns_all.pdb',
                wd+'analysis.dcd')
backmap_med1_totals_10e8  =[[] for _ in range(444)] 
print(len(backmap_med1_10e8 ))
for i in range(len(backmap_med1_10e8 )):
    #acyl location counts ]
    acyl_counts = backmap_med1_10e8[i].count(1)
    backmap_med1_totals_10e8[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = backmap_med1_10e8[i].count(2)
    backmap_med1_totals_10e8[i].append(glycerol_counts)
    #po4 
    po4_counts = backmap_med1_10e8[i].count(3)
    backmap_med1_totals_10e8[i].append(po4_counts)
    #water
    water_counts = backmap_med1_10e8 [i].count(4)
    backmap_med1_totals_10e8[i].append(water_counts)


# In[76]:


wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/10e8/medoid2/"

backmap_med2_10e8 =[[] for _ in range(444)] #number of residues 
calc_res_depths(backmap_med2_10e8 ,
                wd+'500ns_all.pdb',
                wd+'analysis.dcd')
backmap_med2_totals_10e8  =[[] for _ in range(444)] 
print(len(backmap_med2_10e8 ))
for i in range(len(backmap_med2_10e8 )):
    #acyl location counts ]
    acyl_counts = backmap_med2_10e8[i].count(1)
    backmap_med2_totals_10e8[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = backmap_med2_10e8[i].count(2)
    backmap_med2_totals_10e8[i].append(glycerol_counts)
    #po4 
    po4_counts = backmap_med2_10e8[i].count(3)
    backmap_med2_totals_10e8[i].append(po4_counts)
    #water
    water_counts = backmap_med2_10e8 [i].count(4)
    backmap_med2_totals_10e8[i].append(water_counts)


# In[73]:


wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/10e8/medoid3/"

backmap_med3_10e8 =[[] for _ in range(444)] #number of residues 
calc_res_depths(backmap_med3_10e8 ,
                wd+'500ns_all.pdb',
                wd+'analysis.dcd')
backmap_med3_totals_10e8  =[[] for _ in range(444)] 
print(len(backmap_med3_10e8 ))
for i in range(len(backmap_med3_10e8 )):
    #acyl location counts ]
    acyl_counts = backmap_med3_10e8[i].count(1)
    backmap_med3_totals_10e8[i].append(acyl_counts)
    #glycerol_counts 
    glycerol_counts = backmap_med3_10e8[i].count(2)
    backmap_med3_totals_10e8[i].append(glycerol_counts)
    #po4 
    po4_counts = backmap_med3_10e8[i].count(3)
    backmap_med3_totals_10e8[i].append(po4_counts)
    #water
    water_counts = backmap_med3_10e8 [i].count(4)
    backmap_med3_totals_10e8[i].append(water_counts)


# In[78]:


backmap_med1_totals_10e8 = np.array(backmap_med1_totals_10e8)
backmap_med2_totals_10e8 = np.array(backmap_med2_totals_10e8)
backmap_med3_totals_10e8 = np.array(backmap_med3_totals_10e8)


# In[ ]:


plot_interaction_profile_10e8_v2


# In[79]:


wd='/Users/cmaillie/Dropbox (Scripps Research)/mper_ab_manuscript_y3/pngs/'

l_chain_4e10_seq = 'AAAEIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEP'
l_chain_residues = split_seq2list(l_chain_4e10_seq)
 
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC'
chain_residues = split_seq2list(chain_pgzl1_seq)
print(len(chain_pgzl1_seq))
plot_interaction_profile_10e8_v2(backmap_med1_totals_10e8[:,0]/len(backmap_med1_10e8[0]),
                         backmap_med1_totals_10e8[:,1]/len(backmap_med1_10e8[0]),
                         backmap_med1_totals_10e8[:,2]/len(backmap_med1_10e8[0]),
                         l_chain_residues, wd+'10e8_backmap_med1_interaction_profile')


# In[80]:


wd='/Users/cmaillie/Dropbox (Scripps Research)/mper_ab_manuscript_y3/pngs/'

l_chain_4e10_seq = 'AAAEIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEP'
l_chain_residues = split_seq2list(l_chain_4e10_seq)
 
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC'
chain_residues = split_seq2list(chain_pgzl1_seq)
print(len(chain_pgzl1_seq))
plot_interaction_profile_10e8_v2(backmap_med2_totals_10e8[:,0]/len(backmap_med2_10e8[0]),
                         backmap_med2_totals_10e8[:,1]/len(backmap_med2_10e8[0]),
                         backmap_med2_totals_10e8[:,2]/len(backmap_med2_10e8[0]),
                         l_chain_residues, wd+'10e8_backmap_med2_interaction_profile')


# In[81]:


wd='/Users/cmaillie/Dropbox (Scripps Research)/mper_ab_manuscript_y3/pngs/'

l_chain_4e10_seq = 'AAAEIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEQVQLVQSGAEVKRPGSSVTVSCKASGGSFSTYALSWVRQAPGRGLEWMGGVIPLLTITNYAPRFQGRITITADRSTSTAYLELNSLRPEDTAVYYCAREGTTGWGWLGKPIGAFAHWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEP'
l_chain_residues = split_seq2list(l_chain_4e10_seq)
 
#'EIVLTQSPGTQSLSPGERATLSCRASQSVGNNKLAWYQQRPGQAPRLLIYGASSRPSGVADRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGQSLSTFGQGTKVEVKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC'
chain_residues = split_seq2list(chain_pgzl1_seq)
print(len(chain_pgzl1_seq))
plot_interaction_profile_10e8_v2(backmap_med3_totals_10e8[:,0]/len(backmap_med3_10e8[0]),
                         backmap_med3_totals_10e8[:,1]/len(backmap_med3_10e8[0]),
                         backmap_med3_totals_10e8[:,2]/len(backmap_med3_10e8[0]),
                         l_chain_residues, wd+'10e8_backmap_med3_interaction_profile')

