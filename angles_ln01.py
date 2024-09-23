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


def calcRMS(x, axis=None):
    return np.sqrt(np.nanmean(x**2, axis=axis))

#MM
def skew(v):
    if len(v) == 4: 
        v=v[:3]/v[3]
    skv = np.roll(np.roll(np.diag(v.flatten()), 1, 1), -1, 0)
    return skv - skv.T

#MM
def VectorAlign(a, b):
    u=np.cross(a,b)
    s = LA.norm(u)
    c=np.dot(a,b)
    skw=skew(u)
    return np.identity(3) + skw + (1/(1+c)) * np.dot(skw, skw)


#from MM 
def planeFit(points):
	points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
	assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
	ctr = points.mean(axis=1)
	x = points - ctr[:,np.newaxis]
	M = np.dot(x, x.T) # Could also use np.cov(x) here.
	return ctr, svd(M)[0][:,-1]


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


# In[5]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_6snd_ppm_02/gromacs/'
input_pdb = wd+'1000ns_out.pdb'
pdb = parsePDB(input_pdb)
fab = pdb.select('name CA')
print(len(fab))
for i in fab:
    print(i.getResname(), i.getResnum())


# In[3]:


#method to calculate angles of ab - all atom (4e10 & pgzl1 )


def getAngle_ln01_AA(pdb):
    
    #selection strings 
    mem_selection_str = 'resname POPC POPA CHOL'
    fab_selection_str = 'name CA'
    #read in pdb
    input_pdb = pdb #parsePDB(pdb_fp)
    #move pdb to origin 
    moveAtoms(input_pdb, to=np.zeros(3))
    membrane = input_pdb.select(mem_selection_str)
    fab = input_pdb.select(fab_selection_str)

    #align so psuedo axis is aligned to positive z direction 
    #define axis through fab center and membrane center - force fab to point upwards in z direction 
    #print(len(fab.select('resnum 41 and chain A or resnum 41 and chain B')))
    pseudo_fab_cen = calcCenter(fab.select('resnum 41 and chain A or resnum 41 and chain B'))
    
    membrane_cen = calcCenter(membrane)
    psuedo_central_ax = np.array(pseudo_fab_cen-membrane_cen)
    #must normalize axis vector before transforming 
    psuedo_central_ax_norm = psuedo_central_ax / np.linalg.norm(psuedo_central_ax)
    rotation = VectorAlign(psuedo_central_ax_norm, np.array([0, 0, 1]))
    transformCentralAx = Transformation(rotation, np.zeros(3))
    applyTransformation(transformCentralAx, input_pdb)

    
    
    #align membrane plane normal to Z axis for angle calcualtion 
    mem_plane = planeFit(np.transpose(membrane.getCoords()))
    mem_plane_normal = mem_plane[1]
    #in all atom systems - bound to bottom membrane so must flip 
    if mem_plane_normal[2]>0: 
        rotation = VectorAlign(mem_plane_normal, np.array([0, 0, 1]))
        #define transformation based on rotaiton 
        transformCentralAx = Transformation(rotation, np.zeros(3))
        #apply transofrmation  to entire system 
        applyTransformation(transformCentralAx, input_pdb)
    else:
        rotation = VectorAlign(mem_plane_normal, np.array([0, 0, -1]))
        #define transformation based on rotaiton 
        transformCentralAx = Transformation(rotation, np.zeros(3))
        #apply transofrmation  to entire system 
        applyTransformation(transformCentralAx, input_pdb)

    mem_plane = planeFit(np.transpose(membrane.getCoords()))
    mem_plane_normal = mem_plane[1]    
    
    #calc angle b/w approach angle & top phos plane
    res94 = fab.select('resname THR and resnum 94')
    #print(res94)
    res320 = fab.select('resname GLY and resnum 99')
    point_cdr_loops = calcCenter(res94+res320)
    res41 = fab.select('resnum 41 and chain A or resnum 41 and chain B') 

    point_center_fab = calcCenter(res41)     
    point_xy_3 = np.array([point_cdr_loops[0], point_cdr_loops[1], 200 ]) #-50
    var_domain_vect = np.array(point_center_fab-point_cdr_loops)
    xy_norm_vect = np.array(point_xy_3-point_cdr_loops)
    angle_approach = round(angle_between(var_domain_vect, xy_norm_vect), 2)


    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab.select("resnum 107 and chain A").getCoords()
    #print(point_l_edge)
    #print(point_l_edge[0])
    #print(len(fab.select("resnum 107 and chain A")))
    point_h_edge = fab.select("resnum 113 and chain B").getCoords()
    #print(len(fab.select("resnum 113 and chain B")))
    point_xz_3 = np.array([point_l_edge[0][0], point_l_edge[0][1], 200])#-50
    short_ax_vect = np.array(point_l_edge[0]-point_h_edge[0])
    #print(point_xz_3)
    xz_norm_vect = np.array(point_xz_3-point_l_edge[0])
    angle_rotation = round(angle_between(short_ax_vect, xz_norm_vect), 2)
    
    #angles are calcualted relative to psuedo normal vector  
    #angle of approach will be in range [0,90] after reflecting if angle is >90 
    #subtract reflected angle from 90 to find angle to mem plane
    if angle_approach>90:
        angle_approach=180-angle_approach 
    #angle of rotation will be in range [0, 180] 
    
    #report angles as angle to membrane by taking 90-angle 
    return [round(90-angle_approach,2), round(90-angle_rotation, 2)] 
    
    
    
    


# In[4]:


#method to calculate angles of ab - all atom (4e10 & pgzl1 )


def getAngle_ln01_AA(pdb):
    
    #selection strings 
    mem_selection_str = 'resname POPC POPA CHL1'
    fab_selection_str = 'name CA'
    #read in pdb
    input_pdb = pdb #parsePDB(pdb_fp)
    #move pdb to origin 
    moveAtoms(input_pdb, to=np.zeros(3))
    membrane = input_pdb.select(mem_selection_str)
    fab = input_pdb.select(fab_selection_str)

    #align so psuedo axis is aligned to positive z direction 
    #define axis through fab center and membrane center - force fab to point upwards in z direction 
    #print(len(fab.select('resnum 41 and chain A or resnum 41 and chain B')))
    pseudo_fab_cen = calcCenter(fab.select('resnum 41 and chain A or resnum 41 and chain B'))
    
    membrane_cen = calcCenter(membrane)
    psuedo_central_ax = np.array(pseudo_fab_cen-membrane_cen)
    #must normalize axis vector before transforming 
    psuedo_central_ax_norm = psuedo_central_ax / np.linalg.norm(psuedo_central_ax)
    rotation = VectorAlign(psuedo_central_ax_norm, np.array([0, 0, 1]))
    transformCentralAx = Transformation(rotation, np.zeros(3))
    applyTransformation(transformCentralAx, input_pdb)

    
    
    #align membrane plane normal to Z axis for angle calcualtion 
    mem_plane = planeFit(np.transpose(membrane.getCoords()))
    mem_plane_normal = mem_plane[1]
    #in all atom systems - bound to bottom membrane so must flip 
    if mem_plane_normal[2]>0: 
        rotation = VectorAlign(mem_plane_normal, np.array([0, 0, 1]))
        #define transformation based on rotaiton 
        transformCentralAx = Transformation(rotation, np.zeros(3))
        #apply transofrmation  to entire system 
        applyTransformation(transformCentralAx, input_pdb)
    else:
        rotation = VectorAlign(mem_plane_normal, np.array([0, 0, -1]))
        #define transformation based on rotaiton 
        transformCentralAx = Transformation(rotation, np.zeros(3))
        #apply transofrmation  to entire system 
        applyTransformation(transformCentralAx, input_pdb)

    mem_plane = planeFit(np.transpose(membrane.getCoords()))
    mem_plane_normal = mem_plane[1]    
    
    #calc angle b/w approach angle & top phos plane
    res94 = fab.select('resname THR and resnum 94')
    #print(res94)
    res320 = fab.select('resname GLY and resnum 99')
    point_cdr_loops = calcCenter(res94+res320)
    res41 = fab.select('resnum 41 and chain A or resnum 41 and chain B') 

    point_center_fab = calcCenter(res41)     
    point_xy_3 = np.array([point_cdr_loops[0], point_cdr_loops[1], 200 ]) #-50
    var_domain_vect = np.array(point_center_fab-point_cdr_loops)
    xy_norm_vect = np.array(point_xy_3-point_cdr_loops)
    angle_approach = round(angle_between(var_domain_vect, xy_norm_vect), 2)


    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab.select("resnum 107 and chain A").getCoords()
    #print(point_l_edge)
    #print(point_l_edge[0])
    #print(len(fab.select("resnum 107 and chain A")))
    point_h_edge = fab.select("resnum 113 and chain B").getCoords()
    #print(len(fab.select("resnum 113 and chain B")))
    point_xz_3 = np.array([point_l_edge[0][0], point_l_edge[0][1], 200])#-50
    short_ax_vect = np.array(point_l_edge[0]-point_h_edge[0])
    #print(point_xz_3)
    xz_norm_vect = np.array(point_xz_3-point_l_edge[0])
    angle_rotation = round(angle_between(short_ax_vect, xz_norm_vect), 2)
    
    #angles are calcualted relative to psuedo normal vector  
    #angle of approach will be in range [0,90] after reflecting if angle is >90 
    #subtract reflected angle from 90 to find angle to mem plane
    if angle_approach>90:
        angle_approach=180-angle_approach 
    #angle of rotation will be in range [0, 180] 
    
    #report angles as angle to membrane by taking 90-angle 
    return [round(90-angle_approach,2), round(90-angle_rotation, 2)] 
    
    
    
    


# In[5]:


def trajectory_angles_ln01_aa(input_pdb, dcd_traj, output_name): 
    #example output_name input varible: "10e8_ppm"
    angle_running = []
    #note; all atom sims have fab in bottom membrane layer, cg have fab in top 
    input_pdb  = input_pdb 
    dcd_traj  = dcd_traj 
    input_pdb = parsePDB(input_pdb)
    dcd = DCDFile(dcd_traj)
    dcd.setCoords(input_pdb)
    dcd.link(input_pdb)
    dcd.reset()
    #for each frame, calculate angle 
    for i,frame in enumerate(dcd):
        pdb = frame.getAtoms()
        angle_running.append(getAngle_ln01_AA(pdb)) 
    #save to csv file 
    file_out = output_name + "_angles.npy"
    with open(file_out, 'wb') as f:
        np.save(f, angle_running)
    f.close()
    #np.savetxt(file_out, angle_running, delimiter =",",fmt ='% s')
    return print("finished " + file_out)


# In[30]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_6snd_ppm_02/gromacs/'
input_pdb = wd+'1000ns_out.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_ln01_aa(input_pdb, dcd_traj, 'ln01_6snd_ppm_02')


# In[38]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_6sne_ppm_03/gromacs/'
input_pdb = wd+'500ns_out.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_ln01_aa(input_pdb, dcd_traj, 'ln01_6sne_ppm_03')


# In[40]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_6sne_ppm_04/gromacs/'
input_pdb = wd+'1000ns_out.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_ln01_aa(input_pdb, dcd_traj, 'ln01_6sne_ppm_04')


# In[45]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/LN01_OPM_rep1/'
input_pdb = wd+'100ns_all.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_ln01_aa(input_pdb, dcd_traj, 'LN01_OPM_rep1')


# In[49]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_tm_mper_02/'
input_pdb = wd+'1000ns_out.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_ln01_aa(input_pdb, dcd_traj, 'ln01_tm_mper_02')


# In[55]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_tm_mper_04/'
input_pdb = wd+'100ns.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_ln01_aa(input_pdb, dcd_traj, 'ln01_tm_mper_04')


# In[58]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_tm_mper_05/gromacs/'
input_pdb = wd+'500ns_out.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_ln01_aa(input_pdb, dcd_traj, 'ln01_tm_mper_05')


# In[56]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/LN01_TM-MPER/'
input_pdb = wd+'1000ns_all.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_ln01_aa(input_pdb, dcd_traj, 'LN01_TM-MPER')


# In[63]:


plt.rcParams['font.sans-serif']=  ['Arial'] 
plt.rcParams["figure.figsize"] = (12,4)

ln01_tm_mper_02_angles = np.load('ln01_tm_mper_02_angles.npy', mmap_mode='r')

ln01_tm_mper_04_angles = np.load('ln01_tm_mper_04_angles.npy', mmap_mode='r')
 
ln01_tm_mper_05_angles = np.load('ln01_tm_mper_05_angles.npy', mmap_mode='r')

LN01_TM_MPER_angles = np.load('LN01_TM-MPER_angles.npy', mmap_mode='r')
    
#angles_410_ppm = np.load('4e10_ppm_angles.npy', mmap_mode='r')

plt.plot(ln01_tm_mper_02_angles[:,0], color='black', linewidth=0.5)
plt.plot(ln01_tm_mper_02_angles[0:,1], color='red',  linewidth=0.5)

plt.plot(ln01_tm_mper_04_angles[:,0], color='black', linewidth=0.5)
plt.plot(ln01_tm_mper_04_angles[0:,1], color='red',  linewidth=0.5)

plt.plot(ln01_tm_mper_05_angles[:,0], color='black', linewidth=0.5)
plt.plot(ln01_tm_mper_05_angles[0:,1], color='red',  linewidth=0.5)

plt.plot(LN01_TM_MPER_angles[:,0], color='black', linewidth=0.5)
plt.plot(LN01_TM_MPER_angles[0:,1], color='red',  linewidth=0.5)


plt.yticks([ -90, 0, 90], fontsize=55)
plt.ylim(-100,100)
plt.xticks([0, 2500, 5000], [0,  0.5,  1],  fontsize=55)
plt.xlim([-1, 5031])


# In[61]:


plt.rcParams['font.sans-serif']=  ['Arial'] 
plt.rcParams["figure.figsize"] = (12,4)

ln01_6snd_ppm_02_angles = np.load('ln01_6snd_ppm_02_angles.npy', mmap_mode='r')

ln01_6sne_ppm_03_angles = np.load('ln01_6sne_ppm_03_angles.npy', mmap_mode='r')
 
ln01_6sne_ppm_04_angles = np.load('ln01_6sne_ppm_04_angles.npy', mmap_mode='r')


LN01_OPM_rep1_angles = np.load('LN01_OPM_rep1_angles.npy', mmap_mode='r')
#duplicate balues to make match 1 us timeline - skipped every 2 frames in dcd 
LN01_OPM_rep1_angles_ext = [] 
for i in LN01_OPM_rep1_angles: 
    LN01_OPM_rep1_angles_ext.append(i)
    LN01_OPM_rep1_angles_ext.append(i)


with open('LN01_OPM_rep1_ext_angles.npy', 'wb') as f:
    np.save(f, LN01_OPM_rep1_angles_ext)
    f.close()
LN01_OPM_rep1_ext_angles = np.load('LN01_OPM_rep1_ext_angles.npy', mmap_mode='r')
    
#angles_410_ppm = np.load('4e10_ppm_angles.npy', mmap_mode='r')

plt.plot(ln01_6snd_ppm_02_angles[:,0], color='black', linewidth=0.5)
plt.plot(ln01_6snd_ppm_02_angles[0:,1], color='red',  linewidth=0.5)

plt.plot(ln01_6sne_ppm_03_angles[:,0], color='black', linewidth=0.5)
plt.plot(ln01_6sne_ppm_03_angles[0:,1], color='red',  linewidth=0.5)

plt.plot(ln01_6sne_ppm_04_angles[:,0], color='black', linewidth=0.5)
plt.plot(ln01_6sne_ppm_04_angles[0:,1], color='red',  linewidth=0.5)

plt.plot(LN01_OPM_rep1_ext_angles[:,0], color='black', linewidth=0.5)
plt.plot(LN01_OPM_rep1_ext_angles[0:,1], color='red',  linewidth=0.5)


plt.yticks([ -90, 0, 90], fontsize=55)
plt.ylim(-100,100)
plt.xticks([0, 2500, 5000], [0,  0.5,  1],  fontsize=55)
plt.xlim([-1, 5031])


# In[6]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_ppm_6snd_wyf_rep01/gromacs/'
input_pdb = wd+'500ns_all.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_ln01_aa(input_pdb, dcd_traj, 'ln01_ppm_6snd_wyf_rep01')


# In[7]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_ppm_6snd_wyf_rep02/gromacs/'
input_pdb = wd+'500ns_all.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_ln01_aa(input_pdb, dcd_traj, 'ln01_ppm_6snd_wyf_rep02')


# In[8]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_ppm_6snd_wyf_rep03/gromacs/'
input_pdb = wd+'500ns_all.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_ln01_aa(input_pdb, dcd_traj, 'ln01_ppm_6snd_wyf_rep03')


# In[9]:



ln01_ppm_6snd_wyf_rep01 = np.load('ln01_ppm_6snd_wyf_rep01_angles.npy', mmap_mode='r')

ln01_ppm_6snd_wyf_rep03 = np.load('ln01_ppm_6snd_wyf_rep02_angles.npy', mmap_mode='r')
 
ln01_ppm_6snd_wyf_rep03 = np.load('ln01_ppm_6snd_wyf_rep03_angles.npy', mmap_mode='r')


# In[10]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_TM_ppm_6snd_wyf_rep02/gromacs/'
input_pdb = wd+'500ns_all.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_ln01_aa(input_pdb, dcd_traj, 'ln01_TM_ppm_6snd_wyf_rep02')


# In[11]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/ln01_v2/ln01_TM_ppm_6snd_wyf_rep03/gromacs/'
input_pdb = wd+'500ns_all.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_ln01_aa(input_pdb, dcd_traj, 'ln01_TM_ppm_6snd_wyf_rep03')

