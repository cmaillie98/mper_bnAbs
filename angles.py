#!/usr/bin/env python
# coding: utf-8

# In[26]:


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


# In[27]:


pdb = parsePDB('/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_ppm/final_analysis_input.pdb')
fab_aa = pdb.select('name CA')
cg_pdb = parsePDB('/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/4e10/analysis_input.pdb')
fab_cg = cg_pdb.select('name BB')


print(fab_aa[94].getResname(), fab_cg[94].getResname())
print(fab_aa[320].getResname(), fab_cg[320].getResname())
print(fab_aa[41].getResname(), fab_cg[41].getResname())
print(fab_aa[256].getResname(), fab_cg[256].getResname())
print(fab_aa[107].getResname(), fab_cg[107].getResname())
print(fab_aa[340].getResname(), fab_cg[340].getResname())


# In[8]:


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


# In[22]:


#method to calculate angles of ab - all atom (4e10 & pgzl1 )


def getAngle_4e10_AA(pdb):
    
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
    pseudo_fab_cen = calcCenter(fab.select('resnum 41 or resnum 256'))
    #print(fab.select('resnum 41 or resnum 256').getResnames())
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
    #print(mem_plane_normal)
    #calc angle b/w approach angle & top phos plane
    res94 = fab[320]
    res320 = fab[94]
    #print(fab[94].getResname())
    #print(fab[320].getResname())
    point_cdr_loops = calcCenter(res94+res320)
    res41 = fab[256]
    res256 = fab[41]
    #print(fab[41].getResname())
    #print(fab[256].getResname())
    point_center_fab = calcCenter(res41+res256)     
    point_xy_3 = np.array([point_cdr_loops[0], point_cdr_loops[1], 200 ]) #-50
    var_domain_vect = np.array(point_center_fab-point_cdr_loops)
    xy_norm_vect = np.array(point_xy_3-point_cdr_loops)
    angle_approach = round(angle_between(var_domain_vect, xy_norm_vect), 2)
    #print(var_domain_vect)
    #print(xy_norm_vect)
    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[340].getCoords()
    point_h_edge = fab[107].getCoords()
    #print(fab[107].getResname())
    #print(fab[340].getResname())
    
    point_xz_3 = np.array([point_l_edge[0], point_l_edge[1], 200])#-50
    short_ax_vect = np.array(point_l_edge-point_h_edge)
    xz_norm_vect = np.array(point_xz_3-point_l_edge)
    #print(short_ax_vect)
    #print(xz_norm_vect)
    angle_rotation = round(angle_between(short_ax_vect, xz_norm_vect), 2)
    
    #angles are calcualted relative to psuedo normal vector  
    #angle of approach will be in range [0,90] after reflecting if angle is >90 
    #subtract reflected angle from 90 to find angle to mem plane
    if angle_approach>90:
        angle_approach=180-angle_approach 
    #angle of rotation will be in range [0, 180] 
    
    #report angles as angle to membrane by taking 90-angle 
    return [round(90-angle_approach,2), round(90-angle_rotation, 2)] 
    
    
    
    


# In[53]:


#check residues in all atom final_analysis_input.pdb are same as 500ns_all.pdb 
#confirmed this - 11.17.22
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/4e10/medoid1/'
input_pdb = parsePDB(wd+"500ns_all.pdb")
fab_selection_str = 'name CA'
fab = input_pdb.select(fab_selection_str)
res94 = fab[94]
res320 = fab[320]

res41 = fab[41]
res256 = fab[256]
res107 = fab[107]
res340 = fab[340]
print(res94.getResname())
print(res320.getResname())
print(res41.getResname())
print(res256.getResname())
print(res107.getResname())
print(res340.getResname())


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_ppm/'
input_pdb = parsePDB(wd+"final_analysis_input.pdb")
fab_selection_str = 'name CA'
fab = input_pdb.select(fab_selection_str)
res94 = fab[94]
res320 = fab[320]

res41 = fab[41]
res256 = fab[256]
res107 = fab[107]
res340 = fab[340]
print(res94.getResname())
print(res320.getResname())
print(res41.getResname())
print(res256.getResname())
print(res107.getResname())
print(res340.getResname())


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/4e10/medoid1/'
input_pdb = parsePDB(wd+"step5_input.pdb")
fab_selection_str = 'name CA'
fab = input_pdb.select(fab_selection_str)
res94 = fab[94]
res320 = fab[320]

res41 = fab[41]
res256 = fab[256]
res107 = fab[107]
res340 = fab[340]
print(res94.getResname())
print(res320.getResname())
print(res41.getResname())
print(res256.getResname())
print(res107.getResname())
print(res340.getResname())


# In[23]:


def trajectory_angles_4e10_aa(input_pdb, dcd_traj, output_name): 
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
        angle_running.append(getAngle_4e10_AA(pdb)) 
    #save to csv file 
    file_out = output_name + "_angles.npy"
    with open(file_out, 'wb') as f:
        np.save(f, angle_running)
    f.close()
    #np.savetxt(file_out, angle_running, delimiter =",",fmt ='% s')
    return print("finished " + file_out)


# In[ ]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_ppm/'
input_pdb = wd+'final_analysis_input.pdb'
dcd_traj = wd+'final_analysis_traj.dcd'
trajectory_angles_4e10_aa(input_pdb, dcd_traj, '4e10_ppm')


# In[7]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_p15/'
input_pdb = wd+'final_analysis_input.pdb'
dcd_traj = wd+'final_analysis_traj.dcd'
trajectory_angles_4e10_aa(input_pdb, dcd_traj, '4e10_p15')


# In[8]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_n15/'
input_pdb = wd+'final_analysis_input.pdb'
dcd_traj = wd+'final_analysis_traj.dcd'
trajectory_angles_4e10_aa(input_pdb, dcd_traj, '4e10_n15')


# In[9]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_ppm_rep/'
input_pdb = wd+'final_analysis_input.pdb'
dcd_traj = wd+'final_analysis_traj.dcd'
trajectory_angles_4e10_aa(input_pdb, dcd_traj, '4e10_ppm_rep')


# In[9]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/4e10/medoid1/'
input_pdb = wd+'500ns_all.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_4e10_aa(input_pdb, dcd_traj, '4e10_med1_backmapped')


# In[10]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/4e10/medoid2/'
input_pdb = wd+'500ns_all.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_4e10_aa(input_pdb, dcd_traj, '4e10_med2_backmapped')


# In[11]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/4e10/medoid3/'
input_pdb = wd+'500ns_all.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_4e10_aa(input_pdb, dcd_traj, '4e10_med3_backmapped')


# In[19]:


angle_4e10_med1 = np.load('4e10_med1_backmapped_angles.npy', mmap_mode='r')
print(angle_4e10_med1[0:20])
plt.plot(angle_4e10_med1)


# In[18]:


angle_4e10_med2 = np.load('4e10_med2_backmapped_angles.npy', mmap_mode='r')
print(angle_4e10_med2[0:20])
plt.plot(angle_4e10_med2)


# In[17]:


angle_4e10_med3 = np.load('4e10_med3_backmapped_angles.npy', mmap_mode='r')
print(angle_4e10_med3[0:20])
plt.plot(angle_4e10_med3)


# In[49]:


#method to calculate angles of ab to membrane - coarse grain (4e10) 

def getAngle_4e10_cg_spont(pdb):
    
    #selection strings 
    mem_selection_str = 'resname POPC POPA CHOL'
    fab_selection_str = 'name BB'
    #read in pdb
    input_pdb = pdb 
    #move pdb to origin 
    moveAtoms(input_pdb, to=np.zeros(3))
    #writePDB('system_zeroed.pdb', input_pdb)
    #pre select po4 membrane based on geometric constraints to avoid micelle lipids 
    po4_membrane = input_pdb.select('name PO4 and z>5') # and z>5
    
    #mem_coords = po4_membrane.getCoords()[2]
    #writePDB('po4_membrane.pdb', po4_membrane)
    
    #pseudo axis is center of membrane to center of fab 
    membrane = input_pdb.select(mem_selection_str)
    fab = input_pdb.select(fab_selection_str)
    
    #align so psuedo axis is aligned to positive z direction 
    #define axis through fab center and membrane center - force fab to point upwards in z direction 
    pseudo_fab_cen = calcCenter(fab[41]+fab[256])
    membrane_cen = calcCenter(membrane)
    psuedo_central_ax = np.array(pseudo_fab_cen-membrane_cen)
    
    #must normalize axis vector before transforming 
    psuedo_central_ax_norm = psuedo_central_ax / np.linalg.norm(psuedo_central_ax)
    #align to unit vector in z direction 
    rotation = VectorAlign(psuedo_central_ax_norm, np.array([0, 0, 1]))
    transformCentralAx = Transformation(rotation, np.zeros(3))
    applyTransformation(transformCentralAx, input_pdb)
    #writePDB('cg_transform_fabUp.pdb', input_pdb)
    
    
    mem_plane = planeFit(np.transpose(po4_membrane.getCoords()))
    mem_plane_normal = mem_plane[1]
    #print("Plane Norm: ", mem_plane_normal)
    
    #define rotation as alignement from psueo-central-axis to z plane norm vect 
    #if normal vecotr of mem plane in positive, align to positive unit vecotr in Z
    #if normal vecotr of mem plane in negative, align to positive unit vecotr in Z
    
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
    
    mem_plane = planeFit(np.transpose(po4_membrane.getCoords()))
    mem_plane_normal = mem_plane[1]    
#     print("Aligned Plane Norm: ", mem_plane_normal)
#     writePDB('system_transformed.pdb', input_pdb)
    #select 2 fab axes 
    
    #calc angle b/w approach angle & top phos plane
    fab = input_pdb.select(fab_selection_str)
    res94 = fab[94]
    res320 = fab[320]
    point_cdr_loops = calcCenter(res94+res320)
    res41 = fab[41]
    res256 = fab[256]
    point_center_fab = calcCenter(res41+res256)     
    point_xy_3 = np.array([point_cdr_loops[0], point_cdr_loops[1], 200]) 
    var_domain_vect = np.array(point_center_fab-point_cdr_loops)
    xy_norm_vect = np.array(point_xy_3-point_cdr_loops)
    angle_approach = round(angle_between(var_domain_vect, xy_norm_vect), 2)
#     #angle of variable domain to membrane in xy space 

    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[107].getCoords()
    point_h_edge = fab[340].getCoords()
    point_xz_3 = np.array([point_l_edge[0], point_l_edge[1], 200])
    short_ax_vect = np.array(point_l_edge - point_h_edge )
    xz_norm_vect = np.array(point_xz_3-point_l_edge)
    #angle of short axis vector to 'plane normal' reference vector 
    angle_rotation = round(angle_between(short_ax_vect, xz_norm_vect), 2) 
    #print(angle(var_domain_vect, xy_norm_vect), angle(short_ax_vect, xy_norm_vect) )
    #select fab axes 
    
    #angle are calcualted relative to psuedo normal vector  
    #angle of approach will be in range [0,90] after reflecting if angle is >90 
    #subtract reflected angle from 90 to find angle to mem plane
    if angle_approach>90:
        angle_approach=180-angle_approach 
    #angle of rotation will be in range [0, 180] 
    
    #report angles as angle to membrane by taking 90-angle 
    return [round(90-angle_approach,2), round(90-angle_rotation, 2)] 


# In[24]:


def trajectory_angles_4e10_spont_cg(input_pdb, dcd_traj, output_name): 
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
        angle_running.append(getAngle_4e10_cg_spont(pdb)) 
    #save to csv file 
    file_out = output_name + "_angles.npy"
    with open(file_out, 'wb') as f:
        np.save(f, angle_running)
    f.close()
    #np.savetxt(file_out, angle_running, delimiter =",",fmt ='% s')
    return print("finished " + file_out) 


# In[25]:


#method to calculate angles of ab to membrane - coarse grain (4e10) 

def getAngle_4e10_cg_embedded(pdb):
    
    #selection strings 
    mem_selection_str = 'resname POPC POPA CHOL'
    fab_selection_str = 'name BB'
    #read in pdb
    input_pdb = pdb 
    #move pdb to origin 
    moveAtoms(input_pdb, to=np.zeros(3))
    #writePDB('system_zeroed.pdb', input_pdb)
    #pre select po4 membrane based on geometric constraints to avoid micelle lipids 
    po4_membrane = input_pdb.select('name PO4 and z<90') # and z>5
    
    #mem_coords = po4_membrane.getCoords()[2]
    #writePDB('po4_membrane.pdb', po4_membrane)
    
    #pseudo axis is center of membrane to center of fab 
    membrane = input_pdb.select(mem_selection_str)
    fab = input_pdb.select(fab_selection_str)
    
    #align so psuedo axis is aligned to positive z direction 
    #define axis through fab center and membrane center - force fab to point upwards in z direction 
    pseudo_fab_cen = calcCenter(fab[41]+fab[256])
    membrane_cen = calcCenter(membrane)
    psuedo_central_ax = np.array(pseudo_fab_cen-membrane_cen)
    
    #must normalize axis vector before transforming 
    psuedo_central_ax_norm = psuedo_central_ax / np.linalg.norm(psuedo_central_ax)
    #align to unit vector in z direction 
    rotation = VectorAlign(psuedo_central_ax_norm, np.array([0, 0, 1]))
    transformCentralAx = Transformation(rotation, np.zeros(3))
    applyTransformation(transformCentralAx, input_pdb)
    #writePDB('cg_transform_fabUp.pdb', input_pdb)
    
    
    mem_plane = planeFit(np.transpose(po4_membrane.getCoords()))
    mem_plane_normal = mem_plane[1]
    #print("Plane Norm: ", mem_plane_normal)
    
    #define rotation as alignement from psueo-central-axis to z plane norm vect 
    #if normal vecotr of mem plane in positive, align to positive unit vecotr in Z
    #if normal vecotr of mem plane in negative, align to positive unit vecotr in Z
    
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
    
    mem_plane = planeFit(np.transpose(po4_membrane.getCoords()))
    mem_plane_normal = mem_plane[1]    
#     print("Aligned Plane Norm: ", mem_plane_normal)
#     writePDB('system_transformed.pdb', input_pdb)
    #select 2 fab axes 
    
    #calc angle b/w approach angle & top phos plane
    fab = input_pdb.select(fab_selection_str)
    res94 = fab[94]
    res320 = fab[320]
    point_cdr_loops = calcCenter(res94+res320)
    res41 = fab[41]
    w = fab[256]
    point_center_fab = calcCenter(res41+res256)     
    point_xy_3 = np.array([point_cdr_loops[0], point_cdr_loops[1], 200]) 
    var_domain_vect = np.array(point_center_fab-point_cdr_loops)
    xy_norm_vect = np.array(point_xy_3-point_cdr_loops)
    angle_approach = round(angle_between(var_domain_vect, xy_norm_vect), 2)
#     #angle of variable domain to membrane in xy space 

    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[107].getCoords()
    point_h_edge = fab[340].getCoords()
    point_xz_3 = np.array([point_l_edge[0], point_l_edge[1], 200])
    short_ax_vect = np.array(point_l_edge - point_h_edge )
    xz_norm_vect = np.array(point_xz_3-point_l_edge)
    #angle of short axis vector to 'plane normal' reference vector 
    angle_rotation = round(angle_between(short_ax_vect, xz_norm_vect), 2) 
    #print(angle(var_domain_vect, xy_norm_vect), angle(short_ax_vect, xy_norm_vect) )
    #select fab axes 
    
    #angle are calcualted relative to psuedo normal vector  
    #angle of approach will be in range [0,90] after reflecting if angle is >90 
    #subtract reflected angle from 90 to find angle to mem plane
    if angle_approach>90:
        angle_approach=180-angle_approach 
    #angle of rotation will be in range [0, 180] 
    
    #report angles as angle to membrane by taking 90-angle 
    return [round(90-angle_approach,2), round(90-angle_rotation, 2)] 


# In[26]:


def trajectory_angles_4e10_cg_emedded(input_pdb, dcd_traj, output_name): 
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
        angle_running.append(getAngle_4e10_cg_embedded(pdb)) 
    #save to csv file 
    file_out = output_name + "_angles.npy"
    with open(file_out, 'wb') as f:
        np.save(f, angle_running)
    f.close()
    #np.savetxt(file_out, angle_running, delimiter =",",fmt ='% s')
    return print("finished " + file_out) 


# In[10]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/4e10/'
input_pdb = wd+'analysis_input.pdb'
dcd_traj = wd+'gar_top_contact.dcd'
trajectory_angles_4e10_spont_cg(input_pdb, dcd_traj, '4e10_spont_top_gar')


# In[120]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/4e10/'
input_pdb = wd+'analysis_input.pdb'
dcd_traj = wd+'ims_top_contact.dcd'
trajectory_angles_4e10_spont_cg(input_pdb, dcd_traj, '4e10_spont_top_ims')


# In[128]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/4e10/'
input_pdb = wd+'analysis_input.pdb'
dcd_traj = wd+'gar_bot_contact.dcd'
trajectory_angles_4e10_spont_cg(input_pdb, dcd_traj, '4e10_spont_bot_gar')


# In[129]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/4e10/'
input_pdb = wd+'analysis_input.pdb'
dcd_traj = wd+'ims_bot_contact.dcd'
trajectory_angles_4e10_spont_cg(input_pdb, dcd_traj, '4e10_spont_bot_ims')


# In[147]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_embedded/'
input_pdb = wd+'analysis.pdb'
dcd_traj = wd+'4e10_mem_contact_cen.dcd'
trajectory_angles_4e10_cg_emedded(input_pdb, dcd_traj, '4e10_embedded')


# In[6]:


def plot_angles(variable_domain_angles, short_axis_angles, starting_angles, color, darkcolor, prefix):
    #input: variable_domain_angles/short_axis_angles- np array of angles 
    #input: color - string of color code for plot 
    #prefix - string of prefix to add to plot name
    #starting_points - np array of variable & short ax angles 
    sns.set(rc={'figure.figsize':(8,8)})
    sns.set_style("whitegrid", {'axes.linewidth': 1, 'axes.edgecolor':'black'})
    a_plot = sns.scatterplot( x=variable_domain_angles, y=short_axis_angles,
                color=color, alpha=.2)
    a_plot = sns.kdeplot(variable_domain_angles, short_axis_angles, gridsize=100,
                         color=darkcolor, edgecolor="black")  # fill=True,
#     a_plot = sns.scatterplot( x=starting_angles[:,0], y=starting_angles[:,1],
#                  color='black', alpha=1,edgecolor="black" )
    a_plot.set_ylim([-95,95])
    a_plot.set_xlim([-5, 95])
    #control tick mark &  text size 
    xtick_loc = [0, 30, 60, 90]
    ytick_loc = [ -90, -45, 0, 45, 90]
    a_plot.set_xticks(xtick_loc)
    a_plot.set_yticks(ytick_loc)
    a_plot.tick_params(axis="x", labelsize=42)
    a_plot.tick_params(axis="y", labelsize=42)
    #save as png 
    plot_name = prefix+"_angles_kdeplot.png"
    a_plot.figure.savefig(plot_name, transparent=True )
    return 


# In[22]:


aa_starting_angles = [] 
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_ppm/'
input_pdb = wd+'final_analysis_input.pdb'
print(getAngle_4e10_AA(parsePDB(input_pdb)))
aa_starting_angles.append(getAngle_4e10_AA(parsePDB(input_pdb)))
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_p15/'
input_pdb = wd+'final_analysis_input.pdb'
print(getAngle_4e10_AA(parsePDB(input_pdb)))
aa_starting_angles.append(getAngle_4e10_AA(parsePDB(input_pdb)))
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_n15/'
input_pdb = wd+'final_analysis_input.pdb'
print(getAngle_4e10_AA(parsePDB(input_pdb)))
aa_starting_angles.append(getAngle_4e10_AA(parsePDB(input_pdb)))
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_ppm_rep/'
input_pdb = wd+'final_analysis_input.pdb'
print(getAngle_4e10_AA(parsePDB(input_pdb)))
aa_starting_angles.append(getAngle_4e10_AA(parsePDB(input_pdb)))


# In[14]:


angles_410_ppm = np.load('4e10_ppm_angles.npy', mmap_mode='r')
angles_410_p15 = np.load('4e10_p15_angles.npy', mmap_mode='r')
angles_410_n15 = np.load('4e10_n15_angles.npy', mmap_mode='r')
angles_410_ppm_rep = np.load('4e10_ppm_rep_angles.npy', mmap_mode='r')

aggregate_angles_4e10 = []
for i in angles_410_ppm:
    aggregate_angles_4e10.append(i)
for i in angles_410_p15:
    aggregate_angles_4e10.append(i)
for i in angles_410_n15:
    aggregate_angles_4e10.append(i)
for i in angles_410_ppm_rep:
    aggregate_angles_4e10.append(i)
aggregate_angles_4e10= np.array(aggregate_angles_4e10)
plot_angles(aggregate_angles_4e10[:,0], aggregate_angles_4e10[:,1], np.array(aa_starting_angles),
           '#55A3FF', '#2A517F', '4e10_aa')


# In[161]:


starting_angles = [[36.82,20.89],
                    [54.92,21.78],
                    [42.21,32.75],
                    [56.05,10.77],
                    [56.88,32.32],
                    [9.1,-67.59],
                    [34.06,-1.8],
                    [38.83,-0.43],
                    [6.88,59.27],
                    [37.48,-9.74],
                    [39.78,43.97],
                    [30.66,-3.05],
                    [29.85,-10.22],
                    [0.14,48.78],
                    [25.13,42.91],
                    [14.66,-52.38],
                    [58.74,13.77],
                    [45.59,-11.41],
                    [36.54,28.35],
                    [40.7,0.5],
                    [69.5,5.05],
                    [72.21,5.93],
                    [1.34,-25.92],
                    [31.75,-40.07],
                    [0.71,-12.12],
                    [51.31,-23.33],
                    [42.02,52.01],
                    [17.33,-76.67],
                    [31.43,-0.78],
                    [60.02,30.59],
                    [74.71,-5.18],
                    [13.14,-52.55],
                    [40.36,-42.0],
                    [8.97,-69.64],
                    [76.32,7.23],
                    [16.76,-60.47],
                    [59.08,18.4],
                    [68.17,6.89],
                    [1.69,15.68],
                    [30.5,-24.68],
                    [14.81,-56.5]]


# In[162]:


angles_410_embedded = np.load('4e10_embedded_angles.npy', mmap_mode='r')

angles_410_embedded= np.array(angles_410_embedded)

plot_angles(angles_410_embedded[:,0], angles_410_embedded[:,1], np.array(starting_angles),
           '#55A3FF', '#2A517F', '4e10_embedded')


# In[ ]:





# In[19]:


angle_gar_top_4e10 = np.load('4e10_spont_top_gar_angles.npy', mmap_mode='r')
angle_ims_top_4e10 = np.load('4e10_spont_top_ims_angles.npy', mmap_mode='r')
angle_gar_bot_4e10 = np.load('4e10_spont_bot_gar_angles.npy', mmap_mode='r')
angle_ims_bot_4e10 = np.load('4e10_spont_bot_ims_angles.npy', mmap_mode='r')


aggregate_angles_4e10_cg_spont= []
for i in angle_gar_top_4e10:
    aggregate_angles_4e10_cg_spont.append(i)
for i in angle_ims_top_4e10:
    aggregate_angles_4e10_cg_spont.append(i)
for i in angle_gar_bot_4e10:
    aggregate_angles_4e10_cg_spont.append(i)
for i in angle_ims_bot_4e10:
    aggregate_angles_4e10_cg_spont.append(i)
    
aggregate_angles_4e10_cg_spont=np.array(aggregate_angles_4e10_cg_spont)
    
plot_angles(aggregate_angles_4e10_cg_spont[:,0], aggregate_angles_4e10_cg_spont[:,1], 
            np.array([[150,150]]),
           '#55A3FF', '#2A517F', '4e10_spont_insert')


# In[25]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/4e10/medoid1/'
input_pdb = wd+'step5_input.pdb'
print(getAngle_4e10_AA(parsePDB(input_pdb)))

wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/4e10/medoid2/'
input_pdb = wd+'step5_input.pdb'
print(getAngle_4e10_AA(parsePDB(input_pdb)))

wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/4e10/medoid3/'
input_pdb = wd+'step5_input.pdb'
print(getAngle_4e10_AA(parsePDB(input_pdb)))


# In[137]:


# plt.plot(angles_410_ppm[:,0])
# plt.plot(angles_410_p15[:,0])
# plt.plot(angles_410_n15[:,0])
# plt.plot(angles_410_ppm_rep[:,0])

# plt.plot(angles_410_ppm[:,1])
# plt.plot(angles_410_p15[:,1])
# plt.plot(angles_410_n15[:,1])
# plt.plot(angles_410_ppm_rep[:,1])


# In[107]:


print(max(aggregate_angles_4e10_cg_spont[:,0]))
print(min(aggregate_angles_4e10_cg_spont[:,0]))


# In[52]:


# pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/4e10/medoid1.pdb'
# pdb_in = parsePDB(pdb)
# print(getAngle_4e10_cg_spont(pdb_in))

# pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/4e10/medoid1.pdb'
# pdb_in = parsePDB(pdb)
# print(getAngle_4e10_cg_spont_bottom(pdb_in))
# pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/4e10/medoid1/step5_input.pdb'
# pdb_in = parsePDB(pdb)
# print(getAngle_4e10_AA(pdb_in))


pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/4e10/medoid2_cg.pdb'
pdb_in = parsePDB(pdb)
print(getAngle_4e10_cg_spont(pdb_in))
pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/4e10/medoid2/step5_input.pdb'
pdb_in = parsePDB(pdb)
print(getAngle_4e10_AA(pdb_in))

pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/4e10/medoid3_cg.pdb'
pdb_in = parsePDB(pdb)
print(getAngle_4e10_cg_spont(pdb_in))
pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/4e10/medoid3/step5_input.pdb'
pdb_in = parsePDB(pdb)
print(getAngle_4e10_AA(pdb_in))


# In[9]:


#MEDOID CHECKS 3  
pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/4e10/medoid3_cg.pdb'
pdb_in = parsePDB(pdb)
print(getAngle_4e10_cg_spont(pdb_in))

pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/4e10/medoid3_aa_system/gromacs/step5_input.pdb'
pdb_in = parsePDB(pdb)
print(getAngle_4e10_AA(pdb_in))


# In[9]:


#MEDOID CHECKS 2 
pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/4e10/medoid3_cg.pdb'
pdb_in = parsePDB(pdb)
print(getAngle_4e10_cg_spont(pdb_in))

# pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/4e10/medoid2_aa_system/gromacs/step5_input.pdb'
pdb = '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts_v2/medoid3_aa_system/gromacs/step5_input.pdb'
pdb_in = parsePDB(pdb)
print(getAngle_4e10_AA(pdb_in))


# In[10]:


#MEDOID CHECKS 2 
pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/4e10/medoid2_cg.pdb'
pdb_in = parsePDB(pdb)
print(getAngle_4e10_cg_spont(pdb_in))

# pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/4e10/medoid2_aa_system/gromacs/step5_input.pdb'
pdb = '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts_v2/medoid2_4e10_aa_system/gromacs/step5_input.pdb'
pdb_in = parsePDB(pdb)
print(getAngle_4e10_AA(pdb_in))



# In[12]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/4e10/medoid3/'
input_pdb = wd+'analysis_input.pdb'
dcd_traj = wd+'traj_1000ns.dcd'
trajectory_angles_4e10_aa(input_pdb, dcd_traj, '4e10_medoid3_cg_revsion')


# In[16]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/4e10/medoid2/'
input_pdb = wd+'analysis_input.pdb'
dcd_traj = wd+'traj_1000ns.dcd'
trajectory_angles_4e10_aa(input_pdb, dcd_traj, '4e10_medoid2_cg_revsion')


# In[4]:


angle_4e10_medoid3 = np.load('4e10_medoid3_cg_revsion_angles.npy', mmap_mode='r')
angle_4e10_medoid2 = np.load('4e10_medoid2_cg_revsion_angles.npy', mmap_mode='r')
#plt.plot(angle_4e10_medoid3[:,1])

angles_410_ppm = np.load('4e10_ppm_angles.npy', mmap_mode='r')
angles_410_p15 = np.load('4e10_p15_angles.npy', mmap_mode='r')
angles_410_n15 = np.load('4e10_n15_angles.npy', mmap_mode='r')
angles_410_ppm_rep = np.load('4e10_ppm_rep_angles.npy', mmap_mode='r')
plt.rcParams['font.sans-serif']=  ['Arial'] 
plt.rcParams["figure.figsize"] = (12,4)
plt.plot(angles_410_ppm[33:,0], color='black', linewidth=0.1)
plt.plot(angles_410_p15[33:,0], color='grey', linewidth=0.1)
plt.plot(angles_410_n15[33:,0], color='black', linewidth=0.1)
plt.plot(angles_410_ppm_rep[33:,0], color='grey', linewidth=0.1)
plt.plot(angle_4e10_medoid3[0:,0], color='black', linewidth=0.5)
#plt.plot(angle_4e10_medoid2[:,0], color='blue')



plt.plot(angles_410_ppm[33:,1], color='pink', linewidth=0.1)
plt.plot(angles_410_p15[33:,1], color='pink', linewidth=0.1)
plt.plot(angles_410_n15[33:,1], color='pink', linewidth=0.1)
plt.plot(angles_410_ppm_rep[33:,1], color='pink', linewidth=0.1)
plt.plot(angle_4e10_medoid3[0:,1], color='red',  linewidth=0.5)

plt.yticks([ -90, 0, 90], fontsize=55)
plt.ylim(-100,100)
plt.xticks([0, 2500, 5000], [0,  0.5,  1],  fontsize=55)
plt.xlim([-1, 5031])
#plt.plot(angle_4e10_medoid2[:,1], color='blue')


# In[11]:


print(np.mean(angle_4e10_medoid3[2500:5002,0]))
print(np.std(angle_4e10_medoid3[2500:5002,0]))
print(np.mean(angle_4e10_medoid3[2500:5002,1]))


# In[56]:


angle_4e10_medoid2 = np.load('4e10_medoid2_cg_revsion_angles.npy', mmap_mode='r')
angle_4e10_medoid2 = np.load('4e10_medoid2_cg_revsion_angles.npy', mmap_mode='r')
#plt.plot(angle_4e10_medoid2[:,1])

angles_410_ppm = np.load('4e10_ppm_angles.npy', mmap_mode='r')
angles_410_p15 = np.load('4e10_p15_angles.npy', mmap_mode='r')
angles_410_n15 = np.load('4e10_n15_angles.npy', mmap_mode='r')
angles_410_ppm_rep = np.load('4e10_ppm_rep_angles.npy', mmap_mode='r')
plt.rcParams['font.sans-serif']=  ['Arial'] 
plt.rcParams["figure.figsize"] = (12,4)
plt.plot(angles_410_ppm[33:,0], color='black', linewidth=0.1)
plt.plot(angles_410_p15[33:,0], color='grey', linewidth=0.1)
plt.plot(angles_410_n15[33:,0], color='black', linewidth=0.1)
plt.plot(angles_410_ppm_rep[33:,0], color='grey', linewidth=0.1)
plt.plot(angle_4e10_medoid2[0:,0], color='black', linewidth=0.5)
#plt.plot(angle_4e10_medoid2[:,0], color='blue')



plt.plot(angles_410_ppm[33:,1], color='pink', linewidth=0.1)
plt.plot(angles_410_p15[33:,1], color='pink', linewidth=0.1)
plt.plot(angles_410_n15[33:,1], color='pink', linewidth=0.1)
plt.plot(angles_410_ppm_rep[33:,1], color='pink', linewidth=0.1)
plt.plot(angle_4e10_medoid2[0:,1], color='red',  linewidth=0.5)

plt.yticks([ -90, 0, 90], fontsize=55)
plt.ylim(-100,100)
plt.xticks([0, 2500, 5000], [0,  0.5,  1],  fontsize=55)
plt.xlim([-1, 5031])
#plt.plot(angle_4e10_medoid2[:,1], color='blue')


# In[9]:


#Update backmapping method -manual alignemnt; check angles of all atom system versus cg medoid

#MEDOID 1
pdb = '/Users/cmaillie/Dropbox (Scripps Research)/MPER_project/backmapping_CG2AA/4e10/4e10_medoid1/gromacs/step5_input.pdb'
pdb_in = parsePDB(pdb)
print(getAngle_4e10_AA(pdb_in))

#MEDOID 2
pdb = '/Users/cmaillie/Dropbox (Scripps Research)/MPER_project/backmapping_CG2AA/4e10/4e10_medoid2/gromacs/step5_input.pdb'
pdb_in = parsePDB(pdb)
print(getAngle_4e10_AA(pdb_in))

#MEDOID 3
pdb = '/Users/cmaillie/Dropbox (Scripps Research)/MPER_project/backmapping_CG2AA/4e10/4e10_medoid3/gromacs/step5_input.pdb'
pdb_in = parsePDB(pdb)
print(getAngle_4e10_AA(pdb_in))


# In[8]:


angles_410_ppm = np.load('4e10_ppm_angles.npy', mmap_mode='r')
angles_410_p15 = np.load('4e10_p15_angles.npy', mmap_mode='r')
angles_410_n15 = np.load('4e10_n15_angles.npy', mmap_mode='r')


angles = angles_410_ppm[33:,1]
angle_func = np.poly1d(angles)
#print("Polynomial function:\n", angle_func)
  
# calculating the derivative
angle_derivative = angle_func.deriv()
#print("Derivative, f(x)'=\n", angle_derivative)
angle_derived_time=[]
angle_derived_angle=[]
for i in range(len(angles)):
    angle_derived_time.append(angle_derivative(i))
    angle_derived_angle.append(angle_derivative(angles[i]))
    
    
plt.plot(angle_derived_time)
plt.show()
plt.plot(angle_derived_angle)
plt.show()


# In[34]:


cg = parsePDB('/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/4e10/medoid1_cg.pdb')
print(len(cg.select('name BB')))
reversion = parsePDB("/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/4e10/medoid1/step5_input.pdb")
print(len(reversion.select('name CA')))

superpose(reversion.select('name CA'),cg.select('name BB') )
writePDB('TEST_CG_MED_ALN.pdb', reversion )


# In[37]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/4e10/medoid1/'
input_pdb = wd+'500ns_all.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_4e10_aa(input_pdb, dcd_traj, '4e10_medoid1_traj')


# In[38]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/4e10/medoid2/'
input_pdb = wd+'500ns_all.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_4e10_aa(input_pdb, dcd_traj, '4e10_medoid2_traj')


# In[39]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/4e10/medoid3/'
input_pdb = wd+'500ns_all.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_4e10_aa(input_pdb, dcd_traj, '4e10_medoid3_traj')


# In[65]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_ppm/'
input_pdb = wd+'final_analysis_input.pdb'
input_pdb = parsePDB(input_pdb)
getAngle_4e10_AA(input_pdb)


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/4e10/medoid1/'
input_pdb = wd+'500ns_all.pdb'
input_pdb = parsePDB(input_pdb)
getAngle_4e10_AA(input_pdb)


# In[66]:


cg_vect1 = np.array([ 4.14760184, -5.08148843, 25.32270966])
cg_vect2 = np.array([  0.,          0.,        188.7312483])
print(angle_between(cg_vect1, cg_vect2))

cg_vect3 = np.array([ 12.46714511, -39.11678341,  -8.25168155])
cg_vect4 = np.array([  0.,           0.,         160.69991288])
print(angle_between(cg_vect3, cg_vect4))


aa_vect1 = np.array([ -8.48652624,  8.7478337,  23.68218693])
aa_vect2 = np.array([ 0.,           0.,         226.84919099])
print(angle_between(aa_vect1, aa_vect2))

aa_vect3 = np.array([ 25.33299919, 33.05286366, -1.13708216])
aa_vect4 = np.array([  0.,          0.,        195.9865363])
print(angle_between(aa_vect3, aa_vect4))


# In[5]:


#check backbone RMSD 
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/4e10/medoid1/'
input_pdb = wd+'500ns_all.pdb' 
dcd_traj = wd+'analysis.dcd'



dcd_traj  = dcd_traj 
input_pdb = parsePDB(input_pdb)
dcd = DCDFile(dcd_traj)
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()
rmds_list = [] 
xray_pdb = parsePDB('/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_ppm/fab_20.pdb')
xray_fab = xray_pdb.select('name CA') 

for i,frame in enumerate(dcd):
    pdb = frame.getAtoms()
    fab = pdb.select('name CA')
    superpose(fab,xray_fab)
    rmds_list.append(round((calcRMSD(fab, xray_fab)), 2))
    #angle_running.append(getAngle_4e10_cg_spont(pdb))


# In[27]:


plt.plot(rmds_list[100:600],'ro')
plt.plot(rmds_list[100:600],'r', linewidth=.5)


plt.xlabel('Time (ns)', fontsize=30)
#plt.ylabel('Force (kJ/mol*nm^2)', fontsize=22)
plt.ylabel('RMSD ($\AA$)', fontsize=30)
plt.yticks([0, 1, 2, 3, 4, 5], fontsize=22)
#plt.xlim(0, np.max(dist1_4e10) )
plt.xticks([ 0, 100, 200, 300, 400, 500], labels=[ '0', '20', '40', '60', '70', '80'], fontsize=24)
wd = '/Users/cmaillie/Dropbox (Scripps Research)/mper_ab_manuscript_y3/pngs/backmapping/'
plt.savefig(wd+'4e10_backmapping_xrayOverlay_CA_RMSD.png'  , dpi=300, bbox_inches='tight', transparent=True)


# In[24]:


wd = "/Users/cmaillie/Dropbox (Scripps Research)/eLife_revisions_experiments/10e8_hivLike/gromacs/"
input_pdb = wd+'final_analysis_input.pdb'
dcd_traj = wd+'final_analysis_traj.dcd'
trajectory_angles_4e10_aa(input_pdb, dcd_traj, '4e10_hivLike')


# In[25]:


angles_4e10_hivLike = np.load('4e10_hivLike_angles.npy', mmap_mode='r')


angles_4e10_hivLike_collected = []
for i in angles_4e10_hivLike:
    print(i)
    angles_4e10_hivLike_collected.append(i)

angles_4e10_hivLike_collected= np.array(angles_4e10_hivLike_collected)
plot_angles(angles_4e10_hivLike_collected[:,0], angles_4e10_hivLike_collected[:,1], np.array([0,0]), '#55A3FF', '#2A517F',  '4e10_hivLke')


# In[ ]:




