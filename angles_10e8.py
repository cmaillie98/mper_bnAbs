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


pdb = parsePDB('/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_ppm/final_analysis_input.pdb')
fab_aa = pdb.select('name CA')
cg_pdb = parsePDB('/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/10e8/analysis_input.pdb')
fab_cg = cg_pdb.select('name BB')


print(fab_aa[110].getResname(), fab_cg[110].getResname())
print(fab_aa[325].getResname(), fab_cg[324].getResname())
print(fab_aa[41].getResname(), fab_cg[41].getResname())
print(fab_aa[273].getResname(), fab_cg[272].getResname())
print(fab_aa[130].getResname(), fab_cg[130].getResname())
print(fab_aa[340].getResname(), fab_cg[339].getResname())


# In[4]:


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


# In[19]:


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


# In[6]:


#method to calculate angles of ab - all atom (4e10 & 10e8 )


def getAngle_10e8_AA(pdb):
    
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
    res94 = fab[325]
    res320 = fab[110]
    point_cdr_loops = calcCenter(res94+res320)
    res41 = fab[273]
    res256 = fab[41]
    point_center_fab = calcCenter(res41+res256)     
    point_xy_3 = np.array([point_cdr_loops[0], point_cdr_loops[1], 200]) 
    var_domain_vect = np.array(point_center_fab-point_cdr_loops)
    xy_norm_vect = np.array(point_xy_3-point_cdr_loops)
    angle_approach = round(angle_between(var_domain_vect, xy_norm_vect), 2)


    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[340].getCoords()
    point_h_edge = fab[120].getCoords()
    point_xz_3 = np.array([point_l_edge[0], point_l_edge[1], 200])
    short_ax_vect = np.array(point_l_edge-point_h_edge)
    xz_norm_vect = np.array(point_xz_3-point_l_edge)
    angle_rotation = round(angle_between(short_ax_vect, xz_norm_vect), 2)
    
    #angles are calcualted relative to psuedo normal vector  
    #angle of approach will be in range [0,90] after reflecting if angle is >90 
    #subtract reflected angle from 90 to find angle to mem plane
    if angle_approach>90:
        angle_approach=180-angle_approach 
    #angle of rotation will be in range [0, 180] 
    
    #report angles as angle to membrane by taking 90-angle 
    return [round(90-angle_approach,2), round(90-angle_rotation, 2)] 
    
    
    
    


# In[7]:


def trajectory_angles_10e8_aa(input_pdb, dcd_traj, output_name): 
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
        angle_running.append(getAngle_10e8_AA(pdb)) 
    #save to csv file 
    file_out = output_name + "_angles.npy"
    with open(file_out, 'wb') as f:
        np.save(f, angle_running)
    f.close()
    #np.savetxt(file_out, angle_running, delimiter =",",fmt ='% s')
    return print("finished " + file_out)


# In[8]:


#method to calculate angles of ab to membrane - coarse grain (4e10) 

def getAngle_10e8_cg_spont(pdb):
    
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
    pseudo_fab_cen = calcCenter(fab[42]+fab[264])
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
    res94 = fab[324]
    res320 = fab[110]
    point_cdr_loops = calcCenter(res94+res320)
    res41 = fab[272]
    res256 = fab[41]
    point_center_fab = calcCenter(res41+res256)     
    point_xy_3 = np.array([point_cdr_loops[0], point_cdr_loops[1], 200]) 
    var_domain_vect = np.array(point_center_fab-point_cdr_loops)
    xy_norm_vect = np.array(point_xy_3-point_cdr_loops)
    angle_approach = round(angle_between(var_domain_vect, xy_norm_vect), 2)
#     #angle of variable domain to membrane in xy space 

    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[340].getCoords()
    point_h_edge = fab[130].getCoords()
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


# In[9]:


def trajectory_angles_10e8_spont_cg(input_pdb, dcd_traj, output_name): 
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
        angle_running.append(getAngle_pgz1l_cg_spont(pdb)) 
    #save to csv file 
    file_out = output_name + "_angles.npy"
    with open(file_out, 'wb') as f:
        np.save(f, angle_running)
    f.close()
    #np.savetxt(file_out, angle_running, delimiter =",",fmt ='% s')
    return print("finished " + file_out) 


# In[10]:


#method to calculate angles of ab to membrane - coarse grain (4e10) 

def getAngle_10e8_cg_embedded(pdb):
    
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
    pseudo_fab_cen = calcCenter(fab[42]+fab[264])
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
    res94 = fab[324]
    res320 = fab[110]
    point_cdr_loops = calcCenter(res94+res320)
    res41 = fab[272]
    res256 = fab[41]
    point_center_fab = calcCenter(res41+res256)     
    point_xy_3 = np.array([point_cdr_loops[0], point_cdr_loops[1], 200]) 
    var_domain_vect = np.array(point_center_fab-point_cdr_loops)
    xy_norm_vect = np.array(point_xy_3-point_cdr_loops)
    angle_approach = round(angle_between(var_domain_vect, xy_norm_vect), 2)
#     #angle of variable domain to membrane in xy space 

    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[340].getCoords()
    point_h_edge = fab[130].getCoords()
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


# In[11]:


def trajectory_angles_10e8_embedded(input_pdb, dcd_traj, output_name): 
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
        angle_running.append(getAngle_10e8_cg_embedded(pdb)) 
    #save to csv file 
    file_out = output_name + "_angles.npy"
    with open(file_out, 'wb') as f:
        np.save(f, angle_running)
    f.close()
    #np.savetxt(file_out, angle_running, delimiter =",",fmt ='% s')
    return print("finished " + file_out)


# In[23]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_embedded/10e8_ab/'
input_pdb = wd+'analysis_input.pdb'
dcd_traj = wd+'membrane_contact_frames.dcd'
trajectory_angles_10e8_embedded(input_pdb, dcd_traj, '10e8_embedded')


# In[9]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_ppm/'
input_pdb = wd+'final_analysis_input.pdb'
dcd_traj = wd+'final_analysis_traj.dcd'
trajectory_angles_10e8_aa(input_pdb, dcd_traj, '10e8_ppm')
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_p15/'
input_pdb = wd+'final_analysis_input.pdb'
dcd_traj = wd+'final_analysis_traj.dcd'
trajectory_angles_10e8_aa(input_pdb, dcd_traj, '10e8_p15')
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_n15/'
input_pdb = wd+'final_analysis_input.pdb'
dcd_traj = wd+'final_analysis_traj.dcd'
trajectory_angles_10e8_aa(input_pdb, dcd_traj, '10e8_n15')
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_ppm_rep/'
input_pdb = wd+'final_analysis_input.pdb'
dcd_traj = wd+'final_analysis_traj.dcd'
trajectory_angles_10e8_aa(input_pdb, dcd_traj, '10e8_ppm_rep')


# In[ ]:





# In[8]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/10e8/medoid2/'
input_pdb = wd+'step5_input.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_10e8_aa(input_pdb, dcd_traj, 'med2_10e8_cg_backmap')


# In[9]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/10e8/medoid3/'
input_pdb = wd+'step5_input.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_10e8_aa(input_pdb, dcd_traj, 'med3_10e8_cg_backmap')


# In[10]:


aa_starting_angles = [] 
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_ppm/'
input_pdb = wd+'final_analysis_input.pdb'
print(getAngle_10e8_AA(parsePDB(input_pdb)))
aa_starting_angles.append(getAngle_10e8_AA(parsePDB(input_pdb)))
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_p15/'
input_pdb = wd+'final_analysis_input.pdb'
print(getAngle_10e8_AA(parsePDB(input_pdb)))
aa_starting_angles.append(getAngle_10e8_AA(parsePDB(input_pdb)))
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_n15/'
input_pdb = wd+'final_analysis_input.pdb'
print(getAngle_10e8_AA(parsePDB(input_pdb)))
aa_starting_angles.append(getAngle_10e8_AA(parsePDB(input_pdb)))
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_ppm_rep/'
input_pdb = wd+'final_analysis_input.pdb'
print(getAngle_10e8_AA(parsePDB(input_pdb)))
aa_starting_angles.append(getAngle_10e8_AA(parsePDB(input_pdb)))


# In[16]:


# wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/10e8/'
# input_pdb = wd+'analysis_input.pdb'
# dcd_traj = wd+'gar_top_contact.dcd'
# trajectory_angles_10e8_spont_cg(input_pdb, dcd_traj, '10e8_spont_top_gar')

# wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/10e8/'
# input_pdb = wd+'analysis_input.pdb'
# dcd_traj = wd+'ims_top_contact.dcd'
# trajectory_angles_10e8_spont_cg(input_pdb, dcd_traj, '10e8_spont_top_ims')

wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/10e8/'
input_pdb = wd+'analysis_input.pdb'
dcd_traj = wd+'gar_bot_contact.dcd'
trajectory_angles_10e8_spont_cg(input_pdb, dcd_traj, '10e8_spont_bot_gar')

# wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/10e8/'
# input_pdb = wd+'analysis_input.pdb'
# dcd_traj = wd+'ims_bot_contact.dcd'
# trajectory_angles_10e8_spont_cg(input_pdb, dcd_traj, '10e8_spont_bot_ims')


# In[16]:


angles_10e8_ppm = np.load('10e8_ppm_angles.npy', mmap_mode='r')
angles_10e8_p15 = np.load('10e8_p15_angles.npy', mmap_mode='r')
angles_10e8_n15 = np.load('10e8_n15_angles.npy', mmap_mode='r')
angles_10e8_ppm_rep = np.load('10e8_ppm_rep_angles.npy', mmap_mode='r')

aggregate_angles_10e8 = []
for i in angles_10e8_ppm:
    aggregate_angles_10e8.append(i)
for i in angles_10e8_p15:
    aggregate_angles_10e8.append(i)
for i in angles_10e8_n15:
    aggregate_angles_10e8.append(i)
for i in angles_10e8_ppm_rep:
    aggregate_angles_10e8.append(i)
aggregate_angles_10e8= np.array(aggregate_angles_10e8)
plot_angles(aggregate_angles_10e8[:,0], aggregate_angles_10e8[:,1], np.array(aa_starting_angles),
            '#C28EE8', '#460273',  '10e8_AA')


# In[28]:


angle_gar_top_10e8 = np.load('10e8_spont_top_gar_angles.npy', mmap_mode='r')
#angle_ims_top_10e8 = np.load('10e8_spont_top_ims_angles.npy', mmap_mode='r')
angle_gar_bot_10e8 = np.load('10e8_spont_bot_gar_angles.npy', mmap_mode='r')
#angle_ims_bot_10e8 = np.load('10e8_spont_bot_ims_angles.npy', mmap_mode='r')


aggregate_angles_10e8_cg_spont= []
for i in angle_gar_top_10e8:
    aggregate_angles_10e8_cg_spont.append(i)
# for i in angle_ims_top_10e8:
#     aggregate_angles_10e8_cg_spont.append(i)
for i in angle_gar_bot_10e8:
    aggregate_angles_10e8_cg_spont.append(i)
# for i in angle_ims_bot_10e8:
#     aggregate_angles_10e8_cg_spont.append(i)
    
aggregate_angles_10e8_cg_spont=np.array(aggregate_angles_10e8_cg_spont)
    
plot_angles(aggregate_angles_10e8_cg_spont[:,0], aggregate_angles_10e8_cg_spont[:,1], 
            np.array([[150,150]]),
            '#C28EE8', '#460273', '10e8_spont_insert')


# In[32]:


starting_angles =[[63.45, 27.92], [29.65, 68.6], [65.83, 9.46], [62.05, 4.89], [21.15, 6.19], [25.08, 16.51], [18.6, 1.57], [44.35, 22.35], [50.04, 2.98], [1.22, 4.31], [4.56, -64.76], [75.16, 24.58], [30.17, -1.44], [10.0, 29.86], [57.85, 45.51], [21.93, -60.3], [36.41, 2.98], [46.05, -19.72], [27.69, -14.3], [41.02, -51.18], [1.61, 45.64], [28.22, 20.43], [24.1, -68.06], [1.57, 2.94], [39.12, 13.52], [32.61, 50.34], [22.18, -45.71], [39.14, -66.3], [43.93, -60.15], [40.49, -9.1], [36.61, 34.13], [51.83, 6.11], [83.86, -11.71], [26.67, -35.54], [24.14, -5.02], [28.19, 21.92], [26.99, 44.24], [58.59, 19.53], [83.22, -4.48], [12.47, -1.19], [0.4, 10.67]]


# In[33]:


angles_410_embedded = np.load('10e8_embedded_angles.npy', mmap_mode='r')

angles_410_embedded= np.array(angles_410_embedded)
plot_angles(angles_410_embedded[:,0], angles_410_embedded[:,1], np.array(starting_angles),
           '#C28EE8', '#460273', '10e8_embedded')


# In[20]:


print(max(aggregate_angles_10e8[:,0]))
print(min(aggregate_angles_10e8[:,0]))
print(max(aggregate_angles_10e8[:,1]))
print(min(aggregate_angles_10e8[:,1]))


print(max(aggregate_angles_10e8_cg_spont[:,0]))
print(min(aggregate_angles_10e8_cg_spont[:,0]))
print(max(aggregate_angles_10e8_cg_spont[:,1]))
print(min(aggregate_angles_10e8_cg_spont[:,1]))


# In[5]:


fp = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/10e8/medoid1_cg.pdb'
pdb = parsePDB(fp)
print(getAngle_10e8_cg_spont(pdb))


# In[28]:


#method to calculate angles of ab - all atom (4e10 & 10e8 )


def getAngle_10e8_AAv2(pdb):
    
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
    
    fab = input_pdb.select(fab_selection_str)
    res94 = fab[324]
    res320 = fab[110]
    point_cdr_loops = calcCenter(res94+res320)
    res41 = fab[272]
    res256 = fab[41]
    point_center_fab = calcCenter(res41+res256)     
    point_xy_3 = np.array([point_cdr_loops[0], point_cdr_loops[1], 200]) 
    var_domain_vect = np.array(point_center_fab-point_cdr_loops)
    xy_norm_vect = np.array(point_xy_3-point_cdr_loops)
    angle_approach = round(angle_between(var_domain_vect, xy_norm_vect), 2)
#     #angle of variable domain to membrane in xy space 

    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[340].getCoords()
    point_h_edge = fab[130].getCoords()
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


# In[29]:


def trajectory_angles_10e8_aav2(input_pdb, dcd_traj, output_name): 
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
        angle_running.append(getAngle_10e8_AAv2(pdb)) 
    #save to csv file 
    file_out = output_name + "_angles.npy"
    with open(file_out, 'wb') as f:
        np.save(f, angle_running)
    f.close()
    #np.savetxt(file_out, angle_running, delimiter =",",fmt ='% s')
    return print("finished " + file_out)


# In[10]:



fp = '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/cg_reversions/medoid3_10e8_aa_system/gromacs/step5_input.pdb'
pdb = parsePDB(fp)
print(getAngle_10e8_AAv2(pdb))


# In[14]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/10e8/medoid1/'
input_pdb = wd+'analysis_input.pdb'
dcd_traj = wd+'traj_500ns.dcd'
trajectory_angles_10e8_aav2(input_pdb, dcd_traj, '10e8_medoid1_traj')


# In[16]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/10e8/medoid2/'
input_pdb = wd+'analysis_input.pdb'
dcd_traj = wd+'traj_500ns.dcd'
trajectory_angles_10e8_aav2(input_pdb, dcd_traj, '10e8_medoid2_traj')


# In[40]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/10e8/medoid3/'
input_pdb = wd+'analysis_input.pdb'
dcd_traj = wd+'traj_500ns.dcd'
trajectory_angles_10e8_aav2(input_pdb, dcd_traj, '10e8_medoid3_traj')


# In[42]:


angle_10e8_medoid3 = np.load('10e8_medoid3_traj_angles.npy', mmap_mode='r')
#plt.plot(angle_10e8_medoid3[:,1])
angles_10e8_ppm = np.load('10e8_ppm_angles.npy', mmap_mode='r')
angles_10e8_p15 = np.load('10e8_p15_angles.npy', mmap_mode='r')
angles_10e8_n15 = np.load('10e8_n15_angles.npy', mmap_mode='r')
angles_10e8_ppm_rep = np.load('10e8_ppm_rep_angles.npy', mmap_mode='r')

plt.plot(angles_10e8_ppm[33:2533,0], color='grey', linewidth=0.1)
plt.plot(angles_10e8_p15[33:2533,0], color='grey', linewidth=0.1)
plt.plot(angles_10e8_n15[33:2533,0], color='grey', linewidth=0.1)
plt.plot(angles_10e8_ppm_rep[33:2533,0], color='grey', linewidth=0.1)
plt.plot(angle_10e8_medoid3[0:2500,0], color='black', linewidth=0.5)
#plt.plot(angle_10e8_medoid2[:,0], color='blue')



plt.plot(angles_10e8_ppm[33:2533,1], color='pink', linewidth=0.1)
plt.plot(angles_10e8_p15[33:2533,1], color='pink', linewidth=0.1)
plt.plot(angles_10e8_n15[33:2533,1], color='pink', linewidth=0.1)
plt.plot(angles_10e8_ppm_rep[33:2533,1], color='pink', linewidth=0.1)
plt.plot(angle_10e8_medoid3[0:2500,1], color='red', linewidth=0.5)
plt.rcParams['font.sans-serif']=  ['Arial'] 
plt.rcParams["figure.figsize"] = (12,4)
plt.yticks([-45, 0, 45, 90])
plt.xticks([0, 1250, 2500]) 
plt.yticks([ -90, 0, 90], fontsize=55)
plt.ylim(-100,100)
plt.xticks([0, 1250, 2500], [0, 0.25, 0.5],  fontsize=55)
plt.xlim([-1, 2500])
#plt.plot(angle_10e8_medoid2[:,1], color='blue')


# In[38]:


angle_10e8_medoid1 = np.load('10e8_medoid1_traj_angles.npy', mmap_mode='r')
#plt.plot(angle_10e8_medoid3[:,1])
angles_10e8_ppm = np.load('10e8_ppm_angles.npy', mmap_mode='r')
angles_10e8_p15 = np.load('10e8_p15_angles.npy', mmap_mode='r')
angles_10e8_n15 = np.load('10e8_n15_angles.npy', mmap_mode='r')
angles_10e8_ppm_rep = np.load('10e8_ppm_rep_angles.npy', mmap_mode='r')

plt.plot(angles_10e8_ppm[33:2533,0], color='grey', linewidth=0.1)
plt.plot(angles_10e8_p15[33:2533,0], color='grey', linewidth=0.1)
plt.plot(angles_10e8_n15[33:2533,0], color='grey', linewidth=0.1)
plt.plot(angles_10e8_ppm_rep[33:2533,0], color='grey', linewidth=0.1)
plt.plot(angle_10e8_medoid1[0:2500,0], color='black', linewidth=0.5)
#plt.plot(angle_10e8_medoid2[:,0], color='blue')



plt.plot(angles_10e8_ppm[33:2533,1], color='pink', linewidth=0.1)
plt.plot(angles_10e8_p15[33:2533,1], color='pink', linewidth=0.1)
plt.plot(angles_10e8_n15[33:2533,1], color='pink', linewidth=0.1)
plt.plot(angles_10e8_ppm_rep[33:2533,1], color='pink', linewidth=0.1)
plt.plot(angle_10e8_medoid1[0:2500,1], color='red', linewidth=0.5)
plt.rcParams['font.sans-serif']=  ['Arial'] 
plt.rcParams["figure.figsize"] = (12,4)
plt.yticks([-45, 0, 45, 90])
plt.xticks([0, 1250, 2500]) 
plt.yticks([ -90, 0, 90], fontsize=55)
plt.ylim(-100,100)
plt.xticks([0, 1250, 2500], [0, 0.25, 0.5],  fontsize=55)
plt.xlim([-1, 2500])
#plt.plot(angle_10e8_medoid2[:,1], color='blue')


# In[39]:


angle_10e8_medoid2 = np.load('10e8_medoid2_traj_angles.npy', mmap_mode='r')
#plt.plot(angle_10e8_medoid3[:,1])


plt.plot(angles_10e8_ppm[33:2533,0], color='grey', linewidth=0.1)
plt.plot(angles_10e8_p15[33:2533,0], color='grey', linewidth=0.1)
plt.plot(angles_10e8_n15[33:2533,0], color='grey', linewidth=0.1)
plt.plot(angles_10e8_ppm_rep[33:2533,0], color='grey', linewidth=0.1)
plt.plot(angle_10e8_medoid2[0:2500,0], color='black', linewidth=0.5)
#plt.plot(angle_10e8_medoid2[:,0], color='blue')



plt.plot(angles_10e8_ppm[33:2533,1], color='pink', linewidth=0.1)
plt.plot(angles_10e8_p15[33:2533,1], color='pink', linewidth=0.1)
plt.plot(angles_10e8_n15[33:2533,1], color='pink', linewidth=0.1)
plt.plot(angles_10e8_ppm_rep[33:2533,1], color='pink', linewidth=0.1)
plt.plot(angle_10e8_medoid2[0:2500,1], color='red', linewidth=0.5)
plt.rcParams['font.sans-serif']=  ['Arial'] 
plt.rcParams["figure.figsize"] = (12,4)
plt.yticks([-45, 0, 45, 90])
plt.xticks([0, 1250, 2500]) 
plt.yticks([ -90, 0, 90], fontsize=55)
plt.ylim(-100,100)
plt.xticks([0, 1250, 2500], [0, 0.25, 0.5],  fontsize=55)
plt.xlim([-1, 2500])
#plt.plot(angle_10e8_medoid2[:,1], color='blue')


# In[14]:


pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/10e8/medoid1_cg.pdb'
pdb_in = parsePDB(pdb)
print(getAngle_10e8_cg_spont(pdb_in))

# pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/4e10/medoid2_aa_system/gromacs/step5_input.pdb'
pdb = '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/cg_reversions/medoid1_10e8_aa_system/gromacs/step5_input.pdb'
pdb_in = parsePDB(pdb)
print(getAngle_10e8_AAv2(pdb_in))


# In[16]:


pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/10e8/medoid2_cg.pdb'
pdb_in = parsePDB(pdb)
print(getAngle_10e8_cg_spont(pdb_in))

# pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/4e10/medoid2_aa_system/gromacs/step5_input.pdb'
pdb = '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/cg_reversions/medoid2_10e8_aa_system/gromacs/step5_input.pdb'
pdb_in = parsePDB(pdb)
print(getAngle_10e8_AAv2(pdb_in))


# In[6]:


pdb = '/Users/cmaillie/Dropbox (Scripps Research)/MPER_project/backmapping_CG2AA/10e8/10e8_med1/gromacs/step5_input.pdb'
pdb_in = parsePDB(pdb)
print(getAngle_10e8_AA(pdb_in))

pdb = '/Users/cmaillie/Dropbox (Scripps Research)/MPER_project/backmapping_CG2AA/10e8/10e8_med2/gromacs/step5_input.pdb'
pdb_in = parsePDB(pdb)
print(getAngle_10e8_AA(pdb_in))

pdb = '/Users/cmaillie/Dropbox (Scripps Research)/MPER_project/backmapping_CG2AA/10e8/10e8_med3/gromacs/step5_input.pdb'
pdb_in = parsePDB(pdb)
print(getAngle_10e8_AA(pdb_in))


# In[11]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/10e8/med1_v2/charmm-gui-6906372682/gromacs/'
input_pdb = wd+'400ns_all.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_10e8_aa(input_pdb, dcd_traj, 'pgzl1_medoid1_traj')


# In[12]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/10e8/medoid2/'
input_pdb = wd+'500ns_all.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_10e8_aa(input_pdb, dcd_traj, 'pgzl1_medoid2_traj')


# In[13]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/10e8/medoid3/'
input_pdb = wd+'500ns_all.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_10e8_aa(input_pdb, dcd_traj, 'pgzl1_medoid3_traj')


# In[12]:


wd = "/Users/cmaillie/Dropbox (Scripps Research)/eLife_revisions_experiments/10e8_hivLike/gromacs/"
input_pdb = wd+'final_analysis_input.pdb'
dcd_traj = wd+'final_analysis_traj.dcd'
trajectory_angles_10e8_aa(input_pdb, dcd_traj, '10e8_hivLike')


# In[20]:


angles_10e8_hivLike = np.load('10e8_hivLike_angles.npy', mmap_mode='r')


angles_10e8_hivLike_collected = []
for i in angles_10e8_hivLike:
    angles_10e8_hivLike_collected.append(i)

angles_10e8_hivLike_collected= np.array(angles_10e8_hivLike_collected)
plot_angles(angles_10e8_hivLike_collected[:,0], angles_10e8_hivLike_collected[:,1], np.array([0,0]), '#C28EE8', '#460273',  '10e8_hivLke')

