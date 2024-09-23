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


pdb = parsePDB('/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_ppm/final_analysis_input.pdb')
fab_aa = pdb.select('name CA')
cg_pdb = parsePDB('/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/pgzl1/analysis_input.pdb')
fab_cg = cg_pdb.select('name BB')


print(fab_aa[94].getResname(), fab_cg[317].getResname())
print(fab_aa[320].getResname(), fab_cg[106].getResname())
print(fab_aa[41].getResname(), fab_cg[264].getResname())
print(fab_aa[256].getResname(), fab_cg[42].getResname())
print(fab_aa[107].getResname(), fab_cg[330].getResname())
print(fab_aa[340].getResname(), fab_cg[126].getResname())


# In[3]:


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


# In[11]:


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


# In[5]:


#method to calculate angles of ab - all atom (4e10 & pgzl1 )


def getAngle_pgzl1_AA(pdb):
    
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


    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[107].getCoords()
    point_h_edge = fab[340].getCoords()
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
    
    
    
    


# In[6]:


def trajectory_angles_pgzl1_aa(input_pdb, dcd_traj, output_name): 
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
        angle_running.append(getAngle_pgzl1_AA(pdb)) 
    #save to csv file 
    file_out = output_name + "_angles.npy"
    with open(file_out, 'wb') as f:
        np.save(f, angle_running)
    f.close()
    #np.savetxt(file_out, angle_running, delimiter =",",fmt ='% s')
    return print("finished " + file_out)


# In[15]:


#method to calculate angles of ab to membrane - coarse grain (4e10) 

def getAngle_pgz1l_cg_spont(pdb):
    
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
    cg_cdrl3 = fab[317]
    c = fab[106]
    cg_point_cdr_loops = calcCenter(cg_cdrl3+cg_cdrl3)
    l_middle = fab[264]
    h_middle = fab[42]
    point_center_fab = calcCenter(l_middle+h_middle)     
    point_xy_3 = np.array([cg_point_cdr_loops[0], cg_point_cdr_loops[1], 200]) 
    var_domain_vect = np.array(point_center_fab-cg_point_cdr_loops)
    xy_norm_vect = np.array(point_xy_3-cg_point_cdr_loops)
    angle_approach = round(angle_between(var_domain_vect, xy_norm_vect), 2)
#     #angle of variable domain to membrane in xy space 

    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[330].getCoords()
    point_h_edge = fab[126].getCoords()
    point_xz_3 = np.array([point_l_edge[0], point_l_edge[1], 200])
    short_ax_vect = np.array(point_l_edge - point_h_edge )
    xz_norm_vect = np.array(point_xz_3-point_l_edge)
    #angle of short axis vector to 'plane normal' reference vector 
    angle_rotation = round(angle_between(short_ax_vect, xz_norm_vect), 2) 
    #print(angle_between(var_domain_vect, xy_norm_vect), angle_between(short_ax_vect, xy_norm_vect) )
    #select fab axes 
    
    #angle are calcualted relative to psuedo normal vector  
    #angle of approach will be in range [0,90] after reflecting if angle is >90 
    #subtract reflected angle from 90 to find angle to mem plane
    if angle_approach>90:
        angle_approach=180-angle_approach 
    #angle of rotation will be in range [0, 180] 
    
    #report angles as angle to membrane by taking 90-angle 
    return [round(90-angle_approach,2), round(90-angle_rotation, 2)] 


# In[7]:


def trajectory_angles_pgzl1_spont_cg(input_pdb, dcd_traj, output_name): 
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


# In[8]:


#method to calculate angles of ab to membrane - coarse grain (4e10) 

def getAngle_pgz1l_cg_embedded(pdb):
    
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
    res94 = fab[317]
    res320 = fab[106]
    point_cdr_loops = calcCenter(res94+res320)
    res41 = fab[264]
    res256 = fab[42]
    point_center_fab = calcCenter(res41+res256)     
    point_xy_3 = np.array([point_cdr_loops[0], point_cdr_loops[1], 200]) 
    var_domain_vect = np.array(point_center_fab-point_cdr_loops)
    xy_norm_vect = np.array(point_xy_3-point_cdr_loops)
    angle_approach = round(angle_between(var_domain_vect, xy_norm_vect), 2)
#     #angle of variable domain to membrane in xy space 

    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[330].getCoords()
    point_h_edge = fab[126].getCoords()
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


def trajectory_angles_pgzl1_embedded(input_pdb, dcd_traj, output_name): 
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
        angle_running.append(getAngle_pgz1l_cg_embedded(pdb)) 
    #save to csv file 
    file_out = output_name + "_angles.npy"
    with open(file_out, 'wb') as f:
        np.save(f, angle_running)
    f.close()
    #np.savetxt(file_out, angle_running, delimiter =",",fmt ='% s')
    return print("finished " + file_out) 


# In[34]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_embedded/pgzl1_ab/'
input_pdb = wd+'analysis_input.pdb'
dcd_traj = wd+'membrane_contact_frames.dcd'
trajectory_angles_pgzl1_embedded(input_pdb, dcd_traj, 'pgzl1_embedded')


# In[7]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_ppm/'
input_pdb = wd+'final_analysis_input.pdb'
dcd_traj = wd+'final_analysis_traj.dcd'
trajectory_angles_pgzl1_aa(input_pdb, dcd_traj, 'pgzl1_ppm')
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_p15/'
input_pdb = wd+'final_analysis_input.pdb'
dcd_traj = wd+'final_analysis_traj.dcd'
trajectory_angles_pgzl1_aa(input_pdb, dcd_traj, 'pgzl1_p15')
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_n15/'
input_pdb = wd+'final_analysis_input.pdb'
dcd_traj = wd+'final_analysis_traj.dcd'
trajectory_angles_pgzl1_aa(input_pdb, dcd_traj, 'pgzl1_n15')
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_ppm_rep/'
input_pdb = wd+'final_analysis_input.pdb'
dcd_traj = wd+'final_analysis_traj.dcd'
trajectory_angles_pgzl1_aa(input_pdb, dcd_traj, 'pgzl1_ppm_rep')


# In[15]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/pgzl1/'
input_pdb = wd+'analysis_input.pdb'
dcd_traj = wd+'gar_top_contact.dcd'
trajectory_angles_pgzl1_spont_cg(input_pdb, dcd_traj, 'pgzl1_spont_top_gar')

wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/pgzl1/'
input_pdb = wd+'analysis_input.pdb'
dcd_traj = wd+'ims_top_contact.dcd'
trajectory_angles_pgzl1_spont_cg(input_pdb, dcd_traj, 'pgzl1_spont_top_ims')

wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/pgzl1/'
input_pdb = wd+'analysis_input.pdb'
dcd_traj = wd+'gar_bot_contact.dcd'
trajectory_angles_pgzl1_spont_cg(input_pdb, dcd_traj, 'pgzl1_spont_bot_gar')

wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/pgzl1/'
input_pdb = wd+'analysis_input.pdb'
dcd_traj = wd+'ims_bot_contact.dcd'
trajectory_angles_pgzl1_spont_cg(input_pdb, dcd_traj, 'pgzl1_spont_bot_ims')


# In[8]:


aa_starting_angles = [] 
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_ppm/'
input_pdb = wd+'final_analysis_input.pdb'
print(getAngle_pgzl1_AA(parsePDB(input_pdb)))
aa_starting_angles.append(getAngle_pgzl1_AA(parsePDB(input_pdb)))
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_p15/'
input_pdb = wd+'final_analysis_input.pdb'
print(getAngle_pgzl1_AA(parsePDB(input_pdb)))
aa_starting_angles.append(getAngle_pgzl1_AA(parsePDB(input_pdb)))
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_n15/'
input_pdb = wd+'final_analysis_input.pdb'
print(getAngle_pgzl1_AA(parsePDB(input_pdb)))
aa_starting_angles.append(getAngle_pgzl1_AA(parsePDB(input_pdb)))
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_ppm_rep/'
input_pdb = wd+'final_analysis_input.pdb'
print(getAngle_pgzl1_AA(parsePDB(input_pdb)))
aa_starting_angles.append(getAngle_pgzl1_AA(parsePDB(input_pdb)))


# In[17]:


angles_pgzl1_ppm = np.load('pgzl1_ppm_angles.npy', mmap_mode='r')
angles_pgzl1_p15 = np.load('pgzl1_p15_angles.npy', mmap_mode='r')
angles_pgzl1_n15 = np.load('pgzl1_n15_angles.npy', mmap_mode='r')
angles_pgzl1_ppm_rep = np.load('pgzl1_ppm_rep_angles.npy', mmap_mode='r')

aggregate_angles_pgzl1 = []
for i in angles_pgzl1_ppm:
    aggregate_angles_pgzl1.append(i)
for i in angles_pgzl1_p15:
    aggregate_angles_pgzl1.append(i)
for i in angles_pgzl1_n15:
    aggregate_angles_pgzl1.append(i)
for i in angles_pgzl1_ppm_rep:
    aggregate_angles_pgzl1.append(i)
aggregate_angles_pgzl1= np.array(aggregate_angles_pgzl1)
plot_angles(aggregate_angles_pgzl1[:,0], aggregate_angles_pgzl1[:,1], np.array(aa_starting_angles),
           '#F79A99', '#A33634',  'pgzl1_AA')


# In[38]:


angle_gar_top_pgzl1 = np.load('pgzl1_spont_top_gar_angles.npy', mmap_mode='r')
angle_ims_top_pgzl1 = np.load('pgzl1_spont_top_ims_angles.npy', mmap_mode='r')
angle_gar_bot_pgzl1 = np.load('pgzl1_spont_bot_gar_angles.npy', mmap_mode='r')
angle_ims_bot_pgzl1 = np.load('pgzl1_spont_bot_ims_angles.npy', mmap_mode='r')


aggregate_angles_pgzl1_cg_spont= []
for i in angle_gar_top_pgzl1:
    aggregate_angles_pgzl1_cg_spont.append(i)
for i in angle_ims_top_pgzl1:
    aggregate_angles_pgzl1_cg_spont.append(i)
for i in angle_gar_bot_pgzl1:
    aggregate_angles_pgzl1_cg_spont.append(i)
for i in angle_ims_bot_pgzl1:
    aggregate_angles_pgzl1_cg_spont.append(i)
    
aggregate_angles_pgzl1_cg_spont=np.array(aggregate_angles_pgzl1_cg_spont)
    
plot_angles(aggregate_angles_pgzl1_cg_spont[:,0], aggregate_angles_pgzl1_cg_spont[:,1], 
            np.array([[150,150]]),
           '#F79A99', '#A33634', 'pgzl1_spont_insert')


# In[41]:


starting_angles =[[49.55, 27.04], [10.28, 60.81], [9.53, -23.85], [2.61, -25.82], [52.24, -18.09], [67.63, -13.09], [71.63, -17.17], [28.87, 28.93], [24.98, 10.75], [26.52, -26.1], [23.87, -65.83], [1.44, 9.06], [1.94, -23.67], [50.01, 23.72], [14.93, 52.01], [45.65, -40.3], [35.67, 20.91], [61.48, -40.93], [6.76, -30.25], [9.73, -62.82], [28.99, 13.35], [51.87, 19.63], [2.99, -57.01], [22.48, -33.37], [5.33, -4.44], [2.72, 26.85], [1.46, -29.96], [1.13, -74.72], [5.67, -65.82], [39.45, -29.24], [66.06, 11.02], [25.18, -15.55], [27.23, -18.04], [11.83, -8.22], [18.84, -36.82], [61.42, -9.77], [33.49, 25.67], [17.04, 20.55], [31.59, -36.68], [25.58, -36.67], [64.01, -8.86]]


# In[42]:


angles_410_embedded = np.load('pgzl1_embedded_angles.npy', mmap_mode='r')

angles_410_embedded= np.array(angles_410_embedded)
plot_angles(angles_410_embedded[:,0], angles_410_embedded[:,1], np.array(starting_angles),
           '#F79A99', '#A33634', 'pgzl1_embedded')


# In[20]:


print(max(aggregate_angles_pgzl1[:,0]))
print(min(aggregate_angles_pgzl1[:,0]))
print(max(aggregate_angles_pgzl1[:,1]))
print(min(aggregate_angles_pgzl1[:,1]))


# In[21]:


print(max(aggregate_angles_pgzl1_cg_spont[:,0]))
print(min(aggregate_angles_pgzl1_cg_spont[:,0]))
print(max(aggregate_angles_pgzl1_cg_spont[:,1]))
print(min(aggregate_angles_pgzl1_cg_spont[:,1]))


# In[7]:


#method to calculate angles of ab - all atom (4e10 & pgzl1 )


def getAngle_pgzl1_AAv2(pdb):
   
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
    fab = input_pdb.select(fab_selection_str)
    aa_cdrl3 = fab[94]
    aa_cdrh3 = fab[320]
    point_cdr_loops = calcCenter(aa_cdrl3+aa_cdrh3)
    l_middle = fab[41]
    h_middle = fab[256]
    point_center_fab = calcCenter(l_middle+h_middle)     
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
    wtr1 = AtomGroup('Water') 
    coords = np.array([point_cdr_loops, 
                       point_center_fab,
                       point_xy_3, 
                       point_l_edge, 
                       point_h_edge,
                       point_xz_3],
                       dtype=float)
    wtr1.setCoords(coords) 
    wtr1.setNames(['ZN', 'ZN', 'ZN', 'ZN', 'ZN', 'ZN' ])
    wtr1.setResnums([1, 2, 3, 4, 5, 6])
    wtr1.setResnames(['ZN2', 'ZN2', 'ZN2', 'ZN2', 'ZN2', 'ZN2'])
    #fp='/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont'
    #writePDB(fp+'/AAAxisMarkers.pdb', wtr1)
    #writePDB(fp+'/AAPdbAligned.pdb', input_pdb) 
    #report angles as angle to membrane by taking 90-angle 
    return [round(90-angle_approach,2), round(90-angle_rotation, 2)] 
    
    
    
    


# In[8]:


def trajectory_angles_pgzl1_aav2(input_pdb, dcd_traj, output_name): 
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
        angle_running.append(getAngle_pgzl1_AAv2(pdb)) 
    #save to csv file 
    file_out = output_name + "_angles.npy"
    with open(file_out, 'wb') as f:
        np.save(f, angle_running)
    f.close()
    #np.savetxt(file_out, angle_running, delimiter =",",fmt ='% s')
    return print("finished " + file_out)


# In[71]:


fp = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/pgzl1/medoid1_cg.pdb'
#pdb = parsePDB(fp)
print(getAngle_pgz1l_cg_spont(fp))

fp = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/pgzl1/medoid1_aa_system/gromacs/step5_input.pdb'
pdb = parsePDB(fp)
print(getAngle_pgzl1_AAv2(pdb))


# In[21]:


pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/pgzl1/medoid2_cg.pdb'
pdb_in = parsePDB(pdb)
print(getAngle_pgz1l_cg_spont(pdb_in))

# pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/4e10/medoid2_aa_system/gromacs/step5_input.pdb'
pdb = '/Users/cmaillie/Dropbox (Scripps Research)/manuscript/cg_reversions/medoid2_PGZL1_aa_system/gromacs/step5_input.pdb'
pdb_in = parsePDB(pdb)
print(getAngle_pgzl1_AAv2(pdb_in))


# In[23]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/pgzl1/medoid1/'
input_pdb = wd+'analysis_input.pdb'
dcd_traj = wd+'traj_500ns.dcd'
trajectory_angles_pgzl1_aa(input_pdb, dcd_traj, 'pgzl1_medoid1_traj')


# In[31]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/pgzl1/medoid2/'
input_pdb = wd+'analysis_input.pdb'
dcd_traj = wd+'traj_1000ns.dcd'
trajectory_angles_pgzl1_aav2(input_pdb, dcd_traj, 'pgzl1_medoid2_traj_1us')


# In[3]:


angle_pgzl1_medoid1 = np.load('pgzl1_medoid1_traj_angles.npy', mmap_mode='r')
#plt.plot(angle_pgzl1_medoid3[:,1])
angles_pgzl1_ppm = np.load('pgzl1_ppm_angles.npy', mmap_mode='r')
angles_pgzl1_p15 = np.load('pgzl1_p15_angles.npy', mmap_mode='r')
angles_pgzl1_n15 = np.load('pgzl1_n15_angles.npy', mmap_mode='r')
angles_pgzl1_ppm_rep = np.load('pgzl1_ppm_rep_angles.npy', mmap_mode='r')



plt.plot(angles_pgzl1_ppm[33:2533,0], color='grey', linewidth=0.1)
plt.plot(angles_pgzl1_p15[33:2533,0], color='grey', linewidth=0.1)
plt.plot(angles_pgzl1_n15[33:2533,0], color='grey', linewidth=0.1)
plt.plot(angles_pgzl1_ppm_rep[33:2533,0], color='grey', linewidth=0.1)
plt.plot(angle_pgzl1_medoid1[0:2500,0], color='black', linewidth=0.5)
#plt.plot(angle_pgzl1_medoid2[:,0], color='blue')



plt.plot(angles_pgzl1_ppm[33:2533,1], color='pink', linewidth=0.1)
plt.plot(angles_pgzl1_p15[33:2533,1], color='pink', linewidth=0.1)
plt.plot(angles_pgzl1_n15[33:2533,1], color='pink', linewidth=0.1)
plt.plot(angles_pgzl1_ppm_rep[33:2533,1], color='pink', linewidth=0.1)
plt.plot(angle_pgzl1_medoid1[0:2500,1], color='red', linewidth=0.5)

plt.yticks([ -90, 0, 90], fontsize=55)
plt.ylim(-100,100)
plt.xticks([0, 1250, 2500], [0, 0.25, 0.5],  fontsize=55)
plt.xlim([-1, 2500])


# In[ ]:





# In[4]:


angle_pgzl1_medoid2 = np.load('pgzl1_medoid2_traj_1us_angles.npy', mmap_mode='r')
#plt.plot(angle_pgzl1_medoid3[:,1])
angles_pgzl1_ppm = np.load('pgzl1_ppm_angles.npy', mmap_mode='r')
angles_pgzl1_p15 = np.load('pgzl1_p15_angles.npy', mmap_mode='r')
angles_pgzl1_n15 = np.load('pgzl1_n15_angles.npy', mmap_mode='r')
angles_pgzl1_ppm_rep = np.load('pgzl1_ppm_rep_angles.npy', mmap_mode='r')





plt.plot(angles_pgzl1_ppm[33:,0], color='grey', linewidth=0.1)
plt.plot(angles_pgzl1_p15[33:,0], color='grey', linewidth=0.1)
plt.plot(angles_pgzl1_n15[33:,0], color='grey', linewidth=0.1)
plt.plot(angles_pgzl1_ppm_rep[33:,0], color='grey', linewidth=0.1)
plt.plot(angle_pgzl1_medoid2[0:5000,0], color='black', linewidth=0.5)
#plt.plot(angle_pgzl1_medoid2[:,0], color='blue')



plt.plot(angles_pgzl1_ppm[33:,1], color='pink', linewidth=0.1)
plt.plot(angles_pgzl1_p15[33:,1], color='pink', linewidth=0.1)
plt.plot(angles_pgzl1_n15[33:,1], color='pink', linewidth=0.1)
plt.plot(angles_pgzl1_ppm_rep[33:,1], color='pink', linewidth=0.1)
plt.plot(angle_pgzl1_medoid2[0:5000,1], color='red', linewidth=0.5)
plt.rcParams['font.sans-serif']=  ['Arial'] 
plt.rcParams["figure.figsize"] = (12,4)
plt.yticks([ -90, 0, 90], fontsize=55)
plt.ylim(-100,100)
plt.xticks([0, 2500, 5000], [0,  0.5,  1],  fontsize=55)
plt.xlim([-1, 5031])


# In[17]:


#method to calculate angles of ab to membrane - coarse grain (4e10) 

def getAngle_pgz1l_cg_spont(pdb):
    
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
    cg_cdrl3 = fab[317]
    cg_cdrh3 = fab[106]
    cg_point_cdr_loops = calcCenter(cg_cdrl3+cg_cdrh3)
    l_middle = fab[264]
    h_middle = fab[42]
    point_center_fab = calcCenter(l_middle+h_middle)     
    point_xy_3 = np.array([cg_point_cdr_loops[0], cg_point_cdr_loops[1], 200]) 
    var_domain_vect = np.array(point_center_fab-cg_point_cdr_loops)
    xy_norm_vect = np.array(point_xy_3-cg_point_cdr_loops)
    angle_approach = round(angle_between(var_domain_vect, xy_norm_vect), 2)
#     #angle of variable domain to membrane in xy space 

    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[330].getCoords()
    point_h_edge = fab[126].getCoords()
    point_xz_3 = np.array([point_l_edge[0], point_l_edge[1], 200])
    short_ax_vect = np.array(point_l_edge - point_h_edge )
    xz_norm_vect = np.array(point_xz_3-point_l_edge)
    #angle of short axis vector to 'plane normal' reference vector 
    angle_rotation = round(angle_between(short_ax_vect, xz_norm_vect), 2) 
    #print(angle_between(var_domain_vect, xy_norm_vect), angle_between(short_ax_vect, xy_norm_vect) )
    #select fab axes 
    
    #angle are calcualted relative to psuedo normal vector  
    #angle of approach will be in range [0,90] after reflecting if angle is >90 
    #subtract reflected angle from 90 to find angle to mem plane
    if angle_approach>90:
        angle_approach=180-angle_approach 
    #angle of rotation will be in range [0, 180] 
    
    #report angles as angle to membrane by taking 90-angle 
    return [round(90-angle_approach,2), round(90-angle_rotation, 2)] 


# In[18]:


#print angles for input CG medoid frame 
pdb = '/Users/cmaillie/Dropbox (Scripps Research)/MPER_project/backmapping_CG2AA/pgzl1/medoid1_cg.pdb'
pdb_in = parsePDB(pdb)
print("CG MEDOID 1: ", getAngle_pgz1l_cg_spont(pdb_in))

pdb = '/Users/cmaillie/Dropbox (Scripps Research)/MPER_project/backmapping_CG2AA/pgzl1/medoid2_cg.pdb'
pdb_in = parsePDB(pdb)
print("CG MEDOID 2: ", getAngle_pgz1l_cg_spont(pdb_in))

pdb = '/Users/cmaillie/Dropbox (Scripps Research)/MPER_project/backmapping_CG2AA/pgzl1/medoid3_cg.pdb'
pdb_in = parsePDB(pdb)
print("CG MEDOID 3: ", getAngle_pgz1l_cg_spont(pdb_in))


# In[ ]:


#check backmapped systems have same angles as CG input frames 
pdb = '/Users/cmaillie/Dropbox (Scripps Research)/MPER_project/backmapping_CG2AA/pgzl1/medoid1_cg.pdb'
pdb_in = parsePDB(pdb)
print("CG MEDOID 1: ", getAngle_pgz1l_cg_spont(pdb_in))

pdb = '/Users/cmaillie/Dropbox (Scripps Research)/MPER_project/backmapping_CG2AA/pgzl1/medoid2_cg.pdb'
pdb_in = parsePDB(pdb)
print("CG MEDOID 2: ", getAngle_pgz1l_cg_spont(pdb_in))

pdb = '/Users/cmaillie/Dropbox (Scripps Research)/MPER_project/backmapping_CG2AA/pgzl1/medoid3_cg.pdb'
pdb_in = parsePDB(pdb)
print("CG MEDOID 3: ", getAngle_pgz1l_cg_spont(pdb_in))


# In[60]:


# pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/4e10/medoid2_aa_system/gromacs/step5_input.pdb'
pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/pgzl1/origPDB_aln_backmapping/medoid3_stateC/pgzl1_med3_stateC_backmapped_sys/gromacs/step5_input.pdb'
pdb_in = parsePDB(pdb)
#print("Backmapped MEDOID 3: ",getAngle_pgzl1_AA(pdb_in))
print("Backmapped MEDOID 3: ",getAngle_pgzl1_AAv2(pdb_in))


# In[67]:




pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/pgzl1/medoid2/step5_input.pdb'
pdb_in = parsePDB(pdb)
print("Backmapped MEDOID 3: ",getAngle_pgzl1_AA(pdb_in))
print("Backmapped MEDOID 2: ",getAngle_pgzl1_AAv2(pdb_in))


# In[48]:


pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/pgzl1/medoid3_cg.pdb'
pdb_in = parsePDB(pdb)
print("CG MEDOID 3: ", getAngle_pgz1l_cg_spontv2(pdb))


# In[22]:


cg_pdb_path = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/pgzl1/medoid3_cg.pdb'
aa_pdb_path = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/pgzl1/pgzl1_fab_aa.pdb'

cg_pdb = parsePDB(cg_pdb_path) 
cg_fab = cg_pdb.select('name BB')
aa_pdb = parsePDB(aa_pdb_path) 
aa_fab = aa_pdb.select('name CA')
# for i in range(len(cg_fab)):
#     print(i, " CG: ", cg_fab[i].getResname(), i, " AA: ", aa_fab[i].getResname() )
    
print(" CG: ", cg_fab[317].getResname(),  " AA: ", aa_fab[94].getResname() )
print(" CG: ", cg_fab[106].getResname(), " AA: ", aa_fab[320].getResname() )

print(" CG: ", cg_fab[318].getResname(),  " AA: ", aa_fab[95].getResname() )
print(" CG: ", cg_fab[107].getResname(), " AA: ", aa_fab[321].getResname() )

print(" CG: ", cg_fab[264].getResname(), " AA: ", aa_fab[41].getResname() )
print(" CG: ", cg_fab[42].getResname(),  " AA: ", aa_fab[256].getResname() )



print(" CG: ", cg_fab[265].getResname(), " AA: ", aa_fab[42].getResname() )
print(" CG: ", cg_fab[43].getResname(),  " AA: ", aa_fab[257].getResname() )
print(" CG: ", cg_fab[330].getResname(),  " AA: ", aa_fab[107].getResname() )
print(" CG: ", cg_fab[126].getResname(),  " AA: ", aa_fab[340].getResname() )


# In[25]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/pgzl1/medoid1/'
input_pdb = wd+'500ns_all.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_pgzl1_aa(input_pdb, dcd_traj, 'pgzl1_medoid1_traj')


# In[28]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/pgzl1/medoid2/'
input_pdb = wd+'500ns_all.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_pgzl1_aa(input_pdb, dcd_traj, 'pgzl2_medoid1_traj')


# In[29]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/pgzl1/medoid3/'
input_pdb = wd+'500ns_all.pdb'
dcd_traj = wd+'analysis.dcd'
trajectory_angles_pgzl1_aa(input_pdb, dcd_traj, 'pgzl1_medoid3_traj')


# In[17]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/pgzl1/medoid1/'
input_pdb = wd+'step5_input.pdb'
print(getAngle_pgzl1_AA(parsePDB(input_pdb)))
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/pgzl1/medoid2/'
input_pdb = wd+'step5_input.pdb'
print(getAngle_pgzl1_AA(parsePDB(input_pdb)))

wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_reversion/pgzl1/medoid3/'
input_pdb = wd+'step5_input.pdb'
print(getAngle_pgzl1_AA(parsePDB(input_pdb)))
print(getAngle_pgzl1_AAv2(parsePDB(input_pdb)))


# In[9]:


wd = "/Users/cmaillie/Dropbox (Scripps Research)/eLife_revisions_experiments/pgzl1_hivLike/gromacs/"
input_pdb = wd+'final_analysis_input.pdb'
dcd_traj = wd+'final_analysis_traj.dcd'
trajectory_angles_pgzl1_aa(input_pdb, dcd_traj, 'pgzl1_hivLike')


# In[12]:


angles_pgzl1_hivLike = np.load('pgzl1_hivLike_angles.npy', mmap_mode='r')


angles_pgzl1_hivLike_collected = []
for i in angles_pgzl1_hivLike:
    #print(i)
    angles_pgzl1_hivLike_collected.append(i)

angles_pgzl1_hivLike_collected= np.array(angles_pgzl1_hivLike_collected)

plot_angles(angles_pgzl1_hivLike_collected[:,0], 
            angles_pgzl1_hivLike_collected[:,1], 
            np.array([0,0]), 
            '#F79A99', 
            '#A33634',  
            '4e10_hivLke')

