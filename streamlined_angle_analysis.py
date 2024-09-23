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

UnNatAA={}
UnNatAA["ALA"] = 'A'; UnNatAA["CYS"] = 'C'; UnNatAA["ASP"] = 'D'; UnNatAA["GLU"] = 'E'; UnNatAA["PHE"] = 'F';
UnNatAA["GLY"] = 'G'; UnNatAA["HIS"] = 'H'; UnNatAA["ILE"] = 'I'; UnNatAA["LYS"] = 'K';
UnNatAA["LEU"] = 'L'; UnNatAA["MET"] = 'M'; UnNatAA["ASN"] = 'N'; UnNatAA["PRO"] = 'P'; UnNatAA["GLN"] = 'Q';
UnNatAA["ARG"] = 'R'; UnNatAA["SER"] = 'S'; UnNatAA["THR"] = 'T'; UnNatAA["VAL"] = 'V'; UnNatAA["TRP"] = 'W'; UnNatAA["TYR"] = 'Y';
UnNatAA['ABA'] = 'A'; UnNatAA['CSO'] = 'C'; UnNatAA['CSD'] = 'C'; UnNatAA['CME'] = 'C';
UnNatAA['OCS'] = 'C'; UnNatAA["HSD"] = 'H'; UnNatAA['KCX'] = 'K'; UnNatAA['LLP'] = 'K';
UnNatAA['MLY'] = 'K'; UnNatAA['M3L'] = 'K'; UnNatAA['MSE'] = 'M'; UnNatAA['PCA'] = 'P'; UnNatAA['HYP'] = 'P';
UnNatAA['SEP'] = 'S'; UnNatAA['TPO'] = 'T'; UnNatAA['PTR'] = 'Y'

#NOTE: based on numbering for input pdbs; separate methods were created. 
#4e10 uses same numbering in AA & CG strucutres
#PGZL1 uses swapped chain numbering in AA & CG (4e10 AAA is same as PGZL1 AA so same method)
#10e8 has slighlty offset numbering b/w AA& CG 
#all methods have been validated by checking ref points across all pdbs 10.12.21


# In[2]:


file = open("10e8_embedded_cg_angles.csv", "r")
csv_reader = csv.reader(file) 
fts_cg_embedded_init = []
counter = 0 
for row in csv_reader:
    if counter<5033:
        fts_cg_embedded_init.append(row)
        counter = counter+1
fts_cg_embedded =  string_list_to_float(fts_cg_embedded_init, 'fts_10e8_ppm_1us_AA_float' )
print(len(fts_cg_embedded))


# In[9]:


file = open("4e10_embedded_cg_angles.csv", "r")
csv_reader = csv.reader(file) 
fts_cg_embedded_init = []
counter = 0 
for row in csv_reader:
    if counter<5033:
        fts_cg_embedded_init.append(row)
        counter = counter+1
fts_cg_embedded =  string_list_to_float(fts_cg_embedded_init, 'fts_10e8_ppm_1us_AA_float' )
print(len(fts_cg_embedded))


# In[10]:


file = open("pgzl1_embedded_cg_angles.csv", "r")
csv_reader = csv.reader(file) 
fts_cg_embedded_init = []
counter = 0 
for row in csv_reader:
    if counter<5033:
        fts_cg_embedded_init.append(row)
        counter = counter+1
fts_cg_embedded =  string_list_to_float(fts_cg_embedded_init, 'fts_10e8_ppm_1us_AA_float' )
print(len(fts_cg_embedded))


# In[ ]:


file = open("10e8_embedded_cg_angles.csv", "r")
csv_reader = csv.reader(file) 
fts_cg_embedded_init = []
counter = 0 
for row in csv_reader:
    if counter<5033:
        fts_cg_embedded_init.append(row)
        counter = counter+1
fts_cg_embedded =  string_list_to_float(fts_cg_embedded_init, 'fts_10e8_ppm_1us_AA_float' )
print(len(fts_cg_embedded))


# In[3]:


#general functions for calculations 
#plane fit


#from MM 
def planeFit(points):
	points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
	assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
	ctr = points.mean(axis=1)
	x = points - ctr[:,np.newaxis]
	M = np.dot(x, x.T) # Could also use np.cov(x) here.
	return ctr, svd(M)[0][:,-1]
#calc angle 

def angle(v1, v2):
# v1 is your firsr vector
# v2 is your second vector
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    #f (acute == True):
    return 90-np.degrees(angle)

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


def string_list_to_float(list_name, list_name_float ):
    #list_name is var name of filled list
    #list_name_float is str for name of random 'empty' list 
    list_name_float = [] 
    for i in list_name:
        ft_vect = [] 
        for j in i:
            ft_vect.append(float(j))
        list_name_float.append(ft_vect)
    list_name_float = np.array(list_name_float)
    return list_name_float 


# In[4]:


def plot_angles2(variable_domain_angles, short_axis_angles, starting_angles, color, darkcolor, prefix):
    #input: variable_domain_angles/short_axis_angles- np array of angles 
    #input: color - string of color code for plot 
    #prefix - string of prefix to add to plot name
    #starting_points - np array of variable & short ax angles 
    sns.set(rc={'figure.figsize':(8,8)})
    sns.set_style("whitegrid", {'axes.linewidth': 1, 'axes.edgecolor':'black'})

    a_plot = sns.scatterplot( x=variable_domain_angles, y=short_axis_angles,
                color=color, alpha=.2)
    a_plot = sns.kdeplot(variable_domain_angles, short_axis_angles, gridsize=200,
                         color=darkcolor, edgecolor="black")  # fill=True,
    a_plot = sns.scatterplot( x=starting_angles[:,0], y=starting_angles[:,1],
                 color='black', alpha=1,edgecolor="black" )

#     #set axes ranges 
    a_plot.set_ylim([-100,100])
    a_plot.set_xlim([-100, 100])
    #a_plot.set(rc={'figure.figsize':(8,8)}) 
    #control tick mark &  text size 
    xtick_loc = [-100, -50, 0, 50, 100]
    ytick_loc = [ -50, 0, 50, 100]
    a_plot.set_xticks(xtick_loc)
    a_plot.set_yticks(ytick_loc)
    a_plot.tick_params(axis="x", labelsize=42)
    a_plot.tick_params(axis="y", labelsize=42)

    #make transparent background 
    #plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor") 
    
    #save as png 
    plot_name = prefix+"_angles_kdeplot.png"
    a_plot.figure.savefig(plot_name, transparent=True )
    return 


# In[5]:


#method to calculate angles of ab - all atom (4e10 & pgzl1 )


def getStartingAngle_AA(pdb):
    
    #selection strings 
    mem_selection_str = 'resname POPC POPA CHOL'
    fab_selection_str = 'name CA'
    #read in pdb
    input_pdb = pdb #parsePDB(pdb_fp)
    #move pdb to origin 
    moveAtoms(input_pdb, to=np.zeros(3))
    #writePDB('testAA_moved20.pdb', input_pdb )
    #writePDB('aa_input_2zero.pdb', input_pdb)
    
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
    #writePDB('aa_transform_fabUp.pdb', input_pdb)
    
    
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
    #writePDB('aa_transform_z2zero_pgzl1.pdb', input_pdb)
    #select 2 fab axes 
    
    #calc angle b/w approach angle & top phos plane
    res94 = fab[94]
    res320 = fab[320]
    point_cdr_loops = calcCenter(res94+res320)
    res41 = fab[41]
    res256 = fab[256]
    point_center_fab = calcCenter(res41+res256)     
    point_xy_3 = np.array([point_center_fab[0], point_center_fab[1], -50]) 
    var_domain_vect = np.array(point_cdr_loops-point_center_fab)
    xy_norm_vect = np.array(point_xy_3-point_center_fab)
    angle_var = round(angle(var_domain_vect, xy_norm_vect), 2)
    #     #angle of variable domain to membrane in xy space 

#     print(fab[94].getResname())
#     print(fab[320].getResname())
#     print(fab[41].getResname())
#     print(fab[256].getResname())
#     print(fab[107].getResname())
#     print(fab[340].getResname())
    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[107].getCoords()
    point_h_edge = fab[340].getCoords()
    point_xz_3 = np.array([point_l_edge[0], point_l_edge[1], -50])
    short_ax_vect = np.array(point_l_edge-point_h_edge)
    xz_norm_vect = np.array(point_xz_3-point_l_edge)
    angle_short = round(angle(short_ax_vect, xz_norm_vect), 2)
    

    #write psuedoatoms for selected points 
#     wtr1 = AtomGroup('Water') 
#     coords = np.array([fab[94].getCoords(), 
#                        fab[320].getCoords(),
#                        fab[41].getCoords(), 
#                        fab[256].getCoords(), 
#                        fab[107].getCoords(),
#                        fab[340].getCoords()],
#                        dtype=float)
#     wtr1.setCoords(coords) 
#     wtr1.setNames(['O', 'O', 'P', 'P', 'H', 'H' ])
#     wtr1.setResnums([1, 1, 1, 1, 1, 1])
#     wtr1.setResnames(['WAT', 'WAT', 'WAT', 'WAT', 'WAT', 'WAT'])
#     writePDB('AA_PGZL1_selected_ref_points.pdb', wtr1)
    
#     wtr1 = AtomGroup('Water') 
#     coords = np.array([point_cdr_loops, 
#                        point_center_fab,
#                        point_xy_3, 
#                        point_l_edge, 
#                        point_h_edge,
#                        point_xz_3],
#                        dtype=float)
#     wtr1.setCoords(coords) 
#     wtr1.setNames(['O', 'P', 'P', 'P', 'H', 'H', ])
#     wtr1.setResnums([1, 1, 1, 1, 1, 1])
#     wtr1.setResnames(['WAT', 'WAT', 'WAT', 'WAT', 'WAT', 'WAT'])
#     writePDB('CG_angle_calc_points.pdb', wtr1)
    
    #select fab axes 
    return [angle_var, angle_short]
    
    #calculate angles 
    
    
    
    


# In[6]:


#method to calculate angles of ab - all atom (pgzl1 )


def getStartingAngle_AA_PGZL1(pdb):
    
    #selection strings 
    mem_selection_str = 'resname POPC POPA CHOL'
    fab_selection_str = 'name CA'
    #read in pdb
    input_pdb = pdb #parsePDB(pdb_fp)
    #move pdb to origin 
    moveAtoms(input_pdb, to=np.zeros(3))
    #writePDB('testAA_moved20.pdb', input_pdb )
    #writePDB('aa_input_2zero.pdb', input_pdb)
    
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
    #writePDB('aa_transform_fabUp.pdb', input_pdb)
    
    
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
    #writePDB('aa_transform_z2zero_pgzl1.pdb', input_pdb)
    #select 2 fab axes 
    
    #calc angle b/w approach angle & top phos plane
    res94 = fab[94]
    res320 = fab[320]
    point_cdr_loops = calcCenter(res94+res320)
    res41 = fab[41]
    res256 = fab[256]
    point_center_fab = calcCenter(res41+res256)     
    point_xy_3 = np.array([point_center_fab[0], point_center_fab[1], -50]) 
    var_domain_vect = np.array(point_cdr_loops-point_center_fab)
    xy_norm_vect = np.array(point_xy_3-point_center_fab)
    angle_var = round(angle(var_domain_vect, xy_norm_vect), 2)
    #     #angle of variable domain to membrane in xy space 
    
#     for i in range(len(fab)):
#         print(i, fab[i].getResname())
#     print(fab[94].getResname())
#     print(fab[320].getResname())
#     print(fab[41].getResname())
#     print(fab[256].getResname())
#     print(fab[107].getResname())
#     print(fab[340].getResname())
    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[107].getCoords()
    point_h_edge = fab[340].getCoords()
    point_xz_3 = np.array([point_l_edge[0], point_l_edge[1], -50])
    short_ax_vect = np.array(point_l_edge-point_h_edge)
    xz_norm_vect = np.array(point_xz_3-point_l_edge)
    angle_short = round(angle(short_ax_vect, xz_norm_vect), 2)
    

    #write psuedoatoms for selected points 
#     wtr1 = AtomGroup('Water') 
#     coords = np.array([fab[94].getCoords(), 
#                        fab[320].getCoords(),
#                        fab[41].getCoords(), 
#                        fab[256].getCoords(), 
#                        fab[107].getCoords(),
#                        fab[340].getCoords()],
#                        dtype=float)
#     wtr1.setCoords(coords) 
#     wtr1.setNames(['O', 'O', 'P', 'P', 'H', 'H' ])
#     wtr1.setResnums([1, 1, 1, 1, 1, 1])
#     wtr1.setResnames(['WAT', 'WAT', 'WAT', 'WAT', 'WAT', 'WAT'])
#     writePDB('AA_PGZL1_selected_ref_points.pdb', wtr1)
    
#     wtr1 = AtomGroup('Water') 
#     coords = np.array([point_cdr_loops, 
#                        point_center_fab,
#                        point_xy_3, 
#                        point_l_edge, 
#                        point_h_edge,
#                        point_xz_3],
#                        dtype=float)
#     wtr1.setCoords(coords) 
#     wtr1.setNames(['O', 'P', 'P', 'P', 'H', 'H', ])
#     wtr1.setResnums([1, 1, 1, 1, 1, 1])
#     wtr1.setResnames(['WAT', 'WAT', 'WAT', 'WAT', 'WAT', 'WAT'])
#     writePDB('CG_angle_calc_points.pdb', wtr1)
    
    #select fab axes 
    return [angle_var, angle_short]
    
    #calculate angles 
    
    
    
    


# In[7]:


#method to calculate angles of ab - all atom (4e10 & pgzl1 )


def getStartingAngle_AA_10E8(pdb):
    
    #selection strings 
    mem_selection_str = 'resname POPC POPA CHOL'
    fab_selection_str = 'name CA'
    #read in pdb
    input_pdb = pdb #parsePDB(pdb_fp)
    #move pdb to origin 
    moveAtoms(input_pdb, to=np.zeros(3))
    #writePDB('testAA_moved20.pdb', input_pdb )
    #writePDB('aa_input_2zero.pdb', input_pdb)
    
    membrane = input_pdb.select(mem_selection_str)
    fab = input_pdb.select(fab_selection_str)

    #align so psuedo axis is aligned to positive z direction 
    #define axis through fab center and membrane center - force fab to point upwards in z direction 
    pseudo_fab_cen = calcCenter(fab.select('resnum 41 or resnum 273'))
    membrane_cen = calcCenter(membrane)
    psuedo_central_ax = np.array(pseudo_fab_cen-membrane_cen)
    #must normalize axis vector before transforming 
    psuedo_central_ax_norm = psuedo_central_ax / np.linalg.norm(psuedo_central_ax)
    rotation = VectorAlign(psuedo_central_ax_norm, np.array([0, 0, 1]))
    transformCentralAx = Transformation(rotation, np.zeros(3))
    applyTransformation(transformCentralAx, input_pdb)
    #writePDB('aa_transform_fabUp.pdb', input_pdb)
    
    
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
    #writePDB('aa_transform_z2zero_10E8.pdb', input_pdb)
    #select 2 fab axes 
    
    #calc angle b/w approach angle & top phos plane
    res110 = fab[110]
    res325 = fab[325]
    point_cdr_loops = calcCenter(res110+res325)
    res41 = fab[41]
    res273 = fab[273]
    point_center_fab = calcCenter(res41+res273)     
    point_xy_3 = np.array([point_center_fab[0], point_center_fab[1], -50]) 
    var_domain_vect = np.array(point_cdr_loops-point_center_fab)
    xy_norm_vect = np.array(point_xy_3-point_center_fab)
    angle_var = round(angle(var_domain_vect, xy_norm_vect), 2)
    #     #angle of variable domain to membrane in xy space 
    
#     for i in range(len(fab)):
#         print(i, fab[i].getResname())
        
#     print(fab[110].getResname())
#     print(fab[325].getResname())
#     print(fab[41].getResname())
#     print(fab[273].getResname())
#     print(fab[130].getResname())
#     print(fab[340].getResname())
    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[130].getCoords()
    point_h_edge = fab[340].getCoords()
    point_xz_3 = np.array([point_l_edge[0], point_l_edge[1], -50])
    #swap these for 10e8 b/x chain order 
    short_ax_vect = np.array(point_h_edge-point_l_edge)
    xz_norm_vect = np.array(point_xz_3-point_l_edge)
    angle_short = round(angle(short_ax_vect, xz_norm_vect), 2)
    

#     #write psuedoatoms for selected points 
#     wtr1 = AtomGroup('Water') 
#     coords = np.array([fab[110].getCoords(), 
#                        fab[325].getCoords(),
#                        fab[41].getCoords(), 
#                        fab[273].getCoords(), 
#                        fab[130].getCoords(),
#                        fab[340].getCoords()],
#                        dtype=float)
#     wtr1.setCoords(coords) 
#     wtr1.setNames(['O', 'O', 'P', 'P', 'H', 'H' ])
#     wtr1.setResnums([1, 1, 1, 1, 1, 1])
#     wtr1.setResnames(['WAT', 'WAT', 'WAT', 'WAT', 'WAT', 'WAT'])
#     writePDB('AA_10E8_selected_ref_points.pdb', wtr1)
    
#     wtr1 = AtomGroup('Water') 
#     coords = np.array([point_cdr_loops, 
#                        point_center_fab,
#                        point_xy_3, 
#                        point_l_edge, 
#                        point_h_edge,
#                        point_xz_3],
#                        dtype=float)
#     wtr1.setCoords(coords) 
#     wtr1.setNames(['O', 'O', 'O', 'P', 'P', 'P', ])
#     wtr1.setResnums([1, 1, 1, 1, 1, 1])
#     wtr1.setResnames(['WAT', 'WAT', 'WAT', 'WAT', 'WAT', 'WAT'])
#     writePDB('AA_angle_calc_points_10e8.pdb', wtr1)
    
    #select fab axes 
    return [angle_var, angle_short]
    
    #calculate angles 
    
    
    
    


# In[8]:


#method to calculate angles of ab to membrane - coarse grain (4e10 & pgzl1 )

def getStartingAngle_CG(pdb):
    
    #selection strings 
    mem_selection_str = 'resname POPC POPA CHOL'
    fab_selection_str = 'name BB'
    #read in pdb
    input_pdb = pdb 
    #move pdb to origin 
    moveAtoms(input_pdb, to=np.zeros(3))
    #writePDB('cg_input_2zero.pdb', input_pdb)
    #pre select po4 membrane based on geometric constraints to avoid micelle lipids 
    po4_membrane = input_pdb.select('name PO4 and z<90') #and z > 45 or z < 90 
    
    
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
    rotation = VectorAlign(psuedo_central_ax_norm, np.array([0, 0, 1]))
    transformCentralAx = Transformation(rotation, np.zeros(3))
    applyTransformation(transformCentralAx, input_pdb)
    #writePDB('cg_transform_fabUp.pdb', input_pdb)
    
    mem_plane = planeFit(np.transpose(po4_membrane.getCoords()))
    mem_plane_normal = mem_plane[1]

    #this doesnt work in CG for some reason - definition of plane norm is wrong direciton
    #CG systems are already aligned so Z calculations are accurate - can skip for CG 
    #define rotation as alignement from psueo-central-axis to z plane norm vect 
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
   # writePDB('cg_transform_z2zero_10E8.pdb', input_pdb)
    #select 2 fab axes 
    
    #calc angle b/w approach angle & top phos plane
    res94 = fab[94]
    res320 = fab[320]
    point_cdr_loops = calcCenter(res94+res320)
    res41 = fab[41]
    res256 = fab[256]
    point_center_fab = calcCenter(res41+res256)     
    point_xy_3 = np.array([point_center_fab[0], point_center_fab[1], -50]) 
    var_domain_vect = np.array(point_cdr_loops-point_center_fab)
    xy_norm_vect = np.array(point_xy_3-point_center_fab)
    angle_var = round(angle(var_domain_vect, xy_norm_vect), 2)
#     #angle of variable domain to membrane in xy space 

    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[107].getCoords()
    point_h_edge = fab[340].getCoords()
    point_xz_3 = np.array([point_l_edge[0], point_l_edge[1], -50])
    short_ax_vect = np.array(point_l_edge-point_h_edge)
    xz_norm_vect = np.array(point_xz_3-point_l_edge)
    angle_short = round(angle(short_ax_vect, xz_norm_vect), 2) 
    
#     print(fab[94].getResname())
#     print(fab[320].getResname())
#     print(fab[41].getResname())
#     print(fab[256].getResname())
#     print(fab[107].getResname())
#     print(fab[340].getResname())
#     #write psuedoatoms for selected points 
#     wtr1 = AtomGroup('Water') 
#     coords = np.array([fab[94].getCoords(), 
#                        fab[320].getCoords(),
#                        fab[41].getCoords(), 
#                        fab[256].getCoords(), 
#                        fab[107].getCoords(),
#                        fab[340].getCoords()],
#                        dtype=float)
#     wtr1.setCoords(coords) 
#     wtr1.setNames(['O', 'O', 'P', 'P', 'H', 'H' ])
#     wtr1.setResnums([1, 1, 1, 1, 1, 1])
#     wtr1.setResnames(['WAT', 'WAT', 'WAT', 'WAT', 'WAT', 'WAT'])
#     writePDB('CG_selected_ref_points.pdb', wtr1)
    
#     wtr1 = AtomGroup('Water') 
#     coords = np.array([point_cdr_loops, 
#                        point_center_fab,
#                        point_xy_3, 
#                        point_l_edge, 
#                        point_h_edge,
#                        point_xz_3],
#                        dtype=float)
#     wtr1.setCoords(coords) 
#     wtr1.setNames(['O', 'P', 'P', 'P', 'H', 'H', ])
#     wtr1.setResnums([1, 1, 1, 1, 1, 1])
#     wtr1.setResnames(['WAT', 'WAT', 'WAT', 'WAT', 'WAT', 'WAT'])
#     writePDB('CG_angle_calc_points.pdb', wtr1)
#     writePDB('CG_angle_structure.pdb', fab_mem)
    
    #select fab axes 
    return [angle_var, angle_short] 


# In[44]:


#method to calculate angles of ab to membrane - coarse grain (4e10 & pgzl1 )

def getStartingAngle_CG_spont(pdb):
    
    #selection strings 
    mem_selection_str = 'resname POPC POPA CHOL'
    fab_selection_str = 'name BB'
    #read in pdb
    input_pdb = pdb 
    #move pdb to origin 
    moveAtoms(input_pdb, to=np.zeros(3))
    #writePDB('cg_input_2zero.pdb', input_pdb)
    #pre select po4 membrane based on geometric constraints to avoid micelle lipids 
    po4_membrane = input_pdb.select('name PO4 and z>5') #and z > 45 or z < 90 
    #mem_coords = po4_membrane.getCoords()[2]
    writePDB('testPO4_cg_spont.pdb', po4_membrane)
    
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
    rotation = VectorAlign(psuedo_central_ax_norm, np.array([0, 0, 1]))
    transformCentralAx = Transformation(rotation, np.zeros(3))
    applyTransformation(transformCentralAx, input_pdb)
    #writePDB('cg_transform_fabUp.pdb', input_pdb)
    
    mem_plane = planeFit(np.transpose(po4_membrane.getCoords()))
    mem_plane_normal = mem_plane[1]

    #this doesnt work in CG for some reason - definition of plane norm is wrong direciton
    #CG systems are already aligned so Z calculations are accurate - can skip for CG 
    #define rotation as alignement from psueo-central-axis to z plane norm vect 
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
   # writePDB('cg_transform_z2zero_10E8.pdb', input_pdb)
    #select 2 fab axes 
    
    #calc angle b/w approach angle & top phos plane
    res94 = fab[94]
    res320 = fab[320]
    point_cdr_loops = calcCenter(res94+res320)
    res41 = fab[41]
    res256 = fab[256]
    point_center_fab = calcCenter(res41+res256)     
    point_xy_3 = np.array([point_center_fab[0], point_center_fab[1], -50]) 
    var_domain_vect = np.array(point_cdr_loops-point_center_fab)
    xy_norm_vect = np.array(point_xy_3-point_center_fab)
    angle_var = round(angle(var_domain_vect, xy_norm_vect), 2)
#     #angle of variable domain to membrane in xy space 

    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[107].getCoords()
    point_h_edge = fab[340].getCoords()
    point_xz_3 = np.array([point_l_edge[0], point_l_edge[1], -50])
    short_ax_vect = np.array(point_l_edge-point_h_edge)
    xz_norm_vect = np.array(point_xz_3-point_l_edge)
    angle_short = round(angle(short_ax_vect, xz_norm_vect), 2) 

    
    #select fab axes 
    return [angle_var, angle_short] 


# In[ ]:





# In[43]:


#method to calculate angles of ab to membrane - coarse grain (4e10 & pgzl1 )
#plane selection & reference points adjusted for bottom membrane 
def getStartingAngle_CG_spont_bot(pdb):
    
    #selection strings 
    mem_selection_str = 'resname POPC POPA CHOL'
    fab_selection_str = 'name BB'
    #read in pdb
    input_pdb = pdb 
    #move pdb to origin 
    moveAtoms(input_pdb, to=np.zeros(3))
    #writePDB('cg_input_2zero.pdb', input_pdb)
    #pre select po4 membrane based on geometric constraints to avoid micelle lipids 
    po4_membrane = input_pdb.select('name PO4 and z<5') #and z > 45 or z < 90 
    #mem_coords = po4_membrane.getCoords()[2]
    
    
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
    rotation = VectorAlign(psuedo_central_ax_norm, np.array([0, 0, 1]))
    transformCentralAx = Transformation(rotation, np.zeros(3))
    applyTransformation(transformCentralAx, input_pdb)
    
    #writePDB('testPO4_cg_spont_bot.pdb', po4_membrane)
    #writePDB('cg_transform_fabUp.pdb', input_pdb)
    
    mem_plane = planeFit(np.transpose(po4_membrane.getCoords()))
    mem_plane_normal = mem_plane[1]

    #this doesnt work in CG for some reason - definition of plane norm is wrong direciton
    #CG systems are already aligned so Z calculations are accurate - can skip for CG 
    #define rotation as alignement from psueo-central-axis to z plane norm vect 
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
   # writePDB('cg_transform_z2zero_10E8.pdb', input_pdb)
    #select 2 fab axes 
    
    #calc angle b/w approach angle & top phos plane
    res94 = fab[94]
    res320 = fab[320]
    point_cdr_loops = calcCenter(res94+res320)
    res41 = fab[41]
    res256 = fab[256]
    point_center_fab = calcCenter(res41+res256)     
    point_xy_3 = np.array([point_center_fab[0], point_center_fab[1], -50]) 
    var_domain_vect = np.array(point_cdr_loops-point_center_fab)
    xy_norm_vect = np.array(point_xy_3-point_center_fab)
    angle_var = round(angle(var_domain_vect, xy_norm_vect), 2)
#     #angle of variable domain to membrane in xy space 

    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[107].getCoords()
    point_h_edge = fab[340].getCoords()
    point_xz_3 = np.array([point_l_edge[0], point_l_edge[1], -50])
    short_ax_vect = np.array(point_l_edge-point_h_edge)
    xz_norm_vect = np.array(point_xz_3-point_l_edge)
    #test 
    #xz_norm_vect = np.array(point_l_edge-point_xz_3)
    angle_short = round(angle(short_ax_vect, xz_norm_vect), 2) 

#     wtr1 = AtomGroup('Water') 
#     coords = np.array([point_l_edge, 
#                        point_h_edge,
#                        point_xz_3, 
#                        point_xy_3, 
#                        point_cdr_loops,
#                        point_center_fab],
#                        dtype=float)
#     wtr1.setCoords(coords) 
#     wtr1.setNames(['P', 'P', 'P', 'N', 'N', 'N' ])
#     wtr1.setResnums([1, 1, 1, 1, 1, 1])
#     wtr1.setResnames(['WAT', 'WAT', 'WAT', 'WAT', 'WAT', 'WAT'])
#     writePDB('CG_selected_ref_points_BOT.pdb', wtr1)
#     writePDB('CG_FULL_TEST_BOT.pdb', input_pdb)

    #select fab axes 
    return [angle_var, angle_short] 


# In[53]:


pdb_in = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/4e10/bot_TEST.pdb"
pdb=parsePDB(pdb_in)
print(getStartingAngle_CG_spont_bot(pdb))


# In[11]:


#method to calculate angles of ab to membrane - coarse grain (4e10 & pgzl1 )
#cg chains are inverse relative to AA strucutre 
def getStartingAngle_CG_PGZL1(pdb):
    
    #selection strings 
    mem_selection_str = 'resname POPC POPA CHOL'
    fab_selection_str = 'name BB'
    #read in pdb
    input_pdb = pdb 
    #move pdb to origin 
    moveAtoms(input_pdb, to=np.zeros(3))
    #writePDB('cg_input_2zero.pdb', input_pdb)
    #pre select po4 membrane based on geometric constraints to avoid micelle lipids 
    po4_membrane = input_pdb.select('name PO4 and z<90') #and z > 45 or z < 90 
    
    
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
    rotation = VectorAlign(psuedo_central_ax_norm, np.array([0, 0, 1]))
    transformCentralAx = Transformation(rotation, np.zeros(3))
    applyTransformation(transformCentralAx, input_pdb)
    #writePDB('cg_transform_fabUp.pdb', input_pdb)
    
    mem_plane = planeFit(np.transpose(po4_membrane.getCoords()))
    mem_plane_normal = mem_plane[1]

    #this doesnt work in CG for some reason - definition of plane norm is wrong direciton
    #CG systems are already aligned so Z calculations are accurate - can skip for CG 
    #define rotation as alignement from psueo-central-axis to z plane norm vect 
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
    #writePDB('cg_transform_z2zero_10E8.pdb', input_pdb)
    #select 2 fab axes 
    
    #calc angle b/w approach angle & top phos plane
    #numbering of hcains is swapped b/c H,L order in Cg system- this is accurate 10.13.21
    res317 = fab[317]
    res106 = fab[106]
    point_cdr_loops = calcCenter(res317+res106)
    res264 = fab[264]
    res42 = fab[42]
    point_center_fab = calcCenter(res42+res264)     
    point_xy_3 = np.array([point_center_fab[0], point_center_fab[1], -50]) 
    var_domain_vect = np.array(point_cdr_loops-point_center_fab)
    xy_norm_vect = np.array(point_xy_3-point_center_fab)
    angle_var = round(angle(var_domain_vect, xy_norm_vect), 2)
#     #angle of variable domain to membrane in xy space 

    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[330].getCoords()
    point_h_edge = fab[126].getCoords()
    point_xz_3 = np.array([point_l_edge[0], point_l_edge[1], -50])
    short_ax_vect = np.array(point_l_edge-point_h_edge)
    xz_norm_vect = np.array(point_xz_3-point_l_edge)
    angle_short = round(angle(short_ax_vect, xz_norm_vect), 2) 
    
#     for i in range(len(fab)):
#          print(i, fab[i].getResname())
#     print(fab[94].getResname())
#     print(fab[320].getResname())
#     print(fab[41].getResname())
#     print(fab[256].getResname())
#     print(fab[107].getResname())
#     print(fab[340].getResname())
#     #write psuedoatoms for selected points 
#     wtr1 = AtomGroup('Water') 
#     coords = np.array([fab[94].getCoords(), 
#                        fab[320].getCoords(),
#                        fab[41].getCoords(), 
#                        fab[256].getCoords(), 
#                        fab[107].getCoords(),
#                        fab[340].getCoords()],
#                        dtype=float)
#     wtr1.setCoords(coords) 
#     wtr1.setNames(['O', 'O', 'P', 'P', 'H', 'H' ])
#     wtr1.setResnums([1, 1, 1, 1, 1, 1])
#     wtr1.setResnames(['WAT', 'WAT', 'WAT', 'WAT', 'WAT', 'WAT'])
#     writePDB('CG_selected_ref_points.pdb', wtr1)
    
#     wtr1 = AtomGroup('Water') 
#     coords = np.array([point_cdr_loops, 
#                        point_center_fab,
#                        point_xy_3, 
#                        point_l_edge, 
#                        point_h_edge,
#                        point_xz_3],
#                        dtype=float)
#     wtr1.setCoords(coords) 
#     wtr1.setNames(['O', 'P', 'P', 'P', 'H', 'H', ])
#     wtr1.setResnums([1, 1, 1, 1, 1, 1])
#     wtr1.setResnames(['WAT', 'WAT', 'WAT', 'WAT', 'WAT', 'WAT'])
#     writePDB('CG_angle_calc_points.pdb', wtr1)
#     writePDB('CG_angle_structure.pdb', fab_mem)
    
    #select fab axes 
    return [angle_var, angle_short] 


# In[64]:


#method to calculate angles of ab to membrane - coarse grain (4e10 & pgzl1 )
#cg chains are inverse relative to AA strucutre 
def getStartingAngle_CG_PGZL1_spont(pdb):
    
    #selection strings 
    mem_selection_str = 'resname POPC POPA CHOL'
    fab_selection_str = 'name BB'
    #read in pdb
    input_pdb = pdb 
    #move pdb to origin 
    moveAtoms(input_pdb, to=np.zeros(3))
    #writePDB('cg_input_2zero.pdb', input_pdb)
    #pre select po4 membrane based on geometric constraints to avoid micelle lipids 
    po4_membrane = input_pdb.select('name PO4 and z>5') #and z > 45 or z < 90 
    
    
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
    rotation = VectorAlign(psuedo_central_ax_norm, np.array([0, 0, 1]))
    transformCentralAx = Transformation(rotation, np.zeros(3))
    applyTransformation(transformCentralAx, input_pdb)
    #writePDB('cg_transform_fabUp.pdb', input_pdb)
    
    mem_plane = planeFit(np.transpose(po4_membrane.getCoords()))
    mem_plane_normal = mem_plane[1]

    #this doesnt work in CG for some reason - definition of plane norm is wrong direciton
    #CG systems are already aligned so Z calculations are accurate - can skip for CG 
    #define rotation as alignement from psueo-central-axis to z plane norm vect 
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
    #writePDB('cg_transform_z2zero_10E8.pdb', input_pdb)
    #select 2 fab axes 
    
    #calc angle b/w approach angle & top phos plane
    #numbering of hcains is swapped b/c H,L order in Cg system- this is accurate 10.13.21
    res317 = fab[317]
    res106 = fab[106]
    point_cdr_loops = calcCenter(res317+res106)
    res264 = fab[264]
    res42 = fab[42]
    point_center_fab = calcCenter(res42+res264)     
    point_xy_3 = np.array([point_center_fab[0], point_center_fab[1], -50]) 
    var_domain_vect = np.array(point_cdr_loops-point_center_fab)
    xy_norm_vect = np.array(point_xy_3-point_center_fab)
    angle_var = round(angle(var_domain_vect, xy_norm_vect), 2)
#     #angle of variable domain to membrane in xy space 

    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[330].getCoords()
    point_h_edge = fab[126].getCoords()
    point_xz_3 = np.array([point_l_edge[0], point_l_edge[1], -50])
    short_ax_vect = np.array(point_l_edge-point_h_edge)
    xz_norm_vect = np.array(point_xz_3-point_l_edge)
    angle_short = round(angle(short_ax_vect, xz_norm_vect), 2) 
    
    
    #select fab axes 
    return [angle_var, angle_short] 


# In[45]:


#method to calculate angles of ab to membrane - coarse grain (4e10 & pgzl1 )
#cg chains are inverse relative to AA strucutre 
def getStartingAngle_CG_PGZL1_spont_bot(pdb):
    
    #selection strings 
    mem_selection_str = 'resname POPC POPA CHOL'
    fab_selection_str = 'name BB'
    #read in pdb
    input_pdb = pdb 
    #move pdb to origin 
    moveAtoms(input_pdb, to=np.zeros(3))
    #writePDB('cg_input_2zero.pdb', input_pdb)
    #pre select po4 membrane based on geometric constraints to avoid micelle lipids 
    po4_membrane = input_pdb.select('name PO4 and z<5') #and z > 45 or z < 90 
    
    
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
    rotation = VectorAlign(psuedo_central_ax_norm, np.array([0, 0, 1]))
    transformCentralAx = Transformation(rotation, np.zeros(3))
    applyTransformation(transformCentralAx, input_pdb)
    #writePDB('cg_transform_fabUp.pdb', input_pdb)
    
    mem_plane = planeFit(np.transpose(po4_membrane.getCoords()))
    mem_plane_normal = mem_plane[1]

    #this doesnt work in CG for some reason - definition of plane norm is wrong direciton
    #CG systems are already aligned so Z calculations are accurate - can skip for CG 
    #define rotation as alignement from psueo-central-axis to z plane norm vect 
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
    #writePDB('cg_transform_z2zero_10E8.pdb', input_pdb)
    #select 2 fab axes 
    
    #calc angle b/w approach angle & top phos plane
    #numbering of hcains is swapped b/c H,L order in Cg system- this is accurate 10.13.21
    res317 = fab[317]
    res106 = fab[106]
    point_cdr_loops = calcCenter(res317+res106)
    res264 = fab[264]
    res42 = fab[42]
    point_center_fab = calcCenter(res42+res264)     
    point_xy_3 = np.array([point_center_fab[0], point_center_fab[1], -50]) 
    var_domain_vect = np.array(point_cdr_loops-point_center_fab)
    xy_norm_vect = np.array(point_xy_3-point_center_fab)
    angle_var = round(angle(var_domain_vect, xy_norm_vect), 2)
#     #angle of variable domain to membrane in xy space 

    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[330].getCoords()
    point_h_edge = fab[126].getCoords()
    point_xz_3 = np.array([point_l_edge[0], point_l_edge[1], -50])
    short_ax_vect = np.array(point_l_edge-point_h_edge)
    xz_norm_vect = np.array(point_xz_3-point_l_edge)
    angle_short = round(angle(short_ax_vect, xz_norm_vect), 2) 
    
    
    #select fab axes 
    return [angle_var, angle_short] 


# In[67]:


def getStartingAngle_CG_PGZL1_spont_bot_V2(pdb):
    
    #selection strings 
    mem_selection_str = 'resname POPC POPA CHOL'
    fab_selection_str = 'name BB'
    #read in pdb
    input_pdb = pdb 
    #move pdb to origin 
    moveAtoms(input_pdb, to=np.zeros(3))
    #writePDB('cg_input_2zero.pdb', input_pdb)
    #pre select po4 membrane based on geometric constraints to avoid micelle lipids 
    po4_membrane = input_pdb.select('name PO4 and z<5') #and z > 45 or z < 90 
    #mem_coords = po4_membrane.getCoords()[2]
    
    
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
    rotation = VectorAlign(psuedo_central_ax_norm, np.array([0, 0, 1]))
    transformCentralAx = Transformation(rotation, np.zeros(3))
    applyTransformation(transformCentralAx, input_pdb)
    
    #writePDB('testPO4_cg_spont_bot.pdb', po4_membrane)
    #writePDB('cg_transform_fabUp.pdb', input_pdb)
    
    mem_plane = planeFit(np.transpose(po4_membrane.getCoords()))
    mem_plane_normal = mem_plane[1]

    #this doesnt work in CG for some reason - definition of plane norm is wrong direciton
    #CG systems are already aligned so Z calculations are accurate - can skip for CG 
    #define rotation as alignement from psueo-central-axis to z plane norm vect 
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
   # writePDB('cg_transform_z2zero_10E8.pdb', input_pdb)
    #select 2 fab axes 
    
    #calc angle b/w approach angle & top phos plane
    res94 = fab[317]
    res320 = fab[106]
    point_cdr_loops = calcCenter(res94+res320)
    res41 = fab[264]
    res256 = fab[42]
    point_center_fab = calcCenter(res41+res256)     
    point_xy_3 = np.array([point_center_fab[0], point_center_fab[1], -50]) 
    var_domain_vect = np.array(point_cdr_loops-point_center_fab)
    xy_norm_vect = np.array(point_xy_3-point_center_fab)
    angle_var = round(angle(var_domain_vect, xy_norm_vect), 2)
#     #angle of variable domain to membrane in xy space 

    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[330].getCoords()
    point_h_edge = fab[126].getCoords()
    point_xz_3 = np.array([point_l_edge[0], point_l_edge[1], -50])
    short_ax_vect = np.array(point_l_edge-point_h_edge)
    xz_norm_vect = np.array(point_xz_3-point_l_edge)
    #test 
    #xz_norm_vect = np.array(point_l_edge-point_xz_3)
    angle_short = round(angle(short_ax_vect, xz_norm_vect), 2) 

#     wtr1 = AtomGroup('Water') 
#     coords = np.array([point_l_edge, 
#                        point_h_edge,
#                        point_xz_3, 
#                        point_xy_3, 
#                        point_cdr_loops,
#                        point_center_fab],
#                        dtype=float)
#     wtr1.setCoords(coords) 
#     wtr1.setNames(['P', 'P', 'P', 'N', 'N', 'N' ])
#     wtr1.setResnums([1, 1, 1, 1, 1, 1])
#     wtr1.setResnames(['WAT', 'WAT', 'WAT', 'WAT', 'WAT', 'WAT'])
#     writePDB('CG_selected_ref_points_BOT.pdb', wtr1)
#     writePDB('CG_FULL_TEST_BOT.pdb', input_pdb)

    #select fab axes 
    return [angle_var, angle_short] 


# In[47]:


#method to calculate angles of ab to membrane - coarse grain (4e10 & pgzl1 )

def getStartingAngle_CG_10E8(pdb):
    
    #selection strings 
    mem_selection_str = 'resname POPC POPA CHOL'
    fab_selection_str = 'name BB'
    #read in pdb
    input_pdb = pdb 
    #move pdb to origin 
    moveAtoms(input_pdb, to=np.zeros(3))
    #writePDB('cg_input_2zero.pdb', input_pdb)
    #pre select po4 membrane based on geometric constraints to avoid micelle lipids 
    po4_membrane = input_pdb.select('name PO4 and z<90') #and z > 45 or z < 90 
    
    
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
    rotation = VectorAlign(psuedo_central_ax_norm, np.array([0, 0, 1]))
    transformCentralAx = Transformation(rotation, np.zeros(3))
    applyTransformation(transformCentralAx, input_pdb)
    #writePDB('cg_transform_fabUp.pdb', input_pdb)
    
    mem_plane = planeFit(np.transpose(po4_membrane.getCoords()))
    mem_plane_normal = mem_plane[1]

    #this doesnt work in CG for some reason - definition of plane norm is wrong direciton
    #CG systems are already aligned so Z calculations are accurate - can skip for CG 
    #define rotation as alignement from psueo-central-axis to z plane norm vect 
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
    #writePDB('cg_transform_z2zero_10E8.pdb', input_pdb)
    #select 2 fab axes 
    
    #calc angle b/w approach angle & top phos plane
    res110 = fab[110]
    res324 = fab[324]
    point_cdr_loops = calcCenter(res110+res324)
    res41 = fab[41]
    res272 = fab[272]
    point_center_fab = calcCenter(res41+res272)     
    point_xy_3 = np.array([point_center_fab[0], point_center_fab[1], -50]) 
    var_domain_vect = np.array(point_cdr_loops-point_center_fab)
    xy_norm_vect = np.array(point_xy_3-point_center_fab)
    angle_var = round(angle(var_domain_vect, xy_norm_vect), 2)
#     #angle of variable domain to membrane in xy space 

    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[130].getCoords()
    point_h_edge = fab[339].getCoords()
    point_xz_3 = np.array([point_l_edge[0], point_l_edge[1], -50])
    #swapped angle calc b/c chain order is swapped in 10e8 (H,L)
    short_ax_vect = np.array(point_h_edge-point_l_edge)
    xz_norm_vect = np.array(point_xz_3-point_l_edge)
    angle_short = round(angle(short_ax_vect, xz_norm_vect), 2) 
    
#     for i in range(len(fab)):
# #         print(i, fab[i].getResname())
#     print(fab[110].getResname())
#     print(fab[324].getResname())
#     print(fab[41].getResname())
#     print(fab[272].getResname())
#     print(fab[130].getResname())
#     print(fab[339].getResname())
#     #write psuedoatoms for selected points 
#     wtr1 = AtomGroup('Water') 
#     coords = np.array([fab[110].getCoords(), 
#                        fab[324].getCoords(),
#                        fab[41].getCoords(), 
#                        fab[272].getCoords(), 
#                        fab[130].getCoords(),
#                        fab[339].getCoords()],
#                        dtype=float)
#     wtr1.setCoords(coords) 
#     wtr1.setNames(['O', 'O', 'P', 'P', 'H', 'H' ])
#     wtr1.setResnums([1, 1, 1, 1, 1, 1])
#     wtr1.setResnames(['WAT', 'WAT', 'WAT', 'WAT', 'WAT', 'WAT'])
#     writePDB('CG_10E8_selected_ref_points.pdb', wtr1)
    
# 

# wtr1.setCoords(coords) 
# wtr1.setNames(['O', 'O', 'O', 'P', 'P', 'P', ])
# wtr1.setResnums([1, 1, 1, 1, 1, 1])
# wtr1.setResnames(['WAT', 'WAT', 'WAT', 'WAT', 'WAT', 'WAT'])
# writePDB('CG_angle_calc_points_10e8.pdb', wtr1)
# #     writePDB('CG_angle_structure.pdb', fab_mem)
    
    #select fab axes 
    return [angle_var, angle_short] 


# In[68]:


#method to calculate angles of ab to membrane - coarse grain (4e10 & pgzl1 )

def getStartingAngle_CG_10E8_spont(pdb):
    
    #selection strings 
    mem_selection_str = 'resname POPC POPA CHOL'
    fab_selection_str = 'name BB'
    #read in pdb
    input_pdb = pdb 
    #move pdb to origin 
    moveAtoms(input_pdb, to=np.zeros(3))
    #writePDB('cg_input_2zero.pdb', input_pdb)
    #pre select po4 membrane based on geometric constraints to avoid micelle lipids 
    po4_membrane = input_pdb.select('name PO4 and z>5') #and z > 45 or z < 90 
    
    
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
    rotation = VectorAlign(psuedo_central_ax_norm, np.array([0, 0, 1]))
    transformCentralAx = Transformation(rotation, np.zeros(3))
    applyTransformation(transformCentralAx, input_pdb)
    #writePDB('cg_transform_fabUp.pdb', input_pdb)
    
    mem_plane = planeFit(np.transpose(po4_membrane.getCoords()))
    mem_plane_normal = mem_plane[1]

    #this doesnt work in CG for some reason - definition of plane norm is wrong direciton
    #CG systems are already aligned so Z calculations are accurate - can skip for CG 
    #define rotation as alignement from psueo-central-axis to z plane norm vect 
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
    #writePDB('cg_transform_z2zero_10E8.pdb', input_pdb)
    #select 2 fab axes 
    
    #calc angle b/w approach angle & top phos plane
    res110 = fab[110]
    res324 = fab[324]
    point_cdr_loops = calcCenter(res110+res324)
    res41 = fab[41]
    res272 = fab[272]
    point_center_fab = calcCenter(res41+res272)     
    point_xy_3 = np.array([point_center_fab[0], point_center_fab[1], -50]) 
    var_domain_vect = np.array(point_cdr_loops-point_center_fab)
    xy_norm_vect = np.array(point_xy_3-point_center_fab)
    angle_var = round(angle(var_domain_vect, xy_norm_vect), 2)
#     #angle of variable domain to membrane in xy space 

    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[130].getCoords()
    point_h_edge = fab[339].getCoords()
    point_xz_3 = np.array([point_l_edge[0], point_l_edge[1], -50])
    #swapped angle calc b/c chain order is swapped in 10e8 (H,L)
    short_ax_vect = np.array(point_h_edge-point_l_edge)
    xz_norm_vect = np.array(point_xz_3-point_l_edge)
    angle_short = round(angle(short_ax_vect, xz_norm_vect), 2) 
    

    #select fab axes 
    return [angle_var, angle_short] 


# In[49]:


#method to calculate angles of ab to membrane - coarse grain (4e10 & pgzl1 )

def getStartingAngle_CG_10E8_spont_bot(pdb):
    
    #selection strings 
    mem_selection_str = 'resname POPC POPA CHOL'
    fab_selection_str = 'name BB'
    #read in pdb
    input_pdb = pdb 
    #move pdb to origin 
    moveAtoms(input_pdb, to=np.zeros(3))
    #writePDB('cg_input_2zero.pdb', input_pdb)
    #pre select po4 membrane based on geometric constraints to avoid micelle lipids 
    po4_membrane = input_pdb.select('name PO4 and z<5') #and z > 45 or z < 90 
    
    
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
    rotation = VectorAlign(psuedo_central_ax_norm, np.array([0, 0, 1]))
    transformCentralAx = Transformation(rotation, np.zeros(3))
    applyTransformation(transformCentralAx, input_pdb)
    #writePDB('cg_transform_fabUp.pdb', input_pdb)
    
    mem_plane = planeFit(np.transpose(po4_membrane.getCoords()))
    mem_plane_normal = mem_plane[1]

    #this doesnt work in CG for some reason - definition of plane norm is wrong direciton
    #CG systems are already aligned so Z calculations are accurate - can skip for CG 
    #define rotation as alignement from psueo-central-axis to z plane norm vect 
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
    #writePDB('cg_transform_z2zero_10E8.pdb', input_pdb)
    #select 2 fab axes 
    
    #calc angle b/w approach angle & top phos plane
    res110 = fab[110]
    res324 = fab[324]
    point_cdr_loops = calcCenter(res110+res324)
    res41 = fab[41]
    res272 = fab[272]
    point_center_fab = calcCenter(res41+res272)     
    point_xy_3 = np.array([point_center_fab[0], point_center_fab[1], -50]) 
    var_domain_vect = np.array(point_cdr_loops-point_center_fab)
    xy_norm_vect = np.array(point_xy_3-point_center_fab)
    angle_var = round(angle(var_domain_vect, xy_norm_vect), 2)
#     #angle of variable domain to membrane in xy space 

    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[130].getCoords()
    point_h_edge = fab[339].getCoords()
    point_xz_3 = np.array([point_l_edge[0], point_l_edge[1], -50])
    #swapped angle calc b/c chain order is swapped in 10e8 (H,L)
    short_ax_vect = np.array(point_h_edge-point_l_edge)
    xz_norm_vect = np.array(point_xz_3-point_l_edge)
    angle_short = round(angle(short_ax_vect, xz_norm_vect), 2) 
    

    #select fab axes 
    return [angle_var, angle_short] 


# In[ ]:





# In[222]:


wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10E8_ppm/"
pdb_in = wd+"final_analysis_input.pdb"
pdb = parsePDB(pdb_in)

print("10E8 ppm input: ", getStartingAngle_AA_10E8(pdb))

wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_embedded/10e8_ab/"
pdb_in = wd+"analysis_input.pdb"
pdb = parsePDB(pdb_in)

print("10E8 CG  embedded input: ", getStartingAngle_CG_10E8(pdb))


# In[200]:


wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_ppm/"
pdb_in = wd+"final_analysis_input.pdb"
pdb = parsePDB(pdb_in)

print("PGZL1 ppm input: ", getStartingAngle_AA_PGZL1(pdb))

wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_embedded/pgzl1_ab/"
pdb_in = wd+"analysis_input.pdb"
pdb = parsePDB(pdb_in)

print("PGZL1 CG  embedded input: ", getStartingAngle_CG_PGZL1(pdb))


# In[17]:



def calculate_traj_angles_AA(input_pdb, dcd_traj, output_name): 
    #example output_name input varible: "10e8_ppm"
    #print(output_file_name)
    fab_selection_str = 'protein'
    mem_selection_str = 'resname POPC POPA CHOL'
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
        angle_running.append(getStartingAngle_AA(pdb)) 
    #save to csv file 
    file_out = output_name + "_angles.csv"
    np.savetxt(file_out, angle_running, delimiter =",",fmt ='% s')
    return print("finished " + file_out) 


# In[18]:



def calculate_traj_angles_AA_10E8(input_pdb, dcd_traj, output_name): 
    #example output_name input varible: "10e8_ppm"
    #print(output_file_name)
    fab_selection_str = 'protein'
    mem_selection_str = 'resname POPC POPA CHOL'
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
        angle_running.append(getStartingAngle_AA_10E8(pdb)) 
    #save to csv file 
    file_out = output_name + "_angles.csv"
    np.savetxt(file_out, angle_running, delimiter =",",fmt ='% s')
    return print("finished " + file_out) 


# In[19]:



def calculate_traj_angles_CG(input_pdb, dcd_traj, output_name): 
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
        angle_running.append(getStartingAngle_CG(pdb)) 
    #save to csv file 
    file_out = output_name + "_angles.csv"
    np.savetxt(file_out, angle_running, delimiter =",",fmt ='% s')
    return print("finished " + file_out) 


# In[20]:



def calculate_traj_angles_CG_PGZL1(input_pdb, dcd_traj, output_name): 
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
        angle_running.append(getStartingAngle_CG_PGZL1(pdb)) 
    #save to csv file 
    file_out = output_name + "_angles.csv"
    np.savetxt(file_out, angle_running, delimiter =",",fmt ='% s')
    return print("finished " + file_out) 


# In[21]:



def calculate_traj_angles_CG_10E8(input_pdb, dcd_traj, output_name): 
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
        angle_running.append(getStartingAngle_CG_10E8(pdb)) 
    #save to csv file 
    file_out = output_name + "_angles.csv"
    np.savetxt(file_out, angle_running, delimiter =",",fmt ='% s')
    return print("finished " + file_out) 


# In[22]:


wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_ppm/"
pdb_in = wd+"final_analysis_input.pdb"
pdb = parsePDB(pdb_in)

print("4e10 ppm input: ", getStartingAngle_AA(pdb))


# In[133]:


wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_ppm/"
pdb_in = wd+"final_analysis_input.pdb"
traj_in = wd+"final_analysis_traj.dcd"
calculate_traj_angles_AA(pdb_in, traj_in, "4E10_ppm")

wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_p15/"
pdb_in = wd+"final_analysis_input.pdb"
traj_in = wd+"final_analysis_traj.dcd"
calculate_traj_angles_AA(pdb_in, traj_in, "4E10_p15")

wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_n15/"
pdb_in = wd+"final_analysis_input.pdb"
traj_in = wd+"final_analysis_traj.dcd"
calculate_traj_angles_AA(pdb_in, traj_in, "4E10_n15")

wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_ppm_rep/"
pdb_in = wd+"final_analysis_input.pdb"
traj_in = wd+"final_analysis_traj.dcd"
calculate_traj_angles_AA(pdb_in, traj_in, "4E10_ppm_rep")


# In[23]:


#print out list of starting angles for 4e10 systems 
wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_ppm/"
pdb_in = wd+"final_analysis_input.pdb"
pdb = parsePDB(pdb_in)
print("4e10 ppm: ", getStartingAngle_AA(pdb)) 
wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_p15/"
pdb_in = wd+"final_analysis_input.pdb"
pdb = parsePDB(pdb_in)
print("4e10 p15: ", getStartingAngle_AA(pdb)) 
wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_n15/"
pdb_in = wd+"final_analysis_input.pdb"
pdb = parsePDB(pdb_in)
print("4e10 n15: ", getStartingAngle_AA(pdb)) 
wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/4e10_ppm_rep/"
pdb_in = wd+"final_analysis_input.pdb"
pdb = parsePDB(pdb_in)
print("4e10 ppm_rep: ", getStartingAngle_AA(pdb))


# In[80]:


starting_angle_4e10 = np.array([[75.48, 11.36],[60.5, 26.05], [83.6, -4.83], [72.85, 12.4]])

#starting_angle_4e10 = np.array([[75.48, 11.36],[60.5, 26.05], [72.85, 12.4]])


# In[78]:


#read in 4E10 AA 
file = open("4E10_ppm_angles.csv", "r")
csv_reader = csv.reader(file) 
fts_4e10_ppm_AA_init = []
counter = 0 
for row in csv_reader:
    if counter<5033:
        fts_4e10_ppm_AA_init.append(row)
        counter = counter+1
fts_4e10_ppm_AA =  string_list_to_float(fts_4e10_ppm_AA_init, 'fts_4e10_ppm_AA_float' )

file = open("4E10_p15_angles.csv", "r")
csv_reader = csv.reader(file) 
fts_4e10_p15_AA_init = []
counter = 0 
for row in csv_reader:
    if counter<5033:
        fts_4e10_p15_AA_init.append(row)
        counter = counter+1
fts_4e10_p15_AA =  string_list_to_float(fts_4e10_p15_AA_init, 'fts_4e10_p15_AA_float' )


file = open("4E10_n15_angles.csv", "r")
csv_reader = csv.reader(file) 
fts_4e10_n15_AA_init = []
counter = 0 
for row in csv_reader:
    if counter<5033:
        fts_4e10_n15_AA_init.append(row)
        counter = counter+1
fts_4e10_n15_AA =  string_list_to_float(fts_4e10_n15_AA_init, 'fts_4e10_n15_AA_float' )
    
file = open("4E10_ppm_rep_angles.csv", "r")
csv_reader = csv.reader(file) 
fts_4e10_ppm_1us_AA_init = []
counter = 0 
for row in csv_reader:
    if counter<5033:
        fts_4e10_ppm_1us_AA_init.append(row)
        counter = counter+1
fts_4e10_ppm_1us_AA =  string_list_to_float(fts_4e10_ppm_1us_AA_init, 'fts_4e10_ppm_1us_AA_float' )


#combine 4e10 AA 4us into aggregate list 
aggregate_4e10_AA_fts = [] 
for i in fts_4e10_ppm_AA:
    aggregate_4e10_AA_fts.append(i)
for i in fts_4e10_p15_AA:
    aggregate_4e10_AA_fts.append(i)
for i in fts_4e10_n15_AA:
    aggregate_4e10_AA_fts.append(i)
for i in fts_4e10_ppm_1us_AA:
    aggregate_4e10_AA_fts.append(i)

aggregate_4e10_AA_fts = np.array(aggregate_4e10_AA_fts)
#make 4e10 plots 




# In[81]:


plot_angles2(aggregate_4e10_AA_fts[:,0], aggregate_4e10_AA_fts[:,1], starting_angle_4e10,
           '#55A3FF', '#2A517F', '4e10_AA')


# In[147]:


wd = "/Users/cmaillie/Dropbox (Scripps Research)/manuscript/coarse_grain/4e10_ab/"
pdb_in = wd+"analysis.pdb"
traj_in = wd+"membrane_contact_frames_cen.dcd"
calculate_traj_angles_CG(pdb_in, traj_in, "4E10_embedded_cg")


# In[307]:


file = open("4E10_embedded_cg_angles.csv", "r")
csv_reader = csv.reader(file) 
fts_cg_embedded_init = []
counter = 0 
for row in csv_reader:
    if counter<5033:
        fts_cg_embedded_init.append(row)
        counter = counter+1
fts_cg_embedded =  string_list_to_float(fts_cg_embedded_init, 'fts_4e10_ppm_1us_AA_float' )


#make list of starting angles 4e10 embedded CG 
starting_angle_4e10_cg_embedded = [[17.54, -60.61], [24.63, -59.2], [22.17, -17.75], [4.96, 28.29], 
 [23.79, 54.73], [13.01, 62.13], [4.84, 29.3], [-9.62, -16.65], 
 [-19.03, -55.51], [-33.86, 8.84], [-20.14, 15.89], [6.76, 26.49], 
 [50.22, 30.64], [-18.63, -64.82], [-19.19, -53.5], [32.35, -2.36], 
 [-4.25, -26.22], [46.12, 32.19], [5.43, 27.0], [-34.41, 2.6], 
 [26.02, -58.84], [7.56, 18.87], [-48.87, -5.05], [-33.58, 35.5], 
 [8.29, 25.36], [50.08, 0.88], [-48.04, -30.18], [-9.34, 66.28], 
 [-26.39, 58.18], [32.42, -36.02], [51.54, 1.04], [5.91, 25.95], 
 [-35.32, 35.66], [-26.5, 56.98], [-17.3, 65.55], [4.53, 23.98], 
 [18.09, -19.54], [-5.39, -17.79], [7.34, 22.57], [18.57, 63.2], 
 [18.31, 54.26]]
starting_angle_4e10_cg_embedded=np.array(starting_angle_4e10_cg_embedded)
fts_cg_embedded = np.array(fts_cg_embedded)

plot_angles2(fts_cg_embedded[:,0], fts_cg_embedded[:,1], starting_angle_4e10_cg_embedded,
           '#55A3FF', '#2A517F', '4e10_cg_embedded')


# In[154]:


wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_ppm/"
pdb_in = wd+"final_analysis_input.pdb"
traj_in = wd+"final_analysis_traj.dcd"
calculate_traj_angles_AA(pdb_in, traj_in, "pgzl1_ppm")

wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_p15/"
pdb_in = wd+"final_analysis_input.pdb"
traj_in = wd+"final_analysis_traj.dcd"
calculate_traj_angles_AA(pdb_in, traj_in, "pgzl1_p15")

wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_n15/"
pdb_in = wd+"final_analysis_input.pdb"
traj_in = wd+"final_analysis_traj.dcd"
calculate_traj_angles_AA(pdb_in, traj_in, "pgzl1_n15")

wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_ppm_rep/"
pdb_in = wd+"final_analysis_input.pdb"
traj_in = wd+"final_analysis_traj.dcd"
calculate_traj_angles_AA(pdb_in, traj_in, "pgzl1_ppm_rep")


# In[155]:


#print out list of starting angles for pgzl1 systems 
wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_ppm/"
pdb_in = wd+"final_analysis_input.pdb"
pdb = parsePDB(pdb_in)
print("pgzl1 ppm: ", getStartingAngle_AA(pdb)) 
wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_p15/"
pdb_in = wd+"final_analysis_input.pdb"
pdb = parsePDB(pdb_in)
print("pgzl1 p15: ", getStartingAngle_AA(pdb)) 
wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_n15/"
pdb_in = wd+"final_analysis_input.pdb"
pdb = parsePDB(pdb_in)
print("pgzl1 n15: ", getStartingAngle_AA(pdb)) 
wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/pgzl1_ppm_rep/"
pdb_in = wd+"final_analysis_input.pdb"
pdb = parsePDB(pdb_in)
print("pgzl1 ppm_rep: ", getStartingAngle_AA(pdb))


# In[156]:


starting_angle_pgzl1 = np.array([[85.9, 3.69],[76.98, 18.35], [70.78, -11.19], [85.33, 2.73]])


# In[157]:


#read in pgzl1 AA 
file = open("pgzl1_ppm_angles.csv", "r")
csv_reader = csv.reader(file) 
fts_pgzl1_ppm_AA_init = []
counter = 0 
for row in csv_reader:
    if counter<5033:
        fts_pgzl1_ppm_AA_init.append(row)
        counter = counter+1
fts_pgzl1_ppm_AA =  string_list_to_float(fts_pgzl1_ppm_AA_init, 'fts_pgzl1_ppm_AA_float' )

file = open("pgzl1_p15_angles.csv", "r")
csv_reader = csv.reader(file) 
fts_pgzl1_p15_AA_init = []
counter = 0 
for row in csv_reader:
    if counter<5033:
        fts_pgzl1_p15_AA_init.append(row)
        counter = counter+1
fts_pgzl1_p15_AA =  string_list_to_float(fts_pgzl1_p15_AA_init, 'fts_pgzl1_p15_AA_float' )


file = open("pgzl1_n15_angles.csv", "r")
csv_reader = csv.reader(file) 
fts_pgzl1_n15_AA_init = []
counter = 0 
for row in csv_reader:
    if counter<5033:
        fts_pgzl1_n15_AA_init.append(row)
        counter = counter+1
fts_pgzl1_n15_AA =  string_list_to_float(fts_pgzl1_n15_AA_init, 'fts_pgzl1_n15_AA_float' )
    
file = open("pgzl1_ppm_rep_angles.csv", "r")
csv_reader = csv.reader(file) 
fts_pgzl1_ppm_1us_AA_init = []
counter = 0 
for row in csv_reader:
    if counter<5033:
        fts_pgzl1_ppm_1us_AA_init.append(row)
        counter = counter+1
fts_pgzl1_ppm_1us_AA =  string_list_to_float(fts_pgzl1_ppm_1us_AA_init, 'fts_pgzl1_ppm_1us_AA_float' )


#combine pgzl1 AA 4us into aggregate list 
aggregate_pgzl1_AA_fts = [] 
for i in fts_pgzl1_ppm_AA:
    aggregate_pgzl1_AA_fts.append(i)
for i in fts_pgzl1_p15_AA:
    aggregate_pgzl1_AA_fts.append(i)
for i in fts_pgzl1_n15_AA:
    aggregate_pgzl1_AA_fts.append(i)
for i in fts_pgzl1_ppm_1us_AA:
    aggregate_pgzl1_AA_fts.append(i)

aggregate_pgzl1_AA_fts = np.array(aggregate_pgzl1_AA_fts)
#make pgzl1 plots 




# In[308]:


plot_angles2(aggregate_pgzl1_AA_fts[:,0], aggregate_pgzl1_AA_fts[:,1], starting_angle_pgzl1,
           '#F79A99', '#A33634', 'pgzl1_AA')


# In[206]:


wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_embedded/pgzl1_ab/"
pdb_in = wd+"analysis_input.pdb"
traj_in = wd+"membrane_contact_frames.dcd"
calculate_traj_angles_CG_PGZL1(pdb_in, traj_in, "pgzl1_embedded_cg")


# In[309]:


file = open("pgzl1_embedded_cg_angles.csv", "r")
csv_reader = csv.reader(file) 
fts_cg_embedded_init = []
counter = 0 
for row in csv_reader:
    if counter<5033:
        fts_cg_embedded_init.append(row)
        counter = counter+1
fts_cg_embedded =  string_list_to_float(fts_cg_embedded_init, 'fts_pgzl1_ppm_1us_AA_float' )


#make list of starting angles pgzl1 embedded CG 
starting_angle_pgzl1_cg_embedded = [[21.81, -44.04], [18.46, -59.47], [5.7, -28.24], [-8.56, 11.93], 
                                    [25.16, 72.04], [13.15, 57.05], [-7.69, 13.17], [-23.21, -31.36], 
                                    [-24.89, -70.99], [-46.41, -15.28], [-8.53, 26.52], [-9.34, 9.77], 
                                    [31.81, 31.05], [-14.04, -56.75], [-23.55, -72.31], [53.1, 12.84], 
                                    [5.69, -12.39], [33.99, 30.05], [-8.7, 12.74], [-49.72, -9.71], 
                                    [21.27, -55.73], [23.56, 29.84], [-32.77, -0.4], [-51.38, 18.86], 
                                    [-8.78, 13.34], [35.06, -1.07], [-36.93, -28.23], [-16.04, 50.4], 
                                    [-19.11, 58.8], [48.89, -19.4], [30.54, -0.79], [-10.79, 11.21], 
                                    [-42.91, 22.37], [-19.74, 58.05], [-18.25, 50.7], [-8.14, 13.33], 
                                    [6.14, -24.25], [-21.1, -31.6], [-5.74, 7.97], [13.94, 57.85], 
                                    [31.29, 65.48]]

starting_angle_pgzl1_cg_embedded=np.array(starting_angle_pgzl1_cg_embedded)
fts_cg_embedded = np.array(fts_cg_embedded)

plot_angles2(fts_cg_embedded[:,0], fts_cg_embedded[:,1], starting_angle_pgzl1_cg_embedded,
           '#F79A99', '#A33634', 'pgzl1_cg_embedded')


# In[ ]:





# In[230]:


#10e8 AA 
wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_ppm/"
pdb_in = wd+"final_analysis_input.pdb"
traj_in = wd+"final_analysis_traj.dcd"
calculate_traj_angles_AA_10E8(pdb_in, traj_in, "10e8_ppm")

wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_p15/"
pdb_in = wd+"final_analysis_input.pdb"
traj_in = wd+"final_analysis_traj.dcd"
calculate_traj_angles_AA_10E8(pdb_in, traj_in, "10e8_p15")

wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_n15/"
pdb_in = wd+"final_analysis_input.pdb"
traj_in = wd+"final_analysis_traj.dcd"
calculate_traj_angles_AA_10E8(pdb_in, traj_in, "10e8_n15")

wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_ppm_rep/"
pdb_in = wd+"final_analysis_input.pdb"
traj_in = wd+"final_analysis_traj.dcd"
calculate_traj_angles_AA_10E8(pdb_in, traj_in, "10e8_ppm_rep")


# In[226]:


#print out list of starting angles for 10e8 systems 
wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_ppm/"
pdb_in = wd+"final_analysis_input.pdb"
pdb = parsePDB(pdb_in)
print("10e8 ppm: ", getStartingAngle_AA_10E8(pdb)) 
wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_p15/"
pdb_in = wd+"final_analysis_input.pdb"
pdb = parsePDB(pdb_in)
print("10e8 p15: ", getStartingAngle_AA_10E8(pdb)) 
wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_n15/"
pdb_in = wd+"final_analysis_input.pdb"
pdb = parsePDB(pdb_in)
print("10e8 n15: ", getStartingAngle_AA_10E8(pdb)) 
wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/10e8_ppm_rep/"
pdb_in = wd+"final_analysis_input.pdb"
pdb = parsePDB(pdb_in)
print("10e8 ppm_rep: ", getStartingAngle_AA_10E8(pdb))


# In[227]:


starting_angle_10e8 = np.array([[38.08, 29.27],[38.96, 12.6], [24.87, 31.37], [39.52, 28.45]])


# In[231]:


#read in 10e8 AA 
file = open("10e8_ppm_angles.csv", "r")
csv_reader = csv.reader(file) 
fts_10e8_ppm_AA_init = []
counter = 0 
for row in csv_reader:
    if counter<5033:
        fts_10e8_ppm_AA_init.append(row)
        counter = counter+1
fts_10e8_ppm_AA =  string_list_to_float(fts_10e8_ppm_AA_init, 'fts_10e8_ppm_AA_float' )

file = open("10e8_p15_angles.csv", "r")
csv_reader = csv.reader(file) 
fts_10e8_p15_AA_init = []
counter = 0 
for row in csv_reader:
    if counter<5033:
        fts_10e8_p15_AA_init.append(row)
        counter = counter+1
fts_10e8_p15_AA =  string_list_to_float(fts_10e8_p15_AA_init, 'fts_10e8_p15_AA_float' )


file = open("10e8_n15_angles.csv", "r")
csv_reader = csv.reader(file) 
fts_10e8_n15_AA_init = []
counter = 0 
for row in csv_reader:
    if counter<5033:
        fts_10e8_n15_AA_init.append(row)
        counter = counter+1
fts_10e8_n15_AA =  string_list_to_float(fts_10e8_n15_AA_init, 'fts_10e8_n15_AA_float' )
    
file = open("10e8_ppm_rep_angles.csv", "r")
csv_reader = csv.reader(file) 
fts_10e8_ppm_1us_AA_init = []
counter = 0 
for row in csv_reader:
    if counter<5033:
        fts_10e8_ppm_1us_AA_init.append(row)
        counter = counter+1
fts_10e8_ppm_1us_AA =  string_list_to_float(fts_10e8_ppm_1us_AA_init, 'fts_10e8_ppm_1us_AA_float' )


#combine 10e8 AA 4us into aggregate list 
aggregate_10e8_AA_fts = [] 
for i in fts_10e8_ppm_AA:
    aggregate_10e8_AA_fts.append(i)
for i in fts_10e8_p15_AA:
    aggregate_10e8_AA_fts.append(i)
for i in fts_10e8_n15_AA:
    aggregate_10e8_AA_fts.append(i)
for i in fts_10e8_ppm_1us_AA:
    aggregate_10e8_AA_fts.append(i)

aggregate_10e8_AA_fts = np.array(aggregate_10e8_AA_fts)
#make 10e8 plots 




# In[310]:


plot_angles2(aggregate_10e8_AA_fts[:,0], aggregate_10e8_AA_fts[:,1], starting_angle_10e8,
           '#C28EE8', '#460273', '10e8_AA')


# In[233]:


#10e8 CG embedded 
wd = "/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_embedded/10e8_ab/"
pdb_in = wd+"analysis_input.pdb"
traj_in = wd+"membrane_contact_frames.dcd"
calculate_traj_angles_CG_10E8(pdb_in, traj_in, "10e8_embedded_cg")


# In[311]:


file = open("10e8_embedded_cg_angles.csv", "r")
csv_reader = csv.reader(file) 
fts_cg_embedded_init = []
counter = 0 
for row in csv_reader:
    if counter<5033:
        fts_cg_embedded_init.append(row)
        counter = counter+1
fts_cg_embedded =  string_list_to_float(fts_cg_embedded_init, 'fts_10e8_ppm_1us_AA_float' )


#make list of starting angles 10e8 embedded CG 
starting_angle_10e8_cg_embedded = [[-61.51, -35.43], [-72.57, -42.65], [-26.88, -25.9], [21.09, 5.07], 
                                 [40.48, -5.07], [44.97, 5.52], [23.35, 6.06], [-14.29, 4.45], 
                                 [-41.91, 1.05], [17.89, 35.06], [24.05, 23.31], [18.9, 2.96], 
                                 [12.3, -32.02], [-45.97, -2.04], [-39.7, 2.31], [-11.54, -31.86], 
                                 [-20.8, -6.65], [12.48, -27.64], [18.91, 2.2], [12.54, 34.38], 
                                 [-73.46, -44.35], [17.41, -2.06], [11.24, 41.11], [44.49, 45.9], 
                                 [19.55, 1.99], [-16.28, -43.6], [-10.41, 30.16], [62.79, 29.77], 
                                 [72.32, 43.35], [-45.12, -45.81], [-16.31, -44.85], [19.79, 3.9], 
                                 [45.58, 47.54], [69.49, 43.64], [69.85, 35.63], [17.97, 3.84], 
                                 [-29.47, -23.74], [-17.75, 0.4], [15.1, 0.76], [44.03, 1.6], 
                                 [42.95, -0.4]]

starting_angle_10e8_cg_embedded=np.array(starting_angle_10e8_cg_embedded)
fts_cg_embedded = np.array(fts_cg_embedded)

plot_angles2(fts_cg_embedded[:,0], fts_cg_embedded[:,1], starting_angle_10e8_cg_embedded,
           '#C28EE8', '#460273', '10e8_cg_embedded')


# In[22]:


#4e10 spont insert top 
def calculate_traj_angles_spont_4e10(input_pdb, dcd_traj, output_name): 
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
        angle_running.append(getStartingAngle_CG_spont(pdb)) 
    #save to csv file 
    file_out = output_name + "angles.csv"
    np.savetxt(file_out, angle_running, delimiter =",",fmt ='% s')
    return print("finished " + file_out) 
#4e10 spont insert bot 
def calculate_traj_angles_spont_4e10_bot(input_pdb, dcd_traj, output_name): 
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
        angle_running.append(getStartingAngle_CG_spont_bot(pdb)) 
    #save to csv file 
    file_out = output_name + "angles.csv"
    np.savetxt(file_out, angle_running, delimiter =",",fmt ='% s')
    return print("finished " + file_out)


# In[23]:


#pgzl1 spont insert top 
def calculate_traj_angles_spont_pgzl1(input_pdb, dcd_traj, output_name): 
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
       angle_running.append(getStartingAngle_CG_PGZL1_spont(pdb)) 
   #save to csv file 
   file_out = output_name + "angles.csv"
   np.savetxt(file_out, angle_running, delimiter =",",fmt ='% s')
   return print("finished " + file_out) 
#pgzl1 spont insert bot 
def calculate_traj_angles_spont_pgzl1_bot(input_pdb, dcd_traj, output_name): 
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
       angle_running.append(getStartingAngle_CG_spont_bot(pdb)) 
   #save to csv file 
   file_out = output_name + "angles.csv"
   np.savetxt(file_out, angle_running, delimiter =",",fmt ='% s')
   return print("finished " + file_out)


# In[24]:


#10e8 spont insert top 
def calculate_traj_angles_spont_10e8(input_pdb, dcd_traj, output_name): 
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
       angle_running.append(getStartingAngle_CG_10E8_spont(pdb)) 
   #save to csv file 
   file_out = output_name + "angles.csv"
   np.savetxt(file_out, angle_running, delimiter =",",fmt ='% s')
   return print("finished " + file_out) 
#10e8 spont insert bot 
def calculate_traj_angles_spont_10e8_bot(input_pdb, dcd_traj, output_name): 
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
       angle_running.append(getStartingAngle_CG_spont_bot(pdb)) 
   #save to csv file 
   file_out = output_name + "angles.csv"
   np.savetxt(file_out, angle_running, delimiter =",",fmt ='% s')
   return print("finished " + file_out)


# In[72]:


#4e10 cg spont. insert 
wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/4e10/'
input_pdb = wd+'analysis_input.pdb'
dcd_traj = wd+'gar_top_contact.dcd'
calculate_traj_angles_spont_4e10(input_pdb, dcd_traj, '4e10_spont_topA')
dcd_traj = wd+'ims_top_contact.dcd'
calculate_traj_angles_spont_4e10(input_pdb, dcd_traj, '4e10_spont_topB')
dcd_traj = wd+'gar_bot_contact.dcd'
calculate_traj_angles_spont_4e10_bot(input_pdb, dcd_traj, '4e10_spont_botA')
dcd_traj = wd+'ims_bot_contact.dcd'
calculate_traj_angles_spont_4e10_bot(input_pdb, dcd_traj, '4e10_spont_botB')


# In[73]:


file = open("4e10_spont_topAangles.csv", "r")
csv_reader = csv.reader(file) 
fts_cg_spont_init = []
counter = 0 
for row in csv_reader:
    fts_cg_spont_init.append(row)
    counter = counter+1
    
fts_cg_spontA =  string_list_to_float(fts_cg_spont_init, 'fts_10e8_ppm_1us_AA_float' )

file = open("4e10_spont_topBangles.csv", "r")
csv_reader = csv.reader(file) 
fts_cg_spont_init = []
counter = 0 
for row in csv_reader:
    fts_cg_spont_init.append(row)
    counter = counter+1
    
fts_cg_spontB =  string_list_to_float(fts_cg_spont_init, 'fts_10e8_ppm_1us_AA_float' )

file = open("4e10_spont_botAangles.csv", "r")
csv_reader = csv.reader(file) 
fts_cg_spont_init = []
counter = 0 
for row in csv_reader:
    fts_cg_spont_init.append(row)
    counter = counter+1
    
fts_cg_spontC =  string_list_to_float(fts_cg_spont_init, 'fts_10e8_ppm_1us_AA_float' )

file = open("4e10_spont_botBangles.csv", "r")
csv_reader = csv.reader(file) 
fts_cg_spont_init = []
counter = 0 
for row in csv_reader:
    fts_cg_spont_init.append(row)
    counter = counter+1

fts_cg_spontD =  string_list_to_float(fts_cg_spont_init, 'fts_10e8_ppm_1us_AA_float' )


fts_cg_spont_4e10 = [] 
for i in fts_cg_spontA:
    fts_cg_spont_4e10.append(i)
for i in fts_cg_spontB:
    fts_cg_spont_4e10.append(i)  
for i in fts_cg_spontC:
    fts_cg_spont_4e10.append(i)
for i in fts_cg_spontD:
    fts_cg_spont_4e10.append(i)
    
fts_cg_spont_4e10 = np.array(fts_cg_spont_4e10)

plot_angles2(fts_cg_spont_4e10[:,0], fts_cg_spont_4e10[:,1], np.array([[150,150]]),
           '#55A3FF', '#2A517F', '4e10_cg_spont_TEMP')


# In[65]:


#pgzl1 cg spont. insert 

wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/pgzl1/'
input_pdb = wd+'analysis_input.pdb'
dcd_traj = wd+'gar_top_contact.dcd'
calculate_traj_angles_spont_pgzl1(input_pdb, dcd_traj, 'pgzl1_spont_topA')
dcd_traj = wd+'ims_top_contact.dcd'
calculate_traj_angles_spont_pgzl1(input_pdb, dcd_traj, 'pgzl1_spont_topB')
dcd_traj = wd+'gar_bot_contact.dcd'
calculate_traj_angles_spont_pgzl1_bot(input_pdb, dcd_traj, 'pgzl1_spont_botA')
dcd_traj = wd+'ims_bot_contact.dcd'
calculate_traj_angles_spont_pgzl1_bot(input_pdb, dcd_traj, 'pgzl1_spont_botB')


# In[71]:


file = open("pgzl1_spont_topAangles.csv", "r")
csv_reader = csv.reader(file) 
fts_cg_spont_init = []
counter = 0 
for row in csv_reader:
    if counter<5033:
        fts_cg_spont_init.append(row)
        counter = counter+1
    
fts_cg_spontA =  string_list_to_float(fts_cg_spont_init, 'fts_10e8_ppm_1us_AA_float' )
file = open("pgzl1_spont_topBangles.csv", "r")
csv_reader = csv.reader(file) 
fts_cg_spont_init = []
counter = 0 
for row in csv_reader:
    if counter<5033:
        fts_cg_spont_init.append(row)
        counter = counter+1
    
fts_cg_spontB =  string_list_to_float(fts_cg_spont_init, 'fts_10e8_ppm_1us_AA_float' )

file = open("pgzl1_spont_botAangles.csv", "r")
csv_reader = csv.reader(file) 
fts_cg_spont_init = []
counter = 0 
for row in csv_reader:
    if counter<5033:
        fts_cg_spont_init.append(row)
        counter = counter+1
    
fts_cg_spontC =  string_list_to_float(fts_cg_spont_init, 'fts_10e8_ppm_1us_AA_float' )

file = open("pgzl1_spont_botBangles.csv", "r")
csv_reader = csv.reader(file) 
fts_cg_spont_init = []
counter = 0 
for row in csv_reader:
    if counter<5033:
        fts_cg_spont_init.append(row)
        counter = counter+1
    
fts_cg_spontD =  string_list_to_float(fts_cg_spont_init, 'fts_10e8_ppm_1us_AA_float' )


fts_cg_spont_pgzl1 = [] 
for i in fts_cg_spontA:
    fts_cg_spont_pgzl1.append(i)
for i in fts_cg_spontB:
    fts_cg_spont_pgzl1.append(i)  
for i in fts_cg_spontC:
    fts_cg_spont_pgzl1.append(i)
for i in fts_cg_spontD:
    fts_cg_spont_pgzl1.append(i)
    
fts_cg_spont_pgzl1 = np.array(fts_cg_spont_pgzl1)

plot_angles2(fts_cg_spont_pgzl1[:,0], fts_cg_spont_pgzl1[:,1], np.array([[150,150]]),
           '#F79A99', '#A33634', 'pgzl1_cg_spont_TOTAL')


# In[67]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/10e8/'
input_pdb = wd+'analysis_input.pdb'
# dcd_traj = wd+'gar_top_contact.dcd'
# calculate_traj_angles_spont_10e8(input_pdb, dcd_traj, '10e8_spont_topA')

dcd_traj = wd+'gar_bot_contact.dcd'
calculate_traj_angles_spont_10e8_bot(input_pdb, dcd_traj, '10e8_spont_botA')


# In[4]:


file = open("10e8_spont_topAangles.csv", "r")
csv_reader = csv.reader(file) 
fts_cg_spont_init = []
counter = 0 
for row in csv_reader:
    fts_cg_spont_init.append(row)
    counter = counter+1
fts_cg_spontA =  string_list_to_float(fts_cg_spont_init, 'fts_10e8_ppm_1us_AA_float' )


file = open("10e8_spont_botAangles.csv", "r")
csv_reader = csv.reader(file) 
fts_cg_spont_init = []
counter = 0 
for row in csv_reader:
    if counter<5033:
        fts_cg_spont_init.append(row)
        counter = counter+1
    
fts_cg_spontC =  string_list_to_float(fts_cg_spont_init, 'fts_10e8_ppm_1us_AA_float' )



fts_cg_spont_10e8 = [] 
for i in fts_cg_spontA:
    fts_cg_spont_10e8.append(i)
# for i in fts_cg_spontB:
#     fts_cg_spont_10e8.append(i)  
for i in fts_cg_spontC:
    fts_cg_spont_10e8.append(i)
# for i in fts_cg_spontD:
#     fts_cg_spont_10e8.append(i)
    
fts_cg_spont_10e8 = np.array(fts_cg_spont_10e8)

plot_angles2(fts_cg_spont_10e8[:,0], fts_cg_spont_10e8[:,1], np.array([[150,150]]),
           '#C28EE8', '#460273', '10e8_cg_spont_TOTAL')


# In[129]:



pdb = parsePDB('/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/cg2aa_revert_10000_dump.pdb')
print("4e10 cg revert aa: ", getStartingAngle_AA(pdb)) 

pdb = parsePDB('/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/10000_dump.pdb')
print("4e10 coarse grain: ", getStartingAngle_CG(pdb)) 


# In[250]:


wd = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_embedded/'
pdb = parsePDB(wd+'10e8_ab/analysis_input.pdb')
fab = pdb.select("name BB")
fab_str = ''
for i in fab:
    fab_str += UnNatAA[i.getResname()]
print(fab_str)    

# fab = pdb.select('name BB') 


# In[149]:


pdb = parsePDB('/Users/cmaillie/Dropbox (Scripps Research)/manuscript/scripts/minx-sys-solvated.pdb')
print("4e10 coarse grain: ", getStartingAngle_CG(pdb)) 


# In[50]:



def getStartingAngle_CG_spont(pdb):
    
    #selection strings 
    mem_selection_str = 'resname POPC POPA CHOL'
    fab_selection_str = 'name BB'
    #read in pdb
    input_pdb = pdb 
    #move pdb to origin 
    moveAtoms(input_pdb, to=np.zeros(3))
    #writePDB('cg_input_2zero.pdb', input_pdb)
    #pre select po4 membrane based on geometric constraints to avoid micelle lipids 
    po4_membrane = input_pdb.select('name PO4 and z>5') #and z > 45 or z < 90 
    #mem_coords = po4_membrane.getCoords()[2]
    #writePDB('testPO4_cg_spont.pdb', po4_membrane)
    
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
    rotation = VectorAlign(psuedo_central_ax_norm, np.array([0, 0, 1]))
    transformCentralAx = Transformation(rotation, np.zeros(3))
    applyTransformation(transformCentralAx, input_pdb)
    #writePDB('cg_transform_fabUp.pdb', input_pdb)
    
    mem_plane = planeFit(np.transpose(po4_membrane.getCoords()))
    mem_plane_normal = mem_plane[1]

    #this doesnt work in CG for some reason - definition of plane norm is wrong direciton
    #CG systems are already aligned so Z calculations are accurate - can skip for CG 
    #define rotation as alignement from psueo-central-axis to z plane norm vect 
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
    writePDB('angle_test_4E10_MOVED.pdb', input_pdb)
    #select 2 fab axes 
    
    #calc angle b/w approach angle & top phos plane
    res94 = fab[94]
    res320 = fab[320]
    point_cdr_loops = calcCenter(res94+res320)
    res41 = fab[41]
    res256 = fab[256]
    point_center_fab = calcCenter(res41+res256)     
    point_xy_3 = np.array([point_center_fab[0], point_center_fab[1], -50]) 
    var_domain_vect = np.array(point_cdr_loops-point_center_fab)
    xy_norm_vect = np.array(point_xy_3-point_center_fab)
    angle_var = round(angle(var_domain_vect, xy_norm_vect), 2)
#     #angle of variable domain to membrane in xy space 

    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[107].getCoords()
    point_h_edge = fab[340].getCoords()
    point_xz_3 = np.array([point_l_edge[0], point_l_edge[1], -50])
    short_ax_vect = np.array(point_l_edge-point_h_edge)
    xz_norm_vect = np.array(point_xz_3-point_l_edge)
    angle_short = round(angle(short_ax_vect, xz_norm_vect), 2) 
    
    wtr1 = AtomGroup('Water') 
    coords = np.array([point_cdr_loops, 
                       point_center_fab,
                       point_xy_3, 
                       point_l_edge, 
                       point_h_edge,
                       point_xz_3],
                       dtype=float)
    wtr1.setCoords(coords) 
    wtr1.setNames(['O', 'O', 'O', 'P', 'P', 'P', ])
    wtr1.setResnums([1, 1, 1, 1, 1, 1])
    wtr1.setResnames(['WAT', 'WAT', 'WAT', 'WAT', 'WAT', 'WAT'])
    writePDB('angle_test_4E10_POINTS.pdb', wtr1)
    
    
    #select fab axes 
    return [angle_var, angle_short] 


# In[26]:


def getStartingAngle_CG_PGZL1_spont(pdb):
    
    #selection strings 
    mem_selection_str = 'resname POPC POPA CHOL'
    fab_selection_str = 'name BB'
    #read in pdb
    input_pdb = pdb 
    #move pdb to origin 
    moveAtoms(input_pdb, to=np.zeros(3))
    #writePDB('cg_input_2zero.pdb', input_pdb)
    #pre select po4 membrane based on geometric constraints to avoid micelle lipids 
    po4_membrane = input_pdb.select('name PO4 and z>5') #and z > 45 or z < 90 
    
    
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
    rotation = VectorAlign(psuedo_central_ax_norm, np.array([0, 0, 1]))
    transformCentralAx = Transformation(rotation, np.zeros(3))
    applyTransformation(transformCentralAx, input_pdb)
    #writePDB('cg_transform_fabUp.pdb', input_pdb)
    
    mem_plane = planeFit(np.transpose(po4_membrane.getCoords()))
    mem_plane_normal = mem_plane[1]

    #this doesnt work in CG for some reason - definition of plane norm is wrong direciton
    #CG systems are already aligned so Z calculations are accurate - can skip for CG 
    #define rotation as alignement from psueo-central-axis to z plane norm vect 
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
    writePDB('angle_test_PGZL1_MOVED.pdb', input_pdb)
    #select 2 fab axes 
    
    #calc angle b/w approach angle & top phos plane
    #numbering of hcains is swapped b/c H,L order in Cg system- this is accurate 10.13.21
    res317 = fab[317]
    res106 = fab[106]
    point_cdr_loops = calcCenter(res317+res106)
    res264 = fab[264]
    res42 = fab[42]
    point_center_fab = calcCenter(res42+res264)     
    point_xy_3 = np.array([point_center_fab[0], point_center_fab[1], -50]) 
    var_domain_vect = np.array(point_cdr_loops-point_center_fab)
    xy_norm_vect = np.array(point_xy_3-point_center_fab)
    angle_var = round(angle(var_domain_vect, xy_norm_vect), 2)
#     #angle of variable domain to membrane in xy space 

    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[330].getCoords()
    point_h_edge = fab[126].getCoords()
    point_xz_3 = np.array([point_l_edge[0], point_l_edge[1], -50])
    short_ax_vect = np.array(point_l_edge-point_h_edge)
    xz_norm_vect = np.array(point_xz_3-point_l_edge)
    angle_short = round(angle(short_ax_vect, xz_norm_vect), 2) 
    
    wtr1 = AtomGroup('Water') 
    coords = np.array([point_cdr_loops, 
                       point_center_fab,
                       point_xy_3, 
                       point_l_edge, 
                       point_h_edge,
                       point_xz_3],
                       dtype=float)
    wtr1.setCoords(coords) 
    wtr1.setNames(['O', 'O', 'O', 'P', 'P', 'P', ])
    wtr1.setResnums([1, 1, 1, 1, 1, 1])
    wtr1.setResnames(['WAT', 'WAT', 'WAT', 'WAT', 'WAT', 'WAT'])
    writePDB('angle_test_PGZL1_POINTS.pdb', wtr1)
    
    #select fab axes 
    return [angle_var, angle_short] 


# In[27]:



def getStartingAngle_CG_10E8_spont(pdb):
    
    #selection strings 
    mem_selection_str = 'resname POPC POPA CHOL'
    fab_selection_str = 'name BB'
    #read in pdb
    input_pdb = pdb 
    #move pdb to origin 
    moveAtoms(input_pdb, to=np.zeros(3))
    #writePDB('cg_input_2zero.pdb', input_pdb)
    #pre select po4 membrane based on geometric constraints to avoid micelle lipids 
    po4_membrane = input_pdb.select('name PO4 and z>5') #and z > 45 or z < 90 
    
    
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
    rotation = VectorAlign(psuedo_central_ax_norm, np.array([0, 0, 1]))
    transformCentralAx = Transformation(rotation, np.zeros(3))
    applyTransformation(transformCentralAx, input_pdb)
    #writePDB('cg_transform_fabUp.pdb', input_pdb)
    
    mem_plane = planeFit(np.transpose(po4_membrane.getCoords()))
    mem_plane_normal = mem_plane[1]

    #this doesnt work in CG for some reason - definition of plane norm is wrong direciton
    #CG systems are already aligned so Z calculations are accurate - can skip for CG 
    #define rotation as alignement from psueo-central-axis to z plane norm vect 
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
    #writePDB('cg_transform_z2zero_10E8.pdb', input_pdb)
    #select 2 fab axes 
    writePDB('angle_test_10E8_MOVED.pdb', input_pdb)
    #calc angle b/w approach angle & top phos plane
    res110 = fab[110]
    res324 = fab[324]
    point_cdr_loops = calcCenter(res110+res324)
    res41 = fab[41]
    res272 = fab[272]
    point_center_fab = calcCenter(res41+res272)     
    point_xy_3 = np.array([point_center_fab[0], point_center_fab[1], -50]) 
    var_domain_vect = np.array(point_cdr_loops-point_center_fab)
    xy_norm_vect = np.array(point_xy_3-point_center_fab)
    angle_var = round(angle(var_domain_vect, xy_norm_vect), 2)
#     #angle of variable domain to membrane in xy space 

    #calculate angels between rotational vect & top phos plane 
    point_l_edge = fab[130].getCoords()
    point_h_edge = fab[339].getCoords()
    point_xz_3 = np.array([point_l_edge[0], point_l_edge[1], -50])
    #swapped angle calc b/c chain order is swapped in 10e8 (H,L)
    short_ax_vect = np.array(point_h_edge-point_l_edge)
    xz_norm_vect = np.array(point_xz_3-point_l_edge)
    angle_short = round(angle(short_ax_vect, xz_norm_vect), 2) 
    
    wtr1 = AtomGroup('Water') 
    coords = np.array([point_cdr_loops, 
                       point_center_fab,
                       point_xy_3, 
                       point_l_edge, 
                       point_h_edge,
                       point_xz_3],
                       dtype=float)
    wtr1.setCoords(coords) 
    wtr1.setNames(['O', 'O', 'O', 'P', 'P', 'P', ])
    wtr1.setResnums([1, 1, 1, 1, 1, 1])
    wtr1.setResnames(['WAT', 'WAT', 'WAT', 'WAT', 'WAT', 'WAT'])
    writePDB('angle_test_10E8_POINTS.pdb', wtr1)
    
    #select fab axes 
    return [angle_var, angle_short] 


# In[331]:


#to check angle calculations for each system - ref points are correct & angl values match 


#4e10 cg spont 
pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/4e10/angle_test.pdb'
pdb_in = parsePDB(pdb)
print(getStartingAngle_CG_spont(pdb_in))
#pgzl1 cg spont 
pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/pgzl1/angle_test.pdb'
pdb_in = parsePDB(pdb)
print(getStartingAngle_CG_PGZL1_spont(pdb_in))
#10e8 cg spont 
pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/10e8/angle_test.pdb'
pdb_in = parsePDB(pdb)
print(getStartingAngle_CG_10E8_spont(pdb_in))


# In[52]:


#angle check for CG reversion 
pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/4e10/medoid1.pdb'
pdb_in = parsePDB(pdb)
print(getStartingAngle_CG_spont(pdb_in))


# In[32]:


#angle check for CG reversion 
pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/pgzl1/medoid5.pdb'
pdb_in = parsePDB(pdb)
print(getStartingAngle_CG_PGZL1_spont(pdb_in))


# In[28]:



pdb = '/Users/cmaillie/Dropbox (Scripps Research)/mravic_lab/membrane_antibodies/cg_reversions/test10.28.21/gromacs/step5_input.pdb'
pdb_in = parsePDB(pdb)
print(getStartingAngle_AA(pdb_in))


# In[70]:


#4e10 extracted medoid angle checks 
#4e10 medoid 1 D: 256 [ 57.85 -15.19] ([57.91, -15.13])
# 4e10 medoid 2 A: 44 [ 37.43 -21.3 ] ([37.43, -21.3])
# 4e10 medoid 3 B: 976 [26.53 2.29] ([26.53, 2.29])

pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/4e10/medoid3_final.pdb'
pdb_in = parsePDB(pdb)
print(getStartingAngle_CG_spont(pdb_in))


# In[63]:





# In[69]:


#pgzl1 extracted medoid angle checks 
# pgzl1 medoid 1 A: 1277, [49.56 -8.65] ([49.56, -8.65])
# pgzl1 medoid 2 B: 808, [18.05 40.59] ([18.05, 40.59])
# pgzl1 medoid 3 D: 32, [-13.64 39.22] ()


# pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/pgzl1/medoid2_final.pdb'
# pdb_in = parsePDB(pdb)
# print(getStartingAngle_CG_PGZL1_spont(pdb_in))

pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/pgzl1/medoid3_final.pdb'
pdb_in = parsePDB(pdb)
print(getStartingAngle_CG_PGZL1_spont_bot_V2(pdb_in)) 


# In[ ]:





# In[61]:


# 10e8 medoid 1 A: 2469, [63.23 30.93] [63.23, 30.93]
# 10e8 medoid 2 C: 2058 [-7.39 34. ] [-7.39, 34.0]
# 10e8 medoid 3 A: 1064, [ 11.65 -50. ] [11.65, -50.0]


pdb = '/Users/cmaillie/Dropbox (Scripps Research)/colleen-marco/mem_ab_manuscript/trajectories/cg_spont/10e8/medoid2_final.pdb'
pdb_in = parsePDB(pdb)
print(getStartingAngle_CG_spont_bot(pdb_in)) -- this should be used for 10e8?? 

