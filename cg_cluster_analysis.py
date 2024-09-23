#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[15]:


#DEFINE FUNCTIONS


def findMembraneContacts(fab, membrane):
    fab_contacts = Contacts(fab)
    contacts = fab_contacts.select(4, membrane)
    #print(repr(contacts))
    return(contacts)

#from MM 
def planeFit(points):
	points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
	assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
	ctr = points.mean(axis=1)
	x = points - ctr[:,np.newaxis]
	M = np.dot(x, x.T) # Could also use np.cov(x) here.
	return ctr, svd(M)[0][:,-1]

#from MM 
def dist2Plane(points, planeParams):
	cen, norm = planeParams[0], planeParams[1]
	d = np.dot(norm,cen)
	return ( np.dot(norm,points) - d )# / np.linalg.norm(cen)



def getUnitVector(point1, point2): 
    v = np.subtract(point1, point2)
    mag = np.linalg.norm(point1-point2) 
    v_hat = v/mag 
    return v_hat

def getAngle(p1, p2, p3): 
    p2p1 = p1 - p2
    p2p3 = p3 - p2
    cosine_angle = np.dot(p2p1, p2p3) / (np.linalg.norm(p2p1) * np.linalg.norm(p2p3))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def calcRMS(x, axis=None):
    return np.sqrt(np.nanmean(x**2, axis=axis))


# In[50]:


#DEFINE FUNCTIONS

#from MM 
def planeFit(points):
	points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
	assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
	ctr = points.mean(axis=1)
	x = points - ctr[:,np.newaxis]
	M = np.dot(x, x.T) # Could also use np.cov(x) here.
	return ctr, svd(M)[0][:,-1]

#from MM 
def dist2Plane(points, planeParams):
	cen, norm = planeParams[0], planeParams[1]
	d = np.dot(norm,cen)
	return ( np.dot(norm,points) - d )# / np.linalg.norm(cen)


def getUnitVector(point1, point2): 
    v = np.subtract(point1, point2)
    mag = np.linalg.norm(point1-point2) 
    v_hat = v/mag 
    return v_hat

# def getAngle(p1, p2, p3): 
#     p2p1 = p1 - p2
#     p2p3 = p3 - p2
#     cosine_angle = np.dot(p2p1, p2p3) / (np.linalg.norm(p2p1) * np.linalg.norm(p2p3))
#     angle = np.arccos(cosine_angle)
#     return np.degrees(angle)


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

#determine binary list for if a residue is close enough for contact with a plane
def findMemContacts(res_coords, plane): 
    #1 residue index X from res_coords list contacts membrane
    #0 residue index X from res_coords list does not contact membrane 
    contacts = []
    for res in res_coords:
        if dist2Plane(res, plane)>=-4: #written for upside down system (fab is embedded in bottom membrane)
                contacts.append(1)
        else:
            contacts.append(0)      
    return contacts

def countMemContacts(res_coords, plane): 
    #determine how many reisudes in each cdr loop contact membrane 

    cdrl1_contact_count = 0
    cdrl2_contact_count = 0 
    cdrl3_contact_count = 0 
    cdrh1_contact_count = 0 
    cdrh2_contact_count = 0 
    cdrh3_contact_count = 0
    
    contacts = []
    for i in range(len(res_coords)):
        #residues within CDRL1 are in first 5 indexes of res_coords list 
        if i <= 4: 
            res = res_coords[i]
            if dist2Plane(res, plane)>=-2: #written for upside down system (fab is embedded in bottom membrane)
                cdrl1_contact_count = cdrl1_contact_count+1
        elif i<=11: 
            res = res_coords[i]
            if dist2Plane(res, plane)>=-2: 
                cdrl2_contact_count = cdrl2_contact_count+1          
        elif i<=17: 
            res = res_coords[i]
            if dist2Plane(res, plane)>=-2: 
                cdrl3_contact_count = cdrl3_contact_count+1    
        elif i<=23: 
            res = res_coords[i]
            if dist2Plane(res, plane)>=-2: 
                cdrh1_contact_count = cdrh1_contact_count+1          
        elif i<=27: 
            res = res_coords[i]
            if dist2Plane(res, plane)>=-2: 
                cdrh2_contact_count = cdrh2_contact_count+1 
        elif i<=36: 
            res = res_coords[i]
            if dist2Plane(res, plane)>=-2: 
                cdrh3_contact_count = cdrh3_contact_count+1          

    return [cdrl1_contact_count, cdrl2_contact_count, cdrl3_contact_count, cdrh1_contact_count, cdrh2_contact_count, cdrh3_contact_count ] 
def countMemContactsCG(res_coords, plane): 
    #determine how many reisudes in each cdr loop contact membrane 

    cdrl1_contact_count = 0
    cdrl2_contact_count = 0 
    cdrl3_contact_count = 0 
    cdrh1_contact_count = 0 
    cdrh2_contact_count = 0 
    cdrh3_contact_count = 0
    
    contacts = []
    for i in range(len(res_coords)):
        #residues within CDRL1 are in first 5 indexes of res_coords list 
        if i <= 4: 
            res = res_coords[i]
            if dist2Plane(res, plane)<=2: #written for rightside up system (fab is embedded in top membrane)
                cdrl1_contact_count = cdrl1_contact_count+1
        elif i<=11: 
            res = res_coords[i]
            if dist2Plane(res, plane)<=2: 
                cdrl2_contact_count = cdrl2_contact_count+1          
        elif i<=17: 
            res = res_coords[i]
            if dist2Plane(res, plane)<=2: 
                cdrl3_contact_count = cdrl3_contact_count+1    
        elif i<=23: 
            res = res_coords[i]
            if dist2Plane(res, plane)<=2: 
                cdrh1_contact_count = cdrh1_contact_count+1          
        elif i<=27: 
            res = res_coords[i]
            if dist2Plane(res, plane)<=2: 
                cdrh2_contact_count = cdrh2_contact_count+1 
        elif i<=36: 
            res = res_coords[i]
            if dist2Plane(res, plane)<=2: 
                cdrh3_contact_count = cdrh3_contact_count+1          

    return [cdrl1_contact_count, cdrl2_contact_count, cdrl3_contact_count, cdrh1_contact_count, cdrh2_contact_count, cdrh3_contact_count ] 

def countMemContacts108e(res_coords, plane): 
    #determine how many reisudes in each cdr loop contact membrane 

    cdrl1_contact_count = 0
    cdrl2_contact_count = 0 
    cdrl3_contact_count = 0 
    cdrh1_contact_count = 0 
    cdrh2_contact_count = 0 
    cdrh3_contact_count = 0
    
    contacts = []
    for i in range(len(res_coords)):
        #residues within CDRL1 are in first 5 indexes of res_coords list 
        if i <= 4: 
            res = res_coords[i]
            if dist2Plane(res, plane)>=-2: #written for upside down system (fab is embedded in bottom membrane)
                cdrl1_contact_count = cdrl1_contact_count+1
        elif i<=12: 
            res = res_coords[i]
            if dist2Plane(res, plane)>=-2: 
                cdrl2_contact_count = cdrl2_contact_count+1          
        elif i<=24: 
            res = res_coords[i]
            if dist2Plane(res, plane)>=-2: 
                cdrl3_contact_count = cdrl3_contact_count+1    
        elif i<=29: 
            res = res_coords[i]
            if dist2Plane(res, plane)>=-2: 
                cdrh1_contact_count = cdrh1_contact_count+1          
        elif i<=35: 
            res = res_coords[i]
            if dist2Plane(res, plane)>=-2: 
                cdrh2_contact_count = cdrh2_contact_count+1 
        elif i<=42: 
            res = res_coords[i]
            if dist2Plane(res, plane)>=-2: 
                cdrh3_contact_count = cdrh3_contact_count+1          

    return [cdrl1_contact_count, cdrl2_contact_count, cdrl3_contact_count, cdrh1_contact_count, cdrh2_contact_count, cdrh3_contact_count ] 

def midpoint(p1, p2):
    x_mp = (p1[0]+p2[0])/2
    y_mp = (p1[1]+p2[1])/2
    z_mp = (p1[2]+p2[2])/2
    return np.array(x_mp, y_mp, z_mp)  
                    
                    
def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
     return 90-(np.degrees(math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))))


def py_ang(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'   """
    cosang = np.dot(v1, v2)
    sinang = LA.norm(np.cross(v1, v2))
    return 90-np.degrees(np.arctan2(sinang, cosang))


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


# In[69]:


#CG BOUND STATE CELL 1 
fab_selection_str = 'name BB SC1 SC2 SC3 SC4'
mem_selection_str = 'resname POPC POPA CHOL'


input_pdb = parsePDB('/home/bat-gpu/colleen/phos_interactions/4e10_cg/analysis_input.pdb')
dcd = DCDFile('/home/bat-gpu/colleen/phos_interactions/4e10_cg/membrane_contact_frames.dcd')
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

#will be X frame length of ft vector 
#ft vector will contain 15 fts (in following order):
#cdrl1_loop_contact_counts
#cdrl2_loop_contact_counts
#cdrl3_loop_contact_counts
#cdrh1_loop_contact_counts
#cdrh2_loop_contact_counts
#cdrh1_loop_contact_counts
#cdrl1_loop_COM_depth
#cdrl2_loop_COM_depth
#cdrl3_loop_COM_depth
#cdrh1_loop_COM_depth
#cdrh2_loop_COM_depth
#cdrh3_loop_COM_depth
#full_fab_long_ax_ang 
#var_dom_long_ax_ang
#short_ax_ang

fts_4e10_cg = [] 

################ LOOP THROUGH TRAJ TO CALC FEATURE VECTORS ########################   
for i, frame in enumerate(dcd):
    if i<1500: 
        ft_vect = []

        #define fab and membrane 
        frame_fab = frame.getAtoms().select( fab_selection_str )
        frame_fab_bb = frame.getAtoms().select( 'name BB' )
        frame_mem = frame.getAtoms().select( mem_selection_str )

        #DEFINE MEMBRANE LAYERS - bilayer midpoint, top PO4 plane, bottom PO4 plane 

        avg_bilayer_mp = (int(sum(list(frame_mem.getCoords()[:,2]))/len(frame_mem.getResnames())))

        mem_top_sel_str  = 'resname POPC name PO4 and z > '+ str(avg_bilayer_mp)
        mem_bot_sel_str  = 'resname POPC name PO4 and z < '+ str(avg_bilayer_mp)

        top_leaf = frame.getAtoms().select(mem_top_sel_str)
        top_leaf_points = top_leaf.getCoords()
        top_leaf_plane = planeFit(np.transpose(top_leaf_points))


        bot_leaf = frame.getAtoms().select(mem_bot_sel_str)
        bot_leaf_points = bot_leaf.getCoords()
        bot_leaf_plane = planeFit(np.transpose(bot_leaf_points))


        bot_phos_layer_sel_str  = 'resname POPC name PO4 and z < ' +str(avg_bilayer_mp)
        phos = frame.getAtoms().select(bot_phos_layer_sel_str)
        phos_points = phos.getCoords()
        phos_z = phos_points[:,2]
        avg_bot_phos_z = sum(phos_z)/len(phos_z)
        bot_phos_layer_dist2bilayer_cen = avg_bot_phos_z - avg_bilayer_mp


        top_phos_layer_sel_str  = 'resname POPC name PO4 and z > ' +str(avg_bilayer_mp)
        phos = frame.getAtoms().select(top_phos_layer_sel_str)
        phos_points = phos.getCoords()
        phos_z = phos_points[:,2]
        avg_top_phos_z = sum(phos_z)/len(phos_z)
        top_phos_layer_dist2bilayer_cen = avg_top_phos_z - avg_bilayer_mp


    #     avg_bot_phos_dist2bilayer_cen.append(bot_phos_layer_dist2bilayer_cen)
    #     avg_top_phos_dist2bilayer_cen.append(top_phos_layer_dist2bilayer_cen)
    #     bilayer_cen_dist2bilayer_cen.append(avg_bilayer_mp-avg_bilayer_mp)


        ################ CDR LOOP RESIDUES - CONTACT MEMBRANE STATS########################   

        #CDRL1 loops residues 
        ser_027 = frame_fab_bb[27].getCoords()
        val_028 = frame_fab_bb[28].getCoords()  
        gly_029 = frame_fab_bb[29].getCoords()   
        asn_030 = frame_fab_bb[30].getCoords()
        asn_031 = frame_fab_bb[31].getCoords()

        #CDRL2 loops residues 
        tyr_049 = frame_fab_bb[49].getCoords()
        gly_050 = frame_fab_bb[50].getCoords()
        ala_051 = frame_fab_bb[51].getCoords()
        ser_052 = frame_fab_bb[52].getCoords()
        ser_053 = frame_fab_bb[53].getCoords()
        arg_054 = frame_fab_bb[54].getCoords()
        pro_055 = frame_fab_bb[55].getCoords()

        #CDRL3 loops residues   
        tyr_091 = frame_fab_bb[91].getCoords()
        gly_092 = frame_fab_bb[92].getCoords()
        gln_093 = frame_fab_bb[93].getCoords()
        ser_094 = frame_fab_bb[94].getCoords()
        leu_095 = frame_fab_bb[95].getCoords()
        ser_096 = frame_fab_bb[96].getCoords()

        #CDRH1 loop residues 
        gly_239 = frame_fab_bb[239].getCoords()
        gly_240 = frame_fab_bb[240].getCoords()
        ser_241 = frame_fab_bb[241].getCoords()  
        phe_242 = frame_fab_bb[242].getCoords()
        ser_243 = frame_fab_bb[243].getCoords()
        thr_244 = frame_fab_bb[244].getCoords()
        tyr_245 = frame_fab_bb[245].getCoords()


        #CDRH2 loop residues 
        pro_266 = frame_fab_bb[266].getCoords()
        leu_267 = frame_fab_bb[267].getCoords()
        leu_268 = frame_fab_bb[268].getCoords()
        thr_269 = frame_fab_bb[269].getCoords()

        #CDRH3 loop residues 
        gly_316 = frame_fab_bb[316].getCoords()
        trp_317 = frame_fab_bb[317].getCoords() 
        gly_318 = frame_fab_bb[318].getCoords()
        trp_319 = frame_fab_bb[319].getCoords()  
        leu_320 = frame_fab_bb[320].getCoords() 
        gly_321 = frame_fab_bb[321].getCoords()
        lys_322 = frame_fab_bb[322].getCoords()
        pro_323 = frame_fab_bb[323].getCoords()
        ile_324 = frame_fab_bb[324].getCoords()

        #list of crd loop residue coords 
        cdr_loop_res_coords = [ser_027, val_028, gly_029, asn_030, asn_031, 
                               tyr_049, gly_050, ala_051 ,ser_052, ser_053, arg_054, pro_055, 
                               tyr_091, gly_092,  gln_093, ser_094, leu_095, ser_096,
                               gly_239, gly_240, ser_241, phe_242, ser_243, tyr_245, 
                               pro_266, leu_267, leu_268, thr_269,
                               gly_316, trp_317, gly_318, trp_319, leu_320, gly_321, lys_322, pro_323, ile_324]
        #get binary vector of crd loop membrane contacts - do not add to ft vect for clustering currently
        loop_contacts = findMemContacts(cdr_loop_res_coords, top_leaf_plane)

        loop_contact_counts = countMemContactsCG(cdr_loop_res_coords, top_leaf_plane)
        #print(loop_contact_counts)
        #add each loops' residue contact count to ft vect as a value 
        ft_vect.append(loop_contact_counts[0])
        ft_vect.append(loop_contact_counts[1])
        ft_vect.append(loop_contact_counts[2])
        ft_vect.append(loop_contact_counts[3])
        ft_vect.append(loop_contact_counts[4])
        ft_vect.append(loop_contact_counts[5])

    ################ CDR LOOP CENTER OF MASS (COM) DEPTHS - relative to PO4 plane ########################   
            #*remember- actual simulation is upside down, these calculations will be plotted so that Fab is on top of bilayer 
        #CDRL1 
        cdrl1_resis = frame_fab_bb[27:34] #('resnum 27 to 33 ')
        #print(repr(cdrl1_resis))
        cdrl1_com_z = calcCenter(cdrl1_resis)[2]
        cdrl1_loop_COM_depth = avg_top_phos_z-cdrl1_com_z
        #cdrl1_com_dist2po4_bot.append(avg_bot_phos_z-cdrl1_com_z)
        #cdrl1_com_dist2bilayer_cen.append(cdrl1_com_z-avg_bilayer_mp)

        #cdrl1_com_depth_aggregate.append(cdrl1_com[2])

        #CDRL2
        cdrl2_resis = frame_fab_bb[49:57] #.select('resnum 49 to 56')
        cdrl2_com_z = calcCenter(cdrl2_resis)[2]
        cdrl2_loop_COM_depth = avg_top_phos_z-cdrl2_com_z
        #cdrl2_com_dist2po4_bot.append(avg_bot_phos_z-cdrl2_com_z)

        #cdrl2_com_dist2bilayer_cen.append(cdrl2_com_z-avg_bilayer_mp)

        #CDRL3
        cdrl3_resis = frame_fab_bb[91:99] #.select('resnum 91 to 98')
        cdrl3_com_z = calcCenter(cdrl3_resis)[2]
        cdrl3_loop_COM_depth = avg_top_phos_z-cdrl3_com_z
        #cdrl3_com_dist2po4_bot.append(avg_bot_phos_z-cdrl3_com_z)

        #cdrl3_com_dist2bilayer_cen.append(avg_bot_phos_z-cdrl3_com_z)

        #CDRH1
        cdrh1_resis = frame_fab_bb[240:247] #.select('resnum 240 to 246')
        cdrh1_com_z = calcCenter(cdrh1_resis)[2]
        cdrh1_loop_COM_depth = avg_top_phos_z-cdrh1_com_z
        #cdrh1_com_dist2po4_bot.append(avg_bot_phos_z-cdrh1_com_z)
        #cdrh1_com_dist2bilayer_cen.append(avg_bot_phos_z-cdrh1_com_z)

        #CDRH2 resdius 
        cdrh2_resis = frame_fab_bb[267:271] #.select('resnum 267 to 270')
        cdrh2_com_z = calcCenter(cdrh2_resis)[2]
        cdrh2_loop_COM_depth = avg_top_phos_z-cdrh2_com_z
        #cdrh2_com_dist2po4_bot.append(avg_bot_phos_z-cdrh2_com_z)
        #cdrh2_com_dist2bilayer_cen.append(avg_bot_phos_z-cdrh2_com_z)

        #CDRH3 
        cdrh3_resis = frame_fab_bb[315:325] #.select('resnum 315 to 325')
        cdrh3_com_z = calcCenter(cdrh2_resis)[2]
        cdrh3_loop_COM_depth = avg_top_phos_z-cdrh3_com_z
        #cdrh3_com_dist2po4_bot.append(avg_bot_phos_z-cdrh3_com_z)
        #cdrh3_com_dist2bilayer_cen.append(cdrh3_com_z-avg_bilayer_mp)

        ft_vect.append(round(cdrl1_loop_COM_depth*-1, 2))
        ft_vect.append(round(cdrl2_loop_COM_depth*-1, 2))
        ft_vect.append(round(cdrl3_loop_COM_depth*-1, 2))
        ft_vect.append(round(cdrh1_loop_COM_depth*-1, 2))
        ft_vect.append(round(cdrh2_loop_COM_depth*-1, 2))
        ft_vect.append(round(cdrh3_loop_COM_depth*-1, 2))
    ################ ALING NORM OF PO4 PLANE TO Z, calculate ANGLES ########################   

        #define as rotation b/w norm vect of PO4 plane & z-axis (unit normal of Z axis )
        rotation = VectorAlign(top_leaf_plane[1], np.array([0, 0, 1]))
        #define transformation b/w roatino and origin 
        transformZax = Transformation(rotation, np.zeros(3))
        #apply transofrmation of plane to Z axis to entire system 
        applyTransformation(transformZax, frame.getAtoms())

        frame_fab = frame.getAtoms().select( fab_selection_str )

        #define selections for points used to make vectors through fab 

        #center of loops is p2 for all angles 
        res41 = frame_fab_bb[41]
        res256 = frame_fab_bb[256]
        cen_loops = res41+res256 #frame_fab_bb.select('resnum 41 or resnum 256')
        fab_cen = calcCenter(cen_loops)
        
        res243 = frame_fab_bb[243]
        res269 = frame_fab_bb[269]
        res319 = frame_fab_bb[319]
        res31 = frame_fab_bb[31]
        res53 = frame_fab_bb[53]
        res95 = frame_fab_bb[95]
        
        cdrh_points = res243+res269+res319 #frame_fab_bb.select('resnum 243 or resnum 269 or resnum 319')
        cdrl_points = res31+res53+res95 #frame_fab_bb.select('resnum 31 or resnum 53 or resnum 95')
        crd_loops = cdrh_points + cdrl_points    
        cdr_loops_cen = calcCenter(crd_loops)
        
        res213 = frame_fab_bb[213]
        res41 = frame_fab_bb[41]
        fab_tip = res213+res41 #frame_fab_bb.select('resnum 213 or resnum 41')
        fab_tip_cen = calcCenter(fab_tip)

        po4_plan_norm = -1*top_leaf_plane[1] #second vector is norm of PO4 plane 

        #var_dom_long_axis 
        #define vector through center of fab and center of cdr loops 
        var_dom_long_ax =  np.array(fab_cen-cdr_loops_cen)
        var_dom_long_ax_ang = angle(po4_plan_norm, var_dom_long_ax)

        #print('VAR DOM ANG: ', var_dom_long_ax_ang)
        #print('VAR DOM ANG: ',angle(po4_plan_norm, var_dom_long_ax))
        #full_fab_long_axis 
        #full_fab_long_ax =  np.array(fab_tip_cen - cdr_loops_cen)
        full_fab_long_ax =  np.array(fab_tip_cen - cdr_loops_cen)

        full_fab_long_ax_ang = angle(po4_plan_norm, full_fab_long_ax)

        #short_axis 
        res107 = frame_fab_bb[107]
        res340 = frame_fab_bb[340]
        l_chain_middle_edge = res107.getCoords() #frame_fab_bb.select('resnum 107').getCoords()
        h_chain_middle_edge = res340.getCoords() #frame_fab_bb.select('resnum 340').getCoords()
        short_axis_vect = np.array( h_chain_middle_edge - l_chain_middle_edge )
        #short_axis_vect = np.array( l_chain_middle_edge[0] -h_chain_middle_edge[0] ) - this should give opposite sign angle as vector above 
        short_ax_ang = angle(po4_plan_norm, short_axis_vect)

        #add angles to ft vector 
        #ft_vect.append(full_fab_long_ax_ang)
        ft_vect.append(var_dom_long_ax_ang)
        ft_vect.append(short_ax_ang)

        #add ft vect to aggregate lsit and to traj specific list 
        #aggregate_fts.append(ft_vect)
        fts_4e10_cg.append(ft_vect)


# In[70]:


#cluster aggregate data 
import numpy as np
init_T = time.time()
ssd_rmsd_cg = []
#print(type(aggregate_fts))
for pairwise in combinations((fts_4e10_cg), 2):
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    #print(rms)
    ssd_rmsd_cg.append(rms)
print (round( time.time() - init_T, 3), 's to complete rmsd calculation')
print ('mean distance:', round( np.mean(ssd_rmsd_cg), 1), '| StDev: ', round( np.std(ssd_rmsd_cg) , 1))


# In[71]:


cutoff =  120 # #np.mean(ssd_rmsd)*1.5 3.5 = 11 clusters 
ssd_rmsd_cg = np.array( ssd_rmsd_cg )
init_T= time.time()

# Ward hierarchical clustering minimizes variance between clusters
# Complete linkage clustering makes sure all cluster members are within same RMSD cutoff to each other
linkMat= linkage( ssd_rmsd_cg , method='ward', metric='euclidean')
print (round( time.time() - init_T, 3), 's to complete clustering')

h_clust_cg= fcluster( linkMat, cutoff, criterion='distance')
numClust= len( set(h_clust_cg) )
print ('RMS cutoff at %.2f, Unique clusters found:' % cutoff, numClust, '\n')


# In[25]:


f = open('CG_aggregate_data_ssd_rmsd.txt', 'w')
for i in ssd_rmsd:
    f.write(str(i) + "\n")
f.close 


# In[72]:


#calculate centroids for each cluster - Coarse Grain 
clust_01 = []
clust_01_frames = []
clust_02 = []
clust_02_frames = []
clust_03 = []
clust_03_frames = []
clust_04 = []
clust_04_frames = []
clust_05 = []
clust_05_frames = []
clust_06 = []
clust_06_frames = []
clust_07 = []
clust_07_frames = []
clust_08 = []
clust_08_frames = []
clust_09 = []
clust_09_frames = []
for i in range(len(fts_4e10_cg)):
    if h_clust_cg[i] == 1:
        clust_01_frames.append(i)
        clust_01.append(fts_4e10_cg[i])
    elif h_clust_cg[i] == 2:
        clust_02_frames.append(i)
        clust_02.append(fts_4e10_cg[i])
    elif h_clust_cg[i] == 3:
        clust_03_frames.append(i)
        clust_03.append(fts_4e10_cg[i])
    elif h_clust_cg[i] == 4:
        clust_04_frames.append(i)
        clust_04.append(fts_4e10_cg[i])
    elif h_clust_cg[i] == 5:
        clust_05_frames.append(i)
        clust_05.append(fts_4e10_cg[i])
    elif h_clust_cg[i] == 6:
        clust_06_frames.append(i)
        clust_06.append(fts_4e10_cg[i])
    elif h_clust_cg[i] == 7:
        clust_07_frames.append(i)
        clust_07.append(fts_4e10_cg[i])
    elif h_clust_cg[i] == 8:
        clust_08_frames.append(i)
        clust_08.append(fts_4e10_cg[i])
    elif h_clust_cg[i] == 9:
        clust_09_frames.append(i)
        clust_09.append(fts_4e10_cg[i])


# re-do kmedoids clustering to find cluster, including re-calculating dist matrix bc in isnt too long
ssd_rmsd_cg_1 = []
for n, pair in enumerate( combinations( clust_01, 2 ) ):
    #superpose( *pair )
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_rmsd_cg_1.append(rms)
ssd_rmsd_cg_1 = np.array( ssd_rmsd_cg_1 )
k_clust, cost,num	= kmedoids( ssd_rmsd_cg_1, nclusters=1, npass=100)
centroid_ID = list(set(k_clust))[0]
print("centroid ID: " + str(centroid_ID))
print('representative frame: ' + str(clust_01_frames[centroid_ID]))
print("cluster state: " + str(h_clust_cg[clust_01_frames[centroid_ID]]))

ssd_rmsd_cg_2 = []
for n, pair in enumerate( combinations( clust_02, 2 ) ):
    #superpose( *pair )
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_rmsd_cg_2.append(rms)
ssd_rmsd_cg_2 = np.array( ssd_rmsd_cg_2 )
k_clust, cost,num	= kmedoids( ssd_rmsd_cg_2, nclusters=1, npass=100)
centroid_ID = list(set(k_clust))[0]
print("centroid ID: " + str(centroid_ID))
print('representative frame: ' + str(clust_02_frames[centroid_ID]))
print("cluster state: " + str(h_clust_cg[clust_02_frames[centroid_ID]]))

ssd_rmsd_cg_3 = []
for n, pair in enumerate( combinations( clust_03, 2 ) ):
    #superpose( *pair )
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_rmsd_cg_3.append(rms)
ssd_rmsd_cg_3 = np.array( ssd_rmsd_cg_3 )
k_clust, cost,num	= kmedoids( ssd_rmsd_cg_3, nclusters=1, npass=100)
centroid_ID = list(set(k_clust))[0]
print("centroid ID: " + str(centroid_ID))
print('representative frame: ' + str(clust_03_frames[centroid_ID]))
print("cluster state: " + str(h_clust_cg[clust_03_frames[centroid_ID]]))

ssd_rmsd_cg_4 = []
for n, pair in enumerate( combinations( clust_04, 2 ) ):
    #superpose( *pair )
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_rmsd_cg_4.append(rms)
ssd_rmsd_cg_4 = np.array( ssd_rmsd_cg_4 )
k_clust, cost,num	= kmedoids( ssd_rmsd_cg_4, nclusters=1, npass=100)
centroid_ID = list(set(k_clust))[0]
print("centroid ID: " + str(centroid_ID))
print('representative frame: ' + str(clust_04_frames[centroid_ID]))
print("cluster state: " + str(h_clust_cg[clust_04_frames[centroid_ID]]))

# ssd_rmsd_cg = []
# for n, pair in enumerate( combinations( clust_05, 2 ) ):
#     #superpose( *pair )
#     a, b = pairwise[0], pairwise[1]
#     rms = calcRMS(np.array(a)-np.array(b))
#     ssd_rmsd_cg.append(rms)
# ssd_rmsd_cg = np.array( ssd_rmsd_cg )
# k_clust, cost,num	= kmedoids( ssd_rmsd_cg, nclusters=1, npass=100)
# centroid_ID = list(set(k_clust))[0]
# print("centroid ID: " + str(centroid_ID))
# print('representative frame: ' + str(clust_05_frames[centroid_ID]))
# print("cluster state: " + str(h_clust_cg[clust_05_frames[centroid_ID]]))

# ssd_rmsd_cg = []
# for n, pair in enumerate( combinations( clust_06, 2 ) ):
#     #superpose( *pair )
#     a, b = pairwise[0], pairwise[1]
#     rms = calcRMS(np.array(a)-np.array(b))
#     ssd_rmsd_cg.append(rms)
# ssd_rmsd_cg = np.array( ssd_rmsd_cg )
# k_clust, cost,num	= kmedoids( ssd_rmsd_cg, nclusters=1, npass=100)
# centroid_ID = list(set(k_clust))[0]
# print("centroid ID: " + str(centroid_ID))
# print('representative frame: ' + str(clust_06_frames[centroid_ID]))
# print("cluster state: " + str(h_clust_cg[clust_06_frames[centroid_ID]]))

# ssd_rmsd_cg = []
# for n, pair in enumerate( combinations( clust_07, 2 ) ):
#     #superpose( *pair )
#     a, b = pairwise[0], pairwise[1]
#     rms = calcRMS(np.array(a)-np.array(b))
#     ssd_rmsd_cg.append(rms)
# ssd_rmsd_cg = np.array( ssd_rmsd_cg )
# k_clust, cost,num	= kmedoids( ssd_rmsd_cg, nclusters=1, npass=100)
# centroid_ID = list(set(k_clust))[0]
# print("centroid ID: " + str(centroid_ID))
# print('representative frame: ' + str(clust_07_frames[centroid_ID]))
# print("cluster state: " + str(h_clust_cg[clust_07_frames[centroid_ID]]))

# ssd_rmsd_cg = []
# for n, pair in enumerate( combinations( clust_08, 2 ) ):
#     #superpose( *pair )
#     a, b = pairwise[0], pairwise[1]
#     rms = calcRMS(np.array(a)-np.array(b))
#     ssd_rmsd_cg.append(rms)
# ssd_rmsd_cg = np.array( ssd_rmsd_cg )
# k_clust, cost,num	= kmedoids( ssd_rmsd_cg, nclusters=1, npass=100)
# centroid_ID = list(set(k_clust))[0]
# print("centroid ID: " + str(centroid_ID))
# print('representative frame: ' + str(clust_08_frames[centroid_ID]))
# print("cluster state: " + str(h_clust_cg[clust_08_frames[centroid_ID]]))

# ssd_rmsd_cg = []
# for n, pair in enumerate( combinations( clust_09, 2 ) ):
#     #superpose( *pair )
#     a, b = pairwise[0], pairwise[1]
#     rms = calcRMS(np.array(a)-np.array(b))
#     ssd_rmsd_cg.append(rms)
# ssd_rmsd_cg = np.array( ssd_rmsd_cg )
# k_clust, cost,num	= kmedoids( ssd_rmsd_cg, nclusters=1, npass=100)
# centroid_ID = list(set(k_clust))[0]
# print("centroid ID: " + str(centroid_ID))
# print('representative frame: ' + str(clust_09_frames[centroid_ID]))
# print("cluster state: " + str(h_clust_cg[clust_09_frames[centroid_ID]]))


# In[73]:


cg_clust_id_1 = fts_4e10_cg[0]
cg_clust_id_2 = fts_4e10_cg[20]
cg_clust_id_3 = fts_4e10_cg[1]
#cg_clust_id_4 = fts_4e10_cg[65]

print(cg_clust_id_1)
print(cg_clust_id_2)
print(cg_clust_id_3)
#print(cg_clust_id_4)


# In[29]:


cg_clust_id_1 = fts_4e10_cg[117]
cg_clust_id_2 = fts_4e10_cg[0]
cg_clust_id_3 = fts_4e10_cg[20]
cg_clust_id_4 = fts_4e10_cg[1]
cg_clust_id_5 = fts_4e10_cg[112]
cg_clust_id_6 = fts_4e10_cg[1868]
cg_clust_id_7 = fts_4e10_cg[9970]
cg_clust_id_8 = fts_4e10_cg[109]
cg_clust_id_9 = fts_4e10_cg[1990]
cg_clust_centroids = [cg_clust_id_1, cg_clust_id_2, cg_clust_id_3, cg_clust_id_4, 
                     cg_clust_id_5, cg_clust_id_6, cg_clust_id_7, cg_clust_id_8, 
                     cg_clust_id_9]


# In[ ]:





# In[37]:


#CELL 10 - 4e10 1 us 
fab_selection_str = 'protein'
mem_selection_str = 'resname POPC POPA CHOL'


input_pdb = parsePDB('/home/bat-gpu/colleen/phos_interactions/final_traj/4e10_ppm_1us/analysis_input.pdb')
dcd = DCDFile('/home/bat-gpu/colleen/phos_interactions/final_traj/4e10_ppm_1us/complete_md.dcd')
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

#will be X frame length of ft vector 
#ft vector will contain 15 fts (in following order):
#cdrl1_loop_contact_counts
#cdrl2_loop_contact_counts
#cdrl3_loop_contact_counts
#cdrh1_loop_contact_counts
#cdrh2_loop_contact_counts
#cdrh1_loop_contact_counts
#cdrl1_loop_COM_depth
#cdrl2_loop_COM_depth
#cdrl3_loop_COM_depth
#cdrh1_loop_COM_depth
#cdrh2_loop_COM_depth
#cdrh3_loop_COM_depth
#full_fab_long_ax_ang 
#var_dom_long_ax_ang
#short_ax_ang

fts_4e10_aa= [] 

################ LOOP THROUGH TRAJ TO CALC FEATURE VECTORS ########################   
for i, frame in enumerate(dcd):

    ft_vect = []

    #define fab and membrane 
    frame_fab = frame.getAtoms().select( fab_selection_str )
    frame_fab_bb = frame.getAtoms().select( 'name CA' )
    frame_mem = frame.getAtoms().select( mem_selection_str )

    #DEFINE MEMBRANE LAYERS - bilayer midpoint, top PO4 plane, bottom PO4 plane 

    avg_bilayer_mp = (int(sum(list(frame_mem.getCoords()[:,2]))/len(frame_mem.getResnames())))

    mem_top_sel_str  = 'resname POPC name P and z > '+ str(avg_bilayer_mp)
    mem_bot_sel_str  = 'resname POPC name P and z < '+ str(avg_bilayer_mp)

    top_leaf = frame.getAtoms().select(mem_top_sel_str)
    top_leaf_points = top_leaf.getCoords()
    top_leaf_plane = planeFit(np.transpose(top_leaf_points))
    
    
    bot_leaf = frame.getAtoms().select(mem_bot_sel_str)
    bot_leaf_points = bot_leaf.getCoords()
    bot_leaf_plane = planeFit(np.transpose(bot_leaf_points))

    
    bot_phos_layer_sel_str  = 'resname POPC name P and z < ' +str(avg_bilayer_mp)
    phos = frame.getAtoms().select(bot_phos_layer_sel_str)
    phos_points = phos.getCoords()
    phos_z = phos_points[:,2]
    avg_bot_phos_z = sum(phos_z)/len(phos_z)
    bot_phos_layer_dist2bilayer_cen = avg_bot_phos_z - avg_bilayer_mp

    
    top_phos_layer_sel_str  = 'resname POPC name P and z > ' +str(avg_bilayer_mp)
    phos = frame.getAtoms().select(top_phos_layer_sel_str)
    phos_points = phos.getCoords()
    phos_z = phos_points[:,2]
    avg_top_phos_z = sum(phos_z)/len(phos_z)
    top_phos_layer_dist2bilayer_cen = avg_top_phos_z - avg_bilayer_mp
    
    
#     avg_bot_phos_dist2bilayer_cen.append(bot_phos_layer_dist2bilayer_cen)
#     avg_top_phos_dist2bilayer_cen.append(top_phos_layer_dist2bilayer_cen)
#     bilayer_cen_dist2bilayer_cen.append(avg_bilayer_mp-avg_bilayer_mp)
    
    
    ################ CDR LOOP RESIDUES - CONTACT MEMBRANE STATS########################   

    #CDRL1 loops residues 
    ser_027 = frame_fab_bb[27].getCoords()
    val_028 = frame_fab_bb[28].getCoords()  
    gly_029 = frame_fab_bb[29].getCoords()   
    asn_030 = frame_fab_bb[30].getCoords()
    asn_031 = frame_fab_bb[31].getCoords()

    #CDRL2 loops residues 
    tyr_049 = frame_fab_bb[49].getCoords()
    gly_050 = frame_fab_bb[50].getCoords()
    ala_051 = frame_fab_bb[51].getCoords()
    ser_052 = frame_fab_bb[52].getCoords()
    ser_053 = frame_fab_bb[53].getCoords()
    arg_054 = frame_fab_bb[54].getCoords()
    pro_055 = frame_fab_bb[55].getCoords()

    #CDRL3 loops residues   
    tyr_091 = frame_fab_bb[91].getCoords()
    gly_092 = frame_fab_bb[92].getCoords()
    gln_093 = frame_fab_bb[93].getCoords()
    ser_094 = frame_fab_bb[94].getCoords()
    leu_095 = frame_fab_bb[95].getCoords()
    ser_096 = frame_fab_bb[96].getCoords()

    #CDRH1 loop residues 
    gly_239 = frame_fab_bb[239].getCoords()
    gly_240 = frame_fab_bb[240].getCoords()
    ser_241 = frame_fab_bb[241].getCoords()  
    phe_242 = frame_fab_bb[242].getCoords()
    ser_243 = frame_fab_bb[243].getCoords()
    thr_244 = frame_fab_bb[244].getCoords()
    tyr_245 = frame_fab_bb[245].getCoords()

    
    #CDRH2 loop residues 
    pro_266 = frame_fab_bb[266].getCoords()
    leu_267 = frame_fab_bb[267].getCoords()
    leu_268 = frame_fab_bb[268].getCoords()
    thr_269 = frame_fab_bb[269].getCoords()

    #CDRH3 loop residues 
    gly_316 = frame_fab_bb[316].getCoords()
    trp_317 = frame_fab_bb[317].getCoords() 
    gly_318 = frame_fab_bb[318].getCoords()
    trp_319 = frame_fab_bb[319].getCoords()  
    leu_320 = frame_fab_bb[320].getCoords() 
    gly_321 = frame_fab_bb[321].getCoords()
    lys_322 = frame_fab_bb[322].getCoords()
    pro_323 = frame_fab_bb[323].getCoords()
    ile_324 = frame_fab_bb[324].getCoords()
    
    #list of crd loop residue coords 
    cdr_loop_res_coords = [ser_027, val_028, gly_029, asn_030, asn_031, 
                           tyr_049, gly_050, ala_051 ,ser_052, ser_053, arg_054, pro_055, 
                           tyr_091, gly_092,  gln_093, ser_094, leu_095, ser_096,
                           gly_239, gly_240, ser_241, phe_242, ser_243, tyr_245, 
                           pro_266, leu_267, leu_268, thr_269,
                           gly_316, trp_317, gly_318, trp_319, leu_320, gly_321, lys_322, pro_323, ile_324]
    #get binary vector of crd loop membrane contacts - do not add to ft vect for clustering currently
    loop_contacts = findMemContacts(cdr_loop_res_coords, bot_leaf_plane)
    
    loop_contact_counts = countMemContacts(cdr_loop_res_coords, bot_leaf_plane)
    #print(loop_contact_counts)
    #add each loops' residue contact count to ft vect as a value 
    ft_vect.append(loop_contact_counts[0])
    ft_vect.append(loop_contact_counts[1])
    ft_vect.append(loop_contact_counts[2])
    ft_vect.append(loop_contact_counts[3])
    ft_vect.append(loop_contact_counts[4])
    ft_vect.append(loop_contact_counts[5])
    
################ CDR LOOP CENTER OF MASS (COM) DEPTHS - relative to PO4 plane ########################   
       #*remember- actual simulation is upside down, these calculations will be plotted so that Fab is on top of bilayer 
    #CDRL1 
    cdrl1_resis = frame_fab_bb.select('resnum 27 to 33 ')
    cdrl1_com_z = calcCenter(cdrl1_resis)[2]
    cdrl1_loop_COM_depth = avg_bot_phos_z-cdrl1_com_z
    #cdrl1_com_dist2po4_bot.append(avg_bot_phos_z-cdrl1_com_z)
    #cdrl1_com_dist2bilayer_cen.append(cdrl1_com_z-avg_bilayer_mp)
    
    #cdrl1_com_depth_aggregate.append(cdrl1_com[2])
    
    #CDRL2
    cdrl2_resis = frame_fab_bb.select('resnum 49 to 56')
    cdrl2_com_z = calcCenter(cdrl2_resis)[2]
    cdrl2_loop_COM_depth = avg_bot_phos_z-cdrl2_com_z
    #cdrl2_com_dist2po4_bot.append(avg_bot_phos_z-cdrl2_com_z)

    #cdrl2_com_dist2bilayer_cen.append(cdrl2_com_z-avg_bilayer_mp)
    
    #CDRL3
    cdrl3_resis = frame_fab_bb.select('resnum 91 to 98')
    cdrl3_com_z = calcCenter(cdrl3_resis)[2]
    cdrl3_loop_COM_depth = avg_bot_phos_z-cdrl3_com_z
    #cdrl3_com_dist2po4_bot.append(avg_bot_phos_z-cdrl3_com_z)

    #cdrl3_com_dist2bilayer_cen.append(avg_bot_phos_z-cdrl3_com_z)
    
    #CDRH1
    cdrh1_resis = frame_fab_bb.select('resnum 240 to 246')
    cdrh1_com_z = calcCenter(cdrh1_resis)[2]
    cdrh1_loop_COM_depth = avg_bot_phos_z-cdrh1_com_z
    #cdrh1_com_dist2po4_bot.append(avg_bot_phos_z-cdrh1_com_z)
    #cdrh1_com_dist2bilayer_cen.append(avg_bot_phos_z-cdrh1_com_z)
    
    #CDRH2 resdius 
    cdrh2_resis = frame_fab_bb.select('resnum 267 to 270')
    cdrh2_com_z = calcCenter(cdrh2_resis)[2]
    cdrh2_loop_COM_depth = avg_bot_phos_z-cdrh2_com_z
    #cdrh2_com_dist2po4_bot.append(avg_bot_phos_z-cdrh2_com_z)
    #cdrh2_com_dist2bilayer_cen.append(avg_bot_phos_z-cdrh2_com_z)
    
    #CDRH3 
    cdrh3_resis = frame_fab_bb.select('resnum 315 to 325')
    cdrh3_com_z = calcCenter(cdrh2_resis)[2]
    cdrh3_loop_COM_depth = avg_bot_phos_z-cdrh3_com_z
    #cdrh3_com_dist2po4_bot.append(avg_bot_phos_z-cdrh3_com_z)
    #cdrh3_com_dist2bilayer_cen.append(cdrh3_com_z-avg_bilayer_mp)

    ft_vect.append(round(cdrl1_loop_COM_depth, 2))
    ft_vect.append(round(cdrl2_loop_COM_depth, 2))
    ft_vect.append(round(cdrl3_loop_COM_depth, 2))
    ft_vect.append(round(cdrh1_loop_COM_depth, 2))
    ft_vect.append(round(cdrh2_loop_COM_depth, 2))
    ft_vect.append(round(cdrh3_loop_COM_depth, 2))
################ ALING NORM OF PO4 PLANE TO Z, calculate ANGLES ########################   

    #define as rotation b/w norm vect of PO4 plane & z-axis (unit normal of Z axis )
    rotation = VectorAlign(bot_leaf_plane[1], np.array([0, 0, 1]))
    #define transformation b/w roatino and origin 
    transformZax = Transformation(rotation, np.zeros(3))
    #apply transofrmation of plane to Z axis to entire system 
    applyTransformation(transformZax, frame.getAtoms())
    
    frame_fab = frame.getAtoms().select( fab_selection_str )

    #define selections for points used to make vectors through fab 
    
    #center of loops is p2 for all angles 
    cen_loops = frame_fab_bb.select('resnum 41 or resnum 256')
    fab_cen = calcCenter(cen_loops)
    
    cdrh_points = frame_fab_bb.select('resnum 243 or resnum 269 or resnum 319')
    cdrl_points = frame_fab_bb.select('resnum 31 or resnum 53 or resnum 95')
    crd_loops = cdrh_points + cdrl_points    
    cdr_loops_cen = calcCenter(crd_loops)
    
    fab_tip = frame_fab_bb.select('resnum 231 or resnum 442')
    fab_tip_cen = calcCenter(fab_tip)
 
    po4_plan_norm = -1*bot_leaf_plane[1] #second vector is norm of PO4 plane 
    
    #var_dom_long_axis 
    #define vector through center of fab and center of cdr loops 
    var_dom_long_ax =  np.array(fab_cen-cdr_loops_cen)
    var_dom_long_ax_ang = angle(po4_plan_norm, var_dom_long_ax)

    #print('VAR DOM ANG: ', var_dom_long_ax_ang)
    #print('VAR DOM ANG: ',angle(po4_plan_norm, var_dom_long_ax))
    #full_fab_long_axis 
    #full_fab_long_ax =  np.array(fab_tip_cen - cdr_loops_cen)
    full_fab_long_ax =  np.array(fab_tip_cen - cdr_loops_cen)

    full_fab_long_ax_ang = angle(po4_plan_norm, full_fab_long_ax)

    #short_axis 
    l_chain_middle_edge = frame_fab_bb.select('resnum 107').getCoords()
    h_chain_middle_edge = frame_fab_bb.select('resnum 340').getCoords()
    short_axis_vect = np.array( h_chain_middle_edge[0] - l_chain_middle_edge[0] )
    #short_axis_vect = np.array( l_chain_middle_edge[0] -h_chain_middle_edge[0] ) - this should give opposite sign angle as vector above 
    short_ax_ang = angle(po4_plan_norm, short_axis_vect)
     
    #add angles to ft vector 
    #ft_vect.append(full_fab_long_ax_ang)
    ft_vect.append(var_dom_long_ax_ang)
    ft_vect.append(short_ax_ang)
    
    #add ft vect to aggregate lsit and to traj specific list 
    fts_4e10_aa.append(ft_vect)
    #fts_4e10_ppm_1us.append(ft_vect)


# In[38]:


#CELL 1 
fab_selection_str = 'protein'
mem_selection_str = 'resname POPC POPA CHOL'

#note; all atom sims have fab in bottom membrane layer, cg have fab in top 

input_pdb = parsePDB('/home/bat-gpu/colleen/phos_interactions/final_traj/4e10_ppm/analysis_input.pdb')
dcd = DCDFile('/home/bat-gpu/colleen/phos_interactions/final_traj/4e10_ppm/complete_md.dcd')
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

#will be X frame length of ft vector 
#ft vector will contain 15 fts (in following order):
#cdrl1_loop_contact_counts
#cdrl2_loop_contact_counts
#cdrl3_loop_contact_counts
#cdrh1_loop_contact_counts
#cdrh2_loop_contact_counts
#cdrh1_loop_contact_counts
#cdrl1_loop_COM_depth
#cdrl2_loop_COM_depth
#cdrl3_loop_COM_depth
#cdrh1_loop_COM_depth
#cdrh2_loop_COM_depth
#cdrh3_loop_COM_depth
#full_fab_long_ax_ang 
#var_dom_long_ax_ang
#short_ax_ang

#fts_4e10_ppm = [] 

################ LOOP THROUGH TRAJ TO CALC FEATURE VECTORS ########################   
for i, frame in enumerate(dcd):

    ft_vect = []

    #define fab and membrane 
    frame_fab = frame.getAtoms().select( fab_selection_str )
    frame_fab_bb = frame.getAtoms().select( 'name CA' )
    frame_mem = frame.getAtoms().select( mem_selection_str )

    #DEFINE MEMBRANE LAYERS - bilayer midpoint, top PO4 plane, bottom PO4 plane 

    avg_bilayer_mp = (int(sum(list(frame_mem.getCoords()[:,2]))/len(frame_mem.getResnames())))

    mem_top_sel_str  = 'resname POPC name P and z > '+ str(avg_bilayer_mp)
    mem_bot_sel_str  = 'resname POPC name P and z < '+ str(avg_bilayer_mp)

    top_leaf = frame.getAtoms().select(mem_top_sel_str)
    top_leaf_points = top_leaf.getCoords()
    top_leaf_plane = planeFit(np.transpose(top_leaf_points))
    
    
    bot_leaf = frame.getAtoms().select(mem_bot_sel_str)
    bot_leaf_points = bot_leaf.getCoords()
    bot_leaf_plane = planeFit(np.transpose(bot_leaf_points))

    
    bot_phos_layer_sel_str  = 'resname POPC name P and z < ' +str(avg_bilayer_mp)
    phos = frame.getAtoms().select(bot_phos_layer_sel_str)
    phos_points = phos.getCoords()
    phos_z = phos_points[:,2]
    avg_bot_phos_z = sum(phos_z)/len(phos_z)
    bot_phos_layer_dist2bilayer_cen = avg_bot_phos_z - avg_bilayer_mp

    
    top_phos_layer_sel_str  = 'resname POPC name P and z > ' +str(avg_bilayer_mp)
    phos = frame.getAtoms().select(top_phos_layer_sel_str)
    phos_points = phos.getCoords()
    phos_z = phos_points[:,2]
    avg_top_phos_z = sum(phos_z)/len(phos_z)
    top_phos_layer_dist2bilayer_cen = avg_top_phos_z - avg_bilayer_mp
    
    
#     avg_bot_phos_dist2bilayer_cen.append(bot_phos_layer_dist2bilayer_cen)
#     avg_top_phos_dist2bilayer_cen.append(top_phos_layer_dist2bilayer_cen)
#     bilayer_cen_dist2bilayer_cen.append(avg_bilayer_mp-avg_bilayer_mp)
    
    
    ################ CDR LOOP RESIDUES - CONTACT MEMBRANE STATS########################   

    #CDRL1 loops residues 
    ser_027 = frame_fab_bb[27].getCoords()
    val_028 = frame_fab_bb[28].getCoords()  
    gly_029 = frame_fab_bb[29].getCoords()   
    asn_030 = frame_fab_bb[30].getCoords()
    asn_031 = frame_fab_bb[31].getCoords()

    #CDRL2 loops residues 
    tyr_049 = frame_fab_bb[49].getCoords()
    gly_050 = frame_fab_bb[50].getCoords()
    ala_051 = frame_fab_bb[51].getCoords()
    ser_052 = frame_fab_bb[52].getCoords()
    ser_053 = frame_fab_bb[53].getCoords()
    arg_054 = frame_fab_bb[54].getCoords()
    pro_055 = frame_fab_bb[55].getCoords()

    #CDRL3 loops residues   
    tyr_091 = frame_fab_bb[91].getCoords()
    gly_092 = frame_fab_bb[92].getCoords()
    gln_093 = frame_fab_bb[93].getCoords()
    ser_094 = frame_fab_bb[94].getCoords()
    leu_095 = frame_fab_bb[95].getCoords()
    ser_096 = frame_fab_bb[96].getCoords()

    #CDRH1 loop residues 
    gly_239 = frame_fab_bb[239].getCoords()
    gly_240 = frame_fab_bb[240].getCoords()
    ser_241 = frame_fab_bb[241].getCoords()  
    phe_242 = frame_fab_bb[242].getCoords()
    ser_243 = frame_fab_bb[243].getCoords()
    thr_244 = frame_fab_bb[244].getCoords()
    tyr_245 = frame_fab_bb[245].getCoords()

    
    #CDRH2 loop residues 
    pro_266 = frame_fab_bb[266].getCoords()
    leu_267 = frame_fab_bb[267].getCoords()
    leu_268 = frame_fab_bb[268].getCoords()
    thr_269 = frame_fab_bb[269].getCoords()

    #CDRH3 loop residues 
    gly_316 = frame_fab_bb[316].getCoords()
    trp_317 = frame_fab_bb[317].getCoords() 
    gly_318 = frame_fab_bb[318].getCoords()
    trp_319 = frame_fab_bb[319].getCoords()  
    leu_320 = frame_fab_bb[320].getCoords() 
    gly_321 = frame_fab_bb[321].getCoords()
    lys_322 = frame_fab_bb[322].getCoords()
    pro_323 = frame_fab_bb[323].getCoords()
    ile_324 = frame_fab_bb[324].getCoords()
    
    #list of crd loop residue coords 
    cdr_loop_res_coords = [ser_027, val_028, gly_029, asn_030, asn_031, 
                           tyr_049, gly_050, ala_051 ,ser_052, ser_053, arg_054, pro_055, 
                           tyr_091, gly_092,  gln_093, ser_094, leu_095, ser_096,
                           gly_239, gly_240, ser_241, phe_242, ser_243, tyr_245, 
                           pro_266, leu_267, leu_268, thr_269,
                           gly_316, trp_317, gly_318, trp_319, leu_320, gly_321, lys_322, pro_323, ile_324]
    #get binary vector of crd loop membrane contacts - do not add to ft vect for clustering currently
    loop_contacts = findMemContacts(cdr_loop_res_coords, bot_leaf_plane)
    
    loop_contact_counts = countMemContacts(cdr_loop_res_coords, bot_leaf_plane)
    #print(loop_contact_counts)
    #add each loops' residue contact count to ft vect as a value 
    ft_vect.append(loop_contact_counts[0])
    ft_vect.append(loop_contact_counts[1])
    ft_vect.append(loop_contact_counts[2])
    ft_vect.append(loop_contact_counts[3])
    ft_vect.append(loop_contact_counts[4])
    ft_vect.append(loop_contact_counts[5])
    
################ CDR LOOP CENTER OF MASS (COM) DEPTHS - relative to PO4 plane ########################   
    #*remember- actual simulation is upside down, these calculations will be plotted so that Fab is on top of bilayer 
    #CDRL1 
    cdrl1_resis = frame_fab_bb.select('resnum 27_ to 33_ ')
    cdrl1_com_z = calcCenter(cdrl1_resis)[2]
    cdrl1_loop_COM_depth = avg_bot_phos_z-cdrl1_com_z
    #cdrl1_com_dist2po4_bot.append(avg_bot_phos_z-cdrl1_com_z)
    #cdrl1_com_dist2bilayer_cen.append(cdrl1_com_z-avg_bilayer_mp)
    
    #cdrl1_com_depth_aggregate.append(cdrl1_com[2])
    
    #CDRL2
    cdrl2_resis = frame_fab_bb.select('resnum 49 to 56')
    cdrl2_com_z = calcCenter(cdrl2_resis)[2]
    cdrl2_loop_COM_depth = avg_bot_phos_z-cdrl2_com_z
    #cdrl2_com_dist2po4_bot.append(avg_bot_phos_z-cdrl2_com_z)

    #cdrl2_com_dist2bilayer_cen.append(cdrl2_com_z-avg_bilayer_mp)
    
    #CDRL3
    cdrl3_resis = frame_fab_bb.select('resnum 91 to 98')
    cdrl3_com_z = calcCenter(cdrl3_resis)[2]
    cdrl3_loop_COM_depth = avg_bot_phos_z-cdrl3_com_z
    #cdrl3_com_dist2po4_bot.append(avg_bot_phos_z-cdrl3_com_z)

    #cdrl3_com_dist2bilayer_cen.append(avg_bot_phos_z-cdrl3_com_z)
    
    #CDRH1
    cdrh1_resis = frame_fab_bb.select('resnum 240 to 246')
    cdrh1_com_z = calcCenter(cdrh1_resis)[2]
    cdrh1_loop_COM_depth = avg_bot_phos_z-cdrh1_com_z
    #cdrh1_com_dist2po4_bot.append(avg_bot_phos_z-cdrh1_com_z)
    #cdrh1_com_dist2bilayer_cen.append(avg_bot_phos_z-cdrh1_com_z)
    
    #CDRH2 resdius 
    cdrh2_resis = frame_fab_bb.select('resnum 267 to 270')
    cdrh2_com_z = calcCenter(cdrh2_resis)[2]
    cdrh2_loop_COM_depth = avg_bot_phos_z-cdrh2_com_z
    #cdrh2_com_dist2po4_bot.append(avg_bot_phos_z-cdrh2_com_z)
    #cdrh2_com_dist2bilayer_cen.append(avg_bot_phos_z-cdrh2_com_z)
    
    #CDRH3 
    cdrh3_resis = frame_fab_bb.select('resnum 315 to 325')
    cdrh3_com_z = calcCenter(cdrh2_resis)[2]
    cdrh3_loop_COM_depth = avg_bot_phos_z-cdrh3_com_z
    #cdrh3_com_dist2po4_bot.append(avg_bot_phos_z-cdrh3_com_z)
    #cdrh3_com_dist2bilayer_cen.append(cdrh3_com_z-avg_bilayer_mp)

    ft_vect.append(round(cdrl1_loop_COM_depth, 2))
    ft_vect.append(round(cdrl2_loop_COM_depth, 2))
    ft_vect.append(round(cdrl3_loop_COM_depth, 2))
    ft_vect.append(round(cdrh1_loop_COM_depth, 2))
    ft_vect.append(round(cdrh2_loop_COM_depth, 2))
    ft_vect.append(round(cdrh3_loop_COM_depth, 2))
################ ALING NORM OF PO4 PLANE TO Z, calculate ANGLES ########################   

    #define as rotation b/w norm vect of PO4 plane & z-axis (unit normal of Z axis )
    rotation = VectorAlign(bot_leaf_plane[1], np.array([0, 0, 1]))
    #define transformation b/w roatino and origin 
    transformZax = Transformation(rotation, np.zeros(3))
    #apply transofrmation of plane to Z axis to entire system 
    applyTransformation(transformZax, frame.getAtoms())
    
    frame_fab = frame.getAtoms().select( fab_selection_str )

    #define selections for points used to make vectors through fab 
    
    #center of loops is p2 for all angles 
    cen_loops = frame_fab_bb.select('resnum 41 or resnum 256')
    fab_cen = calcCenter(cen_loops)
    
    cdrh_points = frame_fab_bb.select('resnum 243 or resnum 269 or resnum 319')
    cdrl_points = frame_fab_bb.select('resnum 31 or resnum 53 or resnum 95')
    crd_loops = cdrh_points + cdrl_points    
    cdr_loops_cen = calcCenter(crd_loops)
    
    fab_tip = frame_fab_bb.select('resnum 213 or resnum 41')
    fab_tip_cen = calcCenter(fab_tip)
 
    po4_plan_norm = -1*top_leaf_plane[1] #second vector is norm of PO4 plane 
    
    #var_dom_long_axis 
    #define vector through center of fab and center of cdr loops 
    var_dom_long_ax =  np.array(fab_cen-cdr_loops_cen)
    var_dom_long_ax_ang = angle(po4_plan_norm, var_dom_long_ax)

    #print('VAR DOM ANG: ', var_dom_long_ax_ang)
    #print('VAR DOM ANG: ',angle(po4_plan_norm, var_dom_long_ax))
    #full_fab_long_axis 
    #full_fab_long_ax =  np.array(fab_tip_cen - cdr_loops_cen)
    full_fab_long_ax =  np.array(fab_tip_cen - cdr_loops_cen)

    full_fab_long_ax_ang = angle(po4_plan_norm, full_fab_long_ax)

    #short_axis 
    l_chain_middle_edge = frame_fab_bb.select('resnum 107').getCoords()
    h_chain_middle_edge = frame_fab_bb.select('resnum 340').getCoords()
    short_axis_vect = np.array( h_chain_middle_edge[0] - l_chain_middle_edge[0] )
    #short_axis_vect = np.array( l_chain_middle_edge[0] -h_chain_middle_edge[0] ) - this should give opposite sign angle as vector above 
    short_ax_ang = angle(po4_plan_norm, short_axis_vect)
     
    #add angles to ft vector 
    #ft_vect.append(round(full_fab_long_ax_ang, 2))
    ft_vect.append(round(var_dom_long_ax_ang, 2))
    ft_vect.append(round(short_ax_ang, 2)) 
    
    #add ft vect to aggregate lsit and to traj specific list 
    fts_4e10_aa.append(ft_vect)
    #fts_4e10_ppm.append(ft_vect)


# In[39]:


#CELL 2 
fab_selection_str = 'protein'
mem_selection_str = 'resname POPC POPA CHOL'


input_pdb = parsePDB('/home/bat-gpu/colleen/phos_interactions/final_traj/4e10_p15/analysis_input.pdb')
dcd = DCDFile('/home/bat-gpu/colleen/phos_interactions/final_traj/4e10_p15/complete_md.dcd')
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

#will be X frame length of ft vector 
#ft vector will contain 15 fts (in following order):
#cdrl1_loop_contact_counts
#cdrl2_loop_contact_counts
#cdrl3_loop_contact_counts
#cdrh1_loop_contact_counts
#cdrh2_loop_contact_counts
#cdrh1_loop_contact_counts
#cdrl1_loop_COM_depth
#cdrl2_loop_COM_depth
#cdrl3_loop_COM_depth
#cdrh1_loop_COM_depth
#cdrh2_loop_COM_depth
#cdrh3_loop_COM_depth
#full_fab_long_ax_ang 
#var_dom_long_ax_ang
#short_ax_ang

#fts_4e10_p15 = [] 

################ LOOP THROUGH TRAJ TO CALC FEATURE VECTORS ########################   
for i, frame in enumerate(dcd):

    ft_vect = []

    #define fab and membrane 
    frame_fab = frame.getAtoms().select( fab_selection_str )
    frame_fab_bb = frame.getAtoms().select( 'name CA' )
    frame_mem = frame.getAtoms().select( mem_selection_str )

    #DEFINE MEMBRANE LAYERS - bilayer midpoint, top PO4 plane, bottom PO4 plane 

    avg_bilayer_mp = (int(sum(list(frame_mem.getCoords()[:,2]))/len(frame_mem.getResnames())))

    mem_top_sel_str  = 'resname POPC name P and z > '+ str(avg_bilayer_mp)
    mem_bot_sel_str  = 'resname POPC name P and z < '+ str(avg_bilayer_mp)

    top_leaf = frame.getAtoms().select(mem_top_sel_str)
    top_leaf_points = top_leaf.getCoords()
    top_leaf_plane = planeFit(np.transpose(top_leaf_points))
    
    
    bot_leaf = frame.getAtoms().select(mem_bot_sel_str)
    bot_leaf_points = bot_leaf.getCoords()
    bot_leaf_plane = planeFit(np.transpose(bot_leaf_points))

    
    bot_phos_layer_sel_str  = 'resname POPC name P and z < ' +str(avg_bilayer_mp)
    phos = frame.getAtoms().select(bot_phos_layer_sel_str)
    phos_points = phos.getCoords()
    phos_z = phos_points[:,2]
    avg_bot_phos_z = sum(phos_z)/len(phos_z)
    bot_phos_layer_dist2bilayer_cen = avg_bot_phos_z - avg_bilayer_mp

    
    top_phos_layer_sel_str  = 'resname POPC name P and z > ' +str(avg_bilayer_mp)
    phos = frame.getAtoms().select(top_phos_layer_sel_str)
    phos_points = phos.getCoords()
    phos_z = phos_points[:,2]
    avg_top_phos_z = sum(phos_z)/len(phos_z)
    top_phos_layer_dist2bilayer_cen = avg_top_phos_z - avg_bilayer_mp
    
    
#     avg_bot_phos_dist2bilayer_cen.append(bot_phos_layer_dist2bilayer_cen)
#     avg_top_phos_dist2bilayer_cen.append(top_phos_layer_dist2bilayer_cen)
#     bilayer_cen_dist2bilayer_cen.append(avg_bilayer_mp-avg_bilayer_mp)
    
    
    ################ CDR LOOP RESIDUES - CONTACT MEMBRANE STATS########################   

    #CDRL1 loops residues 
    ser_027 = frame_fab_bb[27].getCoords()
    val_028 = frame_fab_bb[28].getCoords()  
    gly_029 = frame_fab_bb[29].getCoords()   
    asn_030 = frame_fab_bb[30].getCoords()
    asn_031 = frame_fab_bb[31].getCoords()

    #CDRL2 loops residues 
    tyr_049 = frame_fab_bb[49].getCoords()
    gly_050 = frame_fab_bb[50].getCoords()
    ala_051 = frame_fab_bb[51].getCoords()
    ser_052 = frame_fab_bb[52].getCoords()
    ser_053 = frame_fab_bb[53].getCoords()
    arg_054 = frame_fab_bb[54].getCoords()
    pro_055 = frame_fab_bb[55].getCoords()

    #CDRL3 loops residues   
    tyr_091 = frame_fab_bb[91].getCoords()
    gly_092 = frame_fab_bb[92].getCoords()
    gln_093 = frame_fab_bb[93].getCoords()
    ser_094 = frame_fab_bb[94].getCoords()
    leu_095 = frame_fab_bb[95].getCoords()
    ser_096 = frame_fab_bb[96].getCoords()

    #CDRH1 loop residues 
    gly_239 = frame_fab_bb[239].getCoords()
    gly_240 = frame_fab_bb[240].getCoords()
    ser_241 = frame_fab_bb[241].getCoords()  
    phe_242 = frame_fab_bb[242].getCoords()
    ser_243 = frame_fab_bb[243].getCoords()
    thr_244 = frame_fab_bb[244].getCoords()
    tyr_245 = frame_fab_bb[245].getCoords()

    
    #CDRH2 loop residues 
    pro_266 = frame_fab_bb[266].getCoords()
    leu_267 = frame_fab_bb[267].getCoords()
    leu_268 = frame_fab_bb[268].getCoords()
    thr_269 = frame_fab_bb[269].getCoords()

    #CDRH3 loop residues 
    gly_316 = frame_fab_bb[316].getCoords()
    trp_317 = frame_fab_bb[317].getCoords() 
    gly_318 = frame_fab_bb[318].getCoords()
    trp_319 = frame_fab_bb[319].getCoords()  
    leu_320 = frame_fab_bb[320].getCoords() 
    gly_321 = frame_fab_bb[321].getCoords()
    lys_322 = frame_fab_bb[322].getCoords()
    pro_323 = frame_fab_bb[323].getCoords()
    ile_324 = frame_fab_bb[324].getCoords()
    
    #list of crd loop residue coords 
    cdr_loop_res_coords = [ser_027, val_028, gly_029, asn_030, asn_031, 
                           tyr_049, gly_050, ala_051 ,ser_052, ser_053, arg_054, pro_055, 
                           tyr_091, gly_092,  gln_093, ser_094, leu_095, ser_096,
                           gly_239, gly_240, ser_241, phe_242, ser_243, tyr_245, 
                           pro_266, leu_267, leu_268, thr_269,
                           gly_316, trp_317, gly_318, trp_319, leu_320, gly_321, lys_322, pro_323, ile_324]
    #get binary vector of crd loop membrane contacts - do not add to ft vect for clustering currently
    loop_contacts = findMemContacts(cdr_loop_res_coords, bot_leaf_plane)
    
    loop_contact_counts = countMemContacts(cdr_loop_res_coords, bot_leaf_plane)
    #print(loop_contact_counts)
    #add each loops' residue contact count to ft vect as a value 
    ft_vect.append(loop_contact_counts[0])
    ft_vect.append(loop_contact_counts[1])
    ft_vect.append(loop_contact_counts[2])
    ft_vect.append(loop_contact_counts[3])
    ft_vect.append(loop_contact_counts[4])
    ft_vect.append(loop_contact_counts[5])
    
################ CDR LOOP CENTER OF MASS (COM) DEPTHS - relative to PO4 plane ########################   
        #*remember- actual simulation is upside down, these calculations will be plotted so that Fab is on top of bilayer 
    #CDRL1 
    cdrl1_resis = frame_fab_bb.select('resnum 27 to 33 ')
    cdrl1_com_z = calcCenter(cdrl1_resis)[2]
    cdrl1_loop_COM_depth = avg_bot_phos_z-cdrl1_com_z
    #cdrl1_com_dist2po4_bot.append(avg_bot_phos_z-cdrl1_com_z)
    #cdrl1_com_dist2bilayer_cen.append(cdrl1_com_z-avg_bilayer_mp)
    
    #cdrl1_com_depth_aggregate.append(cdrl1_com[2])
    
    #CDRL2
    cdrl2_resis = frame_fab_bb.select('resnum 49 to 56')
    cdrl2_com_z = calcCenter(cdrl2_resis)[2]
    cdrl2_loop_COM_depth = avg_bot_phos_z-cdrl2_com_z
    #cdrl2_com_dist2po4_bot.append(avg_bot_phos_z-cdrl2_com_z)

    #cdrl2_com_dist2bilayer_cen.append(cdrl2_com_z-avg_bilayer_mp)
    
    #CDRL3
    cdrl3_resis = frame_fab_bb.select('resnum 91 to 98')
    cdrl3_com_z = calcCenter(cdrl3_resis)[2]
    cdrl3_loop_COM_depth = avg_bot_phos_z-cdrl3_com_z
    #cdrl3_com_dist2po4_bot.append(avg_bot_phos_z-cdrl3_com_z)

    #cdrl3_com_dist2bilayer_cen.append(avg_bot_phos_z-cdrl3_com_z)
    
    #CDRH1
    cdrh1_resis = frame_fab_bb.select('resnum 240 to 246')
    cdrh1_com_z = calcCenter(cdrh1_resis)[2]
    cdrh1_loop_COM_depth = avg_bot_phos_z-cdrh1_com_z
    #cdrh1_com_dist2po4_bot.append(avg_bot_phos_z-cdrh1_com_z)
    #cdrh1_com_dist2bilayer_cen.append(avg_bot_phos_z-cdrh1_com_z)
    
    #CDRH2 resdius 
    cdrh2_resis = frame_fab_bb.select('resnum 267 to 270')
    cdrh2_com_z = calcCenter(cdrh2_resis)[2]
    cdrh2_loop_COM_depth = avg_bot_phos_z-cdrh2_com_z
    #cdrh2_com_dist2po4_bot.append(avg_bot_phos_z-cdrh2_com_z)
    #cdrh2_com_dist2bilayer_cen.append(avg_bot_phos_z-cdrh2_com_z)
    
    #CDRH3 
    cdrh3_resis = frame_fab_bb.select('resnum 315 to 325')
    cdrh3_com_z = calcCenter(cdrh2_resis)[2]
    cdrh3_loop_COM_depth = avg_bot_phos_z-cdrh3_com_z
    #cdrh3_com_dist2po4_bot.append(avg_bot_phos_z-cdrh3_com_z)
    #cdrh3_com_dist2bilayer_cen.append(cdrh3_com_z-avg_bilayer_mp)

    ft_vect.append(round(cdrl1_loop_COM_depth, 2))
    ft_vect.append(round(cdrl2_loop_COM_depth, 2))
    ft_vect.append(round(cdrl3_loop_COM_depth, 2))
    ft_vect.append(round(cdrh1_loop_COM_depth, 2))
    ft_vect.append(round(cdrh2_loop_COM_depth, 2))
    ft_vect.append(round(cdrh3_loop_COM_depth, 2))
################ ALING NORM OF PO4 PLANE TO Z, calculate ANGLES ########################   

    #define as rotation b/w norm vect of PO4 plane & z-axis (unit normal of Z axis )
    rotation = VectorAlign(bot_leaf_plane[1], np.array([0, 0, 1]))
    #define transformation b/w roatino and origin 
    transformZax = Transformation(rotation, np.zeros(3))
    #apply transofrmation of plane to Z axis to entire system 
    applyTransformation(transformZax, frame.getAtoms())
    
    frame_fab = frame.getAtoms().select( fab_selection_str )

    #define selections for points used to make vectors through fab 
    
    #center of loops is p2 for all angles 
    cen_loops = frame_fab_bb.select('resnum 41 or resnum 256')
    fab_cen = calcCenter(cen_loops)
    
    cdrh_points = frame_fab_bb.select('resnum 243 or resnum 269 or resnum 319')
    cdrl_points = frame_fab_bb.select('resnum 31 or resnum 53 or resnum 95')
    crd_loops = cdrh_points + cdrl_points    
    cdr_loops_cen = calcCenter(crd_loops)
    
    fab_tip = frame_fab_bb.select('resnum 213 or resnum 41')
    fab_tip_cen = calcCenter(fab_tip)
 
    po4_plan_norm = -1*bot_leaf_plane[1] #second vector is norm of PO4 plane 
    
    #var_dom_long_axis 
    #define vector through center of fab and center of cdr loops 
    var_dom_long_ax =  np.array(fab_cen-cdr_loops_cen)
    var_dom_long_ax_ang = angle(po4_plan_norm, var_dom_long_ax)

    #print('VAR DOM ANG: ', var_dom_long_ax_ang)
    #print('VAR DOM ANG: ',angle(po4_plan_norm, var_dom_long_ax))
    #full_fab_long_axis 
    #full_fab_long_ax =  np.array(fab_tip_cen - cdr_loops_cen)
    full_fab_long_ax =  np.array(fab_tip_cen - cdr_loops_cen)

    full_fab_long_ax_ang = angle(po4_plan_norm, full_fab_long_ax)

    #short_axis 
    l_chain_middle_edge = frame_fab_bb.select('resnum 107').getCoords()
    h_chain_middle_edge = frame_fab_bb.select('resnum 340').getCoords()
    short_axis_vect = np.array( h_chain_middle_edge[0] - l_chain_middle_edge[0] )
    #short_axis_vect = np.array( l_chain_middle_edge[0] -h_chain_middle_edge[0] ) - this should give opposite sign angle as vector above 
    short_ax_ang = angle(po4_plan_norm, short_axis_vect)
     
    #add angles to ft vector 
    #ft_vect.append(full_fab_long_ax_ang)
    ft_vect.append(var_dom_long_ax_ang)
    ft_vect.append(short_ax_ang)
    
    #add ft vect to aggregate lsit and to traj specific list 
    fts_4e10_aa.append(ft_vect)
    #fts_4e10_p15.append(ft_vect)


# In[40]:


#CELL 3
fab_selection_str = 'protein'
mem_selection_str = 'resname POPC POPA CHOL'


input_pdb = parsePDB('/home/bat-gpu/colleen/phos_interactions/final_traj/4e10_n15/analysis_input.pdb')
dcd = DCDFile('/home/bat-gpu/colleen/phos_interactions/final_traj/4e10_n15/complete_md.dcd')
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

#will be X frame length of ft vector 
#ft vector will contain 15 fts (in following order):
#cdrl1_loop_contact_counts
#cdrl2_loop_contact_counts
#cdrl3_loop_contact_counts
#cdrh1_loop_contact_counts
#cdrh2_loop_contact_counts
#cdrh1_loop_contact_counts
#cdrl1_loop_COM_depth
#cdrl2_loop_COM_depth
#cdrl3_loop_COM_depth
#cdrh1_loop_COM_depth
#cdrh2_loop_COM_depth
#cdrh3_loop_COM_depth
#full_fab_long_ax_ang 
#var_dom_long_ax_ang
#short_ax_ang

#fts_4e10_n15 = [] 

################ LOOP THROUGH TRAJ TO CALC FEATURE VECTORS ########################   
for i, frame in enumerate(dcd):

    ft_vect = []

    #define fab and membrane 
    frame_fab = frame.getAtoms().select( fab_selection_str )
    frame_fab_bb = frame.getAtoms().select( 'name CA' )
    frame_mem = frame.getAtoms().select( mem_selection_str )

    #DEFINE MEMBRANE LAYERS - bilayer midpoint, top PO4 plane, bottom PO4 plane 

    avg_bilayer_mp = (int(sum(list(frame_mem.getCoords()[:,2]))/len(frame_mem.getResnames())))

    mem_top_sel_str  = 'resname POPC name P and z > '+ str(avg_bilayer_mp)
    mem_bot_sel_str  = 'resname POPC name P and z < '+ str(avg_bilayer_mp)

    top_leaf = frame.getAtoms().select(mem_top_sel_str)
    top_leaf_points = top_leaf.getCoords()
    top_leaf_plane = planeFit(np.transpose(top_leaf_points))
    
    
    bot_leaf = frame.getAtoms().select(mem_bot_sel_str)
    bot_leaf_points = bot_leaf.getCoords()
    bot_leaf_plane = planeFit(np.transpose(bot_leaf_points))

    
    bot_phos_layer_sel_str  = 'resname POPC name P and z < ' +str(avg_bilayer_mp)
    phos = frame.getAtoms().select(bot_phos_layer_sel_str)
    phos_points = phos.getCoords()
    phos_z = phos_points[:,2]
    avg_bot_phos_z = sum(phos_z)/len(phos_z)
    bot_phos_layer_dist2bilayer_cen = avg_bot_phos_z - avg_bilayer_mp

    
    top_phos_layer_sel_str  = 'resname POPC name P and z > ' +str(avg_bilayer_mp)
    phos = frame.getAtoms().select(top_phos_layer_sel_str)
    phos_points = phos.getCoords()
    phos_z = phos_points[:,2]
    avg_top_phos_z = sum(phos_z)/len(phos_z)
    top_phos_layer_dist2bilayer_cen = avg_top_phos_z - avg_bilayer_mp
    
    
#     avg_bot_phos_dist2bilayer_cen.append(bot_phos_layer_dist2bilayer_cen)
#     avg_top_phos_dist2bilayer_cen.append(top_phos_layer_dist2bilayer_cen)
#     bilayer_cen_dist2bilayer_cen.append(avg_bilayer_mp-avg_bilayer_mp)
    
    
    ################ CDR LOOP RESIDUES - CONTACT MEMBRANE STATS########################   

    #CDRL1 loops residues 
    ser_027 = frame_fab_bb[27].getCoords()
    val_028 = frame_fab_bb[28].getCoords()  
    gly_029 = frame_fab_bb[29].getCoords()   
    asn_030 = frame_fab_bb[30].getCoords()
    asn_031 = frame_fab_bb[31].getCoords()

    #CDRL2 loops residues 
    tyr_049 = frame_fab_bb[49].getCoords()
    gly_050 = frame_fab_bb[50].getCoords()
    ala_051 = frame_fab_bb[51].getCoords()
    ser_052 = frame_fab_bb[52].getCoords()
    ser_053 = frame_fab_bb[53].getCoords()
    arg_054 = frame_fab_bb[54].getCoords()
    pro_055 = frame_fab_bb[55].getCoords()

    #CDRL3 loops residues   
    tyr_091 = frame_fab_bb[91].getCoords()
    gly_092 = frame_fab_bb[92].getCoords()
    gln_093 = frame_fab_bb[93].getCoords()
    ser_094 = frame_fab_bb[94].getCoords()
    leu_095 = frame_fab_bb[95].getCoords()
    ser_096 = frame_fab_bb[96].getCoords()

    #CDRH1 loop residues 
    gly_239 = frame_fab_bb[239].getCoords()
    gly_240 = frame_fab_bb[240].getCoords()
    ser_241 = frame_fab_bb[241].getCoords()  
    phe_242 = frame_fab_bb[242].getCoords()
    ser_243 = frame_fab_bb[243].getCoords()
    thr_244 = frame_fab_bb[244].getCoords()
    tyr_245 = frame_fab_bb[245].getCoords()

    
    #CDRH2 loop residues 
    pro_266 = frame_fab_bb[266].getCoords()
    leu_267 = frame_fab_bb[267].getCoords()
    leu_268 = frame_fab_bb[268].getCoords()
    thr_269 = frame_fab_bb[269].getCoords()

    #CDRH3 loop residues 
    gly_316 = frame_fab_bb[316].getCoords()
    trp_317 = frame_fab_bb[317].getCoords() 
    gly_318 = frame_fab_bb[318].getCoords()
    trp_319 = frame_fab_bb[319].getCoords()  
    leu_320 = frame_fab_bb[320].getCoords() 
    gly_321 = frame_fab_bb[321].getCoords()
    lys_322 = frame_fab_bb[322].getCoords()
    pro_323 = frame_fab_bb[323].getCoords()
    ile_324 = frame_fab_bb[324].getCoords()
    
    #list of crd loop residue coords 
    cdr_loop_res_coords = [ser_027, val_028, gly_029, asn_030, asn_031, 
                           tyr_049, gly_050, ala_051 ,ser_052, ser_053, arg_054, pro_055, 
                           tyr_091, gly_092,  gln_093, ser_094, leu_095, ser_096,
                           gly_239, gly_240, ser_241, phe_242, ser_243, tyr_245, 
                           pro_266, leu_267, leu_268, thr_269,
                           gly_316, trp_317, gly_318, trp_319, leu_320, gly_321, lys_322, pro_323, ile_324]
    #get binary vector of crd loop membrane contacts - do not add to ft vect for clustering currently
    loop_contacts = findMemContacts(cdr_loop_res_coords, bot_leaf_plane)
    
    loop_contact_counts = countMemContacts(cdr_loop_res_coords, bot_leaf_plane)
    #print(loop_contact_counts)
    #add each loops' residue contact count to ft vect as a value 
    ft_vect.append(loop_contact_counts[0])
    ft_vect.append(loop_contact_counts[1])
    ft_vect.append(loop_contact_counts[2])
    ft_vect.append(loop_contact_counts[3])
    ft_vect.append(loop_contact_counts[4])
    ft_vect.append(loop_contact_counts[5])
    
################ CDR LOOP CENTER OF MASS (COM) DEPTHS - relative to PO4 plane ########################   
       #*remember- actual simulation is upside down, these calculations will be plotted so that Fab is on top of bilayer 
    #CDRL1 
    cdrl1_resis = frame_fab_bb.select('resnum 27 to 33 ')
    cdrl1_com_z = calcCenter(cdrl1_resis)[2]
    cdrl1_loop_COM_depth = avg_bot_phos_z-cdrl1_com_z
    #cdrl1_com_dist2po4_bot.append(avg_bot_phos_z-cdrl1_com_z)
    #cdrl1_com_dist2bilayer_cen.append(cdrl1_com_z-avg_bilayer_mp)
    
    #cdrl1_com_depth_aggregate.append(cdrl1_com[2])
    
    #CDRL2
    cdrl2_resis = frame_fab_bb.select('resnum 49 to 56')
    cdrl2_com_z = calcCenter(cdrl2_resis)[2]
    cdrl2_loop_COM_depth = avg_bot_phos_z-cdrl2_com_z
    #cdrl2_com_dist2po4_bot.append(avg_bot_phos_z-cdrl2_com_z)

    #cdrl2_com_dist2bilayer_cen.append(cdrl2_com_z-avg_bilayer_mp)
    
    #CDRL3
    cdrl3_resis = frame_fab_bb.select('resnum 91 to 98')
    cdrl3_com_z = calcCenter(cdrl3_resis)[2]
    cdrl3_loop_COM_depth = avg_bot_phos_z-cdrl3_com_z
    #cdrl3_com_dist2po4_bot.append(avg_bot_phos_z-cdrl3_com_z)

    #cdrl3_com_dist2bilayer_cen.append(avg_bot_phos_z-cdrl3_com_z)
    
    #CDRH1
    cdrh1_resis = frame_fab_bb.select('resnum 240 to 246')
    cdrh1_com_z = calcCenter(cdrh1_resis)[2]
    cdrh1_loop_COM_depth = avg_bot_phos_z-cdrh1_com_z
    #cdrh1_com_dist2po4_bot.append(avg_bot_phos_z-cdrh1_com_z)
    #cdrh1_com_dist2bilayer_cen.append(avg_bot_phos_z-cdrh1_com_z)
    
    #CDRH2 resdius 
    cdrh2_resis = frame_fab_bb.select('resnum 267 to 270')
    cdrh2_com_z = calcCenter(cdrh2_resis)[2]
    cdrh2_loop_COM_depth = avg_bot_phos_z-cdrh2_com_z
    #cdrh2_com_dist2po4_bot.append(avg_bot_phos_z-cdrh2_com_z)
    #cdrh2_com_dist2bilayer_cen.append(avg_bot_phos_z-cdrh2_com_z)
    
    #CDRH3 
    cdrh3_resis = frame_fab_bb.select('resnum 315 to 325')
    cdrh3_com_z = calcCenter(cdrh2_resis)[2]
    cdrh3_loop_COM_depth = avg_bot_phos_z-cdrh3_com_z
    #cdrh3_com_dist2po4_bot.append(avg_bot_phos_z-cdrh3_com_z)
    #cdrh3_com_dist2bilayer_cen.append(cdrh3_com_z-avg_bilayer_mp)

    ft_vect.append(round(cdrl1_loop_COM_depth, 2))
    ft_vect.append(round(cdrl2_loop_COM_depth, 2))
    ft_vect.append(round(cdrl3_loop_COM_depth, 2))
    ft_vect.append(round(cdrh1_loop_COM_depth, 2))
    ft_vect.append(round(cdrh2_loop_COM_depth, 2))
    ft_vect.append(round(cdrh3_loop_COM_depth, 2))
################ ALING NORM OF PO4 PLANE TO Z, calculate ANGLES ########################   

    #define as rotation b/w norm vect of PO4 plane & z-axis (unit normal of Z axis )
    rotation = VectorAlign(bot_leaf_plane[1], np.array([0, 0, 1]))
    #define transformation b/w roatino and origin 
    transformZax = Transformation(rotation, np.zeros(3))
    #apply transofrmation of plane to Z axis to entire system 
    applyTransformation(transformZax, frame.getAtoms())
    
    frame_fab = frame.getAtoms().select( fab_selection_str )

    #define selections for points used to make vectors through fab 
    
    #center of loops is p2 for all angles 
    cen_loops = frame_fab_bb.select('resnum 41 or resnum 256')
    fab_cen = calcCenter(cen_loops)
    
    cdrh_points = frame_fab_bb.select('resnum 243 or resnum 269 or resnum 319')
    cdrl_points = frame_fab_bb.select('resnum 31 or resnum 53 or resnum 95')
    crd_loops = cdrh_points + cdrl_points    
    cdr_loops_cen = calcCenter(crd_loops)
    
    fab_tip = frame_fab_bb.select('resnum 213 or resnum 41')
    fab_tip_cen = calcCenter(fab_tip)
 
    po4_plan_norm = -1*bot_leaf_plane[1] #second vector is norm of PO4 plane 
    
    #var_dom_long_axis 
    #define vector through center of fab and center of cdr loops 
    var_dom_long_ax =  np.array(fab_cen-cdr_loops_cen)
    var_dom_long_ax_ang = angle(po4_plan_norm, var_dom_long_ax)

    #print('VAR DOM ANG: ', var_dom_long_ax_ang)
    #print('VAR DOM ANG: ',angle(po4_plan_norm, var_dom_long_ax))
    #full_fab_long_axis 
    #full_fab_long_ax =  np.array(fab_tip_cen - cdr_loops_cen)
    full_fab_long_ax =  np.array(fab_tip_cen - cdr_loops_cen)

    full_fab_long_ax_ang = angle(po4_plan_norm, full_fab_long_ax)

    #short_axis 
    l_chain_middle_edge = frame_fab_bb.select('resnum 107').getCoords()
    h_chain_middle_edge = frame_fab_bb.select('resnum 340').getCoords()
    short_axis_vect = np.array( h_chain_middle_edge[0] - l_chain_middle_edge[0] )
    #short_axis_vect = np.array( l_chain_middle_edge[0] -h_chain_middle_edge[0] ) - this should give opposite sign angle as vector above 
    short_ax_ang = angle(po4_plan_norm, short_axis_vect)
     
    #add angles to ft vector 
    #ft_vect.append(full_fab_long_ax_ang)
    ft_vect.append(var_dom_long_ax_ang)
    ft_vect.append(short_ax_ang)
    
    #add ft vect to aggregate lsit and to traj specific list 
    fts_4e10_aa.append(ft_vect)
    #fts_4e10_n15.append(ft_vect)


# In[41]:


init_T = time.time()
ssd_rmsd_4e10_aa = []
#print(type(aggregate_fts))
for pairwise in combinations((fts_4e10_aa), 2):
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    #print(rms)
    ssd_rmsd_4e10_aa.append(rms)
print (round( time.time() - init_T, 3), 's to complete rmsd calculation')
print ('mean distance:', round( np.mean(ssd_rmsd_4e10_aa), 1), '| StDev: ', round( np.std(ssd_rmsd_4e10_aa) , 1))


# In[42]:


f = open('4e10_aa_ssd_rmsd.txt', 'w')
for i in ssd_rmsd:
    f.write(str(i) + "\n")
f.close


# In[44]:


## use mean value of RMSD matrix as cut-off to define clusters
cutoff =  120 # #np.mean(ssd_rmsd)*1.5 3.5 = 11 clusters 
ssd_rmsd_4e10_aa = np.array( ssd_rmsd_4e10_aa )
init_T= time.time()

# Ward hierarchical clustering minimizes variance between clusters
# Complete linkage clustering makes sure all cluster members are within same RMSD cutoff to each other
linkMat= linkage( ssd_rmsd_4e10_aa , method='ward', metric='euclidean')
print (round( time.time() - init_T, 3), 's to complete clustering')

h_clust= fcluster( linkMat, cutoff, criterion='distance')
numClust= len( set(h_clust) )
print ('RMS cutoff at %.2f, Unique clusters found:' % cutoff, numClust, '\n')


# In[46]:


#calculate centroids for each cluster 
clust_01 = []
clust_01_frames = []
clust_02 = []
clust_02_frames = []
clust_03 = []
clust_03_frames = []
clust_04 = []
clust_04_frames = []
clust_05 = []
clust_05_frames = []
clust_06 = []
clust_06_frames = []
clust_07 = []
clust_07_frames = []
clust_08 = []
clust_08_frames = []
clust_09 = []
clust_09_frames = []
for i in range(len(fts_4e10_aa)):
    if h_clust[i] == 1:
        clust_01_frames.append(i)
        clust_01.append(fts_4e10_aa[i])
    elif h_clust[i] == 2:
        clust_02_frames.append(i)
        clust_02.append(fts_4e10_aa[i])
    elif h_clust[i] == 3:
        clust_03_frames.append(i)
        clust_03.append(fts_4e10_aa[i])
    elif h_clust[i] == 4:
        clust_04_frames.append(i)
        clust_04.append(fts_4e10_aa[i])
    elif h_clust[i] == 5:
        clust_05_frames.append(i)
        clust_05.append(fts_4e10_aa[i])
    elif h_clust[i] == 6:
        clust_06_frames.append(i)
        clust_06.append(fts_4e10_aa[i])
    elif h_clust[i] == 7:
        clust_07_frames.append(i)
        clust_07.append(fts_4e10_aa[i])
    elif h_clust[i] == 8:
        clust_08_frames.append(i)
        clust_08.append(fts_4e10_aa[i])
    elif h_clust[i] == 9:
        clust_09_frames.append(i)
        clust_09.append(fts_4e10_aa[i])


# re-do kmedoids clustering to find cluster, including re-calculating dist matrix bc in isnt too long
ssd_rmsd_aa_1 = []
for n, pair in enumerate( combinations( clust_01, 2 ) ):
    #superpose( *pair )
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_rmsd_aa_1.append(rms)
ssd_rmsd_aa_1 = np.array( ssd_rmsd_aa_1 )
k_clust, cost,num	= kmedoids( ssd_rmsd_aa_1, nclusters=1, npass=100)
centroid_ID = list(set(k_clust))[0]
print("centroid ID: " + str(centroid_ID))
print('representative frame: ' + str(clust_01_frames[centroid_ID]))
print("cluster state: " + str(h_clust[clust_01_frames[centroid_ID]]))

ssd_rmsd_aa_2 = []
for n, pair in enumerate( combinations( clust_02, 2 ) ):
    #superpose( *pair )
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_rmsd_aa_2.append(rms)
ssd_rmsd_aa_2 = np.array( ssd_rmsd_aa_2 )
k_clust, cost,num	= kmedoids( ssd_rmsd_aa_2, nclusters=1, npass=100)
centroid_ID = list(set(k_clust))[0]
print("centroid ID: " + str(centroid_ID))
print('representative frame: ' + str(clust_02_frames[centroid_ID]))
print("cluster state: " + str(h_clust[clust_02_frames[centroid_ID]]))

ssd_rmsd_aa_3 = []
for n, pair in enumerate( combinations( clust_03, 2 ) ):
    #superpose( *pair )
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_rmsd_aa_3.append(rms)
ssd_rmsd_aa_3 = np.array( ssd_rmsd_aa_3 )
k_clust, cost,num	= kmedoids( ssd_rmsd_aa_3, nclusters=1, npass=100)
centroid_ID = list(set(k_clust))[0]
print("centroid ID: " + str(centroid_ID))
print('representative frame: ' + str(clust_03_frames[centroid_ID]))
print("cluster state: " + str(h_clust[clust_03_frames[centroid_ID]]))

ssd_rmsd_aa_4 = []
for n, pair in enumerate( combinations( clust_04, 2 ) ):
    #superpose( *pair )
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_rmsd_aa_4.append(rms)
ssd_rmsd_aa_4 = np.array( ssd_rmsd_aa_4 )
k_clust, cost,num	= kmedoids( ssd_rmsd_aa_4, nclusters=1, npass=100)
centroid_ID = list(set(k_clust))[0]
print("centroid ID: " + str(centroid_ID))
print('representative frame: ' + str(clust_04_frames[centroid_ID]))
print("cluster state: " + str(h_clust[clust_04_frames[centroid_ID]]))


# In[47]:


#all atom cluster centroids 
AA_clust_id_1 = fts_4e10_aa[1640] #frame 1640  

AA_clust_id_2 = fts_4e10_aa[0]  #frame 0  

AA_clust_id_3 =  fts_4e10_aa[346] #frame 346 

AA_clust_id_4 = fts_4e10_aa[65] #frame 65 

print(AA_clust_id_1)
print(AA_clust_id_2)
print(AA_clust_id_3)
print(AA_clust_id_4)
# AA_clust_id_5 = [0, 0, 0, 2, 2, 8, 7.84, 6.85, 10.69, 2.38, 1.77, 1.77, 70.07, 11.64]  #frame 44  

# AA_clust_id_6 = [1, 0, 0, 5, 2, 9, 4.8, 4.44, 8.54, 0.58, 0.71, 0.71, 73.72, 12.38] #frame 0  

# AA_clust_id_7 =  [0, 0, 0, 1, 3, 2, 7.74, 12.08, 9.65, 4.23, -0.1, -0.1, 76.55441433944779, -3.880694546398203] #frame 4881 

# AA_clust_id_8 =  [0, 0, 0, 3, 2, 7, 11.8, 11.25, 13.43, 1.4, 0.79, 0.79, 83.47, -1.67]  #frame 1522  

# AA_clust_id_9 = [0, 0, 0, 5, 2, 9, 7.71, 6.5, 11.06, 0.49, 1.29, 1.29, 77.31, 8.15] #frame 73  


# AA_clust_centroids = [AA_clust_id_1, AA_clust_id_2, AA_clust_id_3, AA_clust_id_4, 
#                      AA_clust_id_5, AA_clust_id_6, AA_clust_id_7, AA_clust_id_8, 
#                      AA_clust_id_9]


# In[68]:


#cetroid correlation - 
from scipy.spatial import distance_matrix
import seaborn as sns
import matplotlib.pylab as plt

print(AA_clust_id_1)
print(cg_clust_id_1)
centroids = [AA_clust_id_1, AA_clust_id_2, AA_clust_id_3, AA_clust_id_4,  cg_clust_id_1, cg_clust_id_2, cg_clust_id_3] #, cg_clust_id_4
centroid_distances = distance_matrix(centroids, centroids ) 


ax = sns.heatmap(centroid_distances, linewidth=0.5)
plt.show()


# In[34]:


cg_centroid_distances = distance_matrix(cg_clust_centroids, cg_clust_centroids ) 


ax = sns.heatmap(cg_centroid_distances, linewidth=0.5)
plt.show()


# In[36]:


total_centroids =[cg_clust_id_1, cg_clust_id_2, cg_clust_id_3, cg_clust_id_4, 
                     cg_clust_id_5, cg_clust_id_6, cg_clust_id_7, cg_clust_id_8, 
                     cg_clust_id_9, AA_clust_id_1, AA_clust_id_2, AA_clust_id_3, AA_clust_id_4, 
                     AA_clust_id_5, AA_clust_id_6, AA_clust_id_7, AA_clust_id_8, 
                     AA_clust_id_9 ]
total_cent_distances = distance_matrix(total_centroids, total_centroids ) 


ax = sns.heatmap(total_cent_distances, linewidth=0.5)
plt.show()


# In[48]:


print(total_centroids[6])
print(total_centroids[9])


# In[35]:


#rint(centroid_distances)
for i in cg_clust_centroids:
    print(i)
for i in AA_clust_centroids:
    print(i)    


# In[31]:


fab_selection_str = 'name BB SC1 SC2 SC3 SC4'
mem_selection_str = 'resname POPC POPA CHOL'

input_pdb = parsePDB('/Volumes/MyPassport/mravic_rotation/coarse_grain/dcd_traj/051_input.pdb')  #'../../B1.2/NPT_equilx_sys.pdb'

dcd = DCDFile('/Volumes/MyPassport/mravic_rotation/coarse_grain/dcd_traj/mem_contact_frames_skip10.dcd')#'../../B1.2/b1.2_intermediate'
dcd.setCoords(input_pdb)
dcd.link(input_pdb)
dcd.reset()

features=[0]*len(dcd) 


################ LOOP THROUGH TRAJ TO CALC FEATURE VECTORS ########################   

for i, frame in enumerate(dcd):

    ft_vect = []


    #define fab and membrane 
    frame_fab = frame.getAtoms().select( fab_selection_str )
    frame_fab_bb = frame.getAtoms().select( 'name BB' )
    frame_mem = frame.getAtoms().select( mem_selection_str )

    #calculate top and bottom membrane leaflets  


    #calculate top and bottom membrane leaflets  
    mp = (int(sum(list(frame_mem.getCoords()[:,2]))/8270))

    mem_top_sel_str  = 'resname POPC name PO4 and z > ' + str(mp)
    mem_bot_sel_str  = 'resname POPC name PO4 and z < ' + str(mp)

    top_leaf = frame.getAtoms().select(mem_top_sel_str)
    top_leaf_points = top_leaf.getCoords()
    top_leaf_plane = planeFit(np.transpose(top_leaf_points))

    bot_leaf = frame.getAtoms().select(mem_bot_sel_str)
    bot_leaf_points = bot_leaf.getCoords()

    bot_leaf_plane = planeFit(np.transpose(bot_leaf_points))



    
################ CENTER OF FAB - DISTNACE TO MEM ########################   
    frame_fab_cen = calcCenter(frame_fab_bb)
    fab_cen_dist = dist2Plane(frame_fab_cen, bot_leaf_plane) #calc to bottom plane b/c this sim is flipped 
    #fab_cen_dist2mem_BOT_by_frame[i] = fab_cen_dist
    

    
################ ******** FEATURE ******** BOUND POTENTIAL ########################   
    print(fab_cen_dist)
    if abs(fab_cen_dist)<80:
        bound_potential = 0 #true
    else:
        bound_potential = 1 #false 
    ft_vect.append(bound_potential*5)
################ CDR LOOP RESIDUES - DISTNACE TO MEM ########################   
    
#     cdrl1= calcCenter(frame_fab_bb[27:31]) #.getCoords()
#     cdrl1_dist = dist2Plane(cdrl1, top_leaf_plane)
    
#     if cdrl1_dist<6: 
#         cdrl1_touch= 1 
#     else: 
#         cdrl1_touch = 0 
#     ft_vect.append(cdrl1_touch) 
    
#     cdrl2= calcCenter(frame_fab_bb[49:55]) #.getCoords()
#     cdrl2_dist = dist2Plane(cdrl2, top_leaf_plane)
    
#     if cdrl2_dist<6: 
#         cdrl2_touch= 1 
#     else: 
#         cdrl2_touch = 0 
#     ft_vect.append(cdrl2_touch) 
    
#     cdrl3= calcCenter(frame_fab_bb[91:96]) #.getCoords()
#     cdrl3_dist = dist2Plane(cdrl3, top_leaf_plane)

#     if cdrl2_dist<6: 
#         cdrl2_touch= 1 
#     else: 
#         cdrl2_touch = 0 
#     ft_vect.append(cdrl2_touch) 
    

#     cdrh1= calcCenter(frame_fab_bb[239:245]) #.getCoords()
#     cdrh1_dist = dist2Plane(cdrh1, top_leaf_plane)
    
#     if cdrh1_dist<6: 
#         cdrh1_touch= 1 
#     else: 
#         cdrh1_touch = 0 
#     ft_vect.append(cdrh1_touch)
    
    
#     cdrh2= calcCenter(frame_fab_bb[266:269]) #.getCoords()
#     cdrh2_dist = dist2Plane(cdrh2, top_leaf_plane)
    
#     if cdrh2_dist<6: 
#         cdrh2_touch= 1 
#     else: 
#         cdrh2_touch = 0 
#     ft_vect.append(cdrh2_touch)
    
#     cdrh3= calcCenter(frame_fab_bb[316:324]) #.getCoords()
#     cdrh3_dist = dist2Plane(cdrh3, top_leaf_plane)
    
#     if cdrh3_dist<6: 
#         cdrh3_touch= 1 
#     else: 
#         cdrh3_touch = 0 
#     ft_vect.append(cdrh3_touch)
    
    
#     #CDRL1 loops residues 
    ser_027 = frame_fab_bb[27].getCoords()
    ser_027_dist = dist2Plane(ser_027, top_leaf_plane)
    
    if ser_027_dist<4: 
        ser_027_touch= 1 
    else: 
        ser_027_touch = 0 
    ft_vect.append(ser_027_touch)   
    
    
    val_028 = frame_fab_bb[28].getCoords()
    val_028_dist = dist2Plane(val_028, top_leaf_plane)
   
    if val_028_dist<4: 
        val_028_touch= 1 
    else: 
        val_028_touch = 0 
    ft_vect.append(val_028_touch)   
    
    gly_029 = frame_fab_bb[29].getCoords()
    gly_029_dist = dist2Plane(gly_029, top_leaf_plane)
   
    if gly_029_dist<4: 
        gly_029_touch= 1 
    else: 
        gly_029_touch = 0 
    ft_vect.append(gly_029_touch)   
    
    asn_030 = frame_fab_bb[30].getCoords()
    asn_030_dist = dist2Plane(asn_030, top_leaf_plane)

    if asn_030_dist<4: 
        asn_030_touch= 1 
    else: 
        asn_030_touch = 0 
    ft_vect.append(asn_030_touch)   
    
    asn_031 = frame_fab_bb[31].getCoords()
    asn_031_dist = dist2Plane(asn_031, top_leaf_plane)
   
    if asn_031_dist<4: 
        asn_031_touch= 1 
    else: 
        asn_031_touch = 0 
    ft_vect.append(asn_031_touch)    
    
    #CDRL2 loops residues 
    tyr_049 = frame_fab_bb[49].getCoords()
    tyr_049_dist = dist2Plane(tyr_049, top_leaf_plane)
    
    if tyr_049_dist<4: 
        tyr_049_touch= 1 
    else: 
        tyr_049_touch = 0 
    ft_vect.append(tyr_049_touch)
    
    gly_050 = frame_fab_bb[50].getCoords()
    gly_050_dist = dist2Plane(gly_050, top_leaf_plane)
    
    if gly_050_dist<4: 
        gly_050_touch= 1 
    else: 
        gly_050_touch = 0 
    ft_vect.append(gly_050_touch)
    
    ala_051 = frame_fab_bb[51].getCoords()
    ala_051_dist = dist2Plane(ala_051, top_leaf_plane)

    if ala_051_dist<4: 
        ala_051_touch= 1 
    else: 
        ala_051_touch = 0 
    ft_vect.append(ala_051_touch)
    
    ser_052 = frame_fab_bb[52].getCoords()
    ser_052_dist = dist2Plane(ser_052, top_leaf_plane)
    
    if ser_052_dist<4: 
        ser_052_touch= 1 
    else: 
        ser_052_touch = 0 
    ft_vect.append(ser_052_touch)
    
    ser_053 = frame_fab_bb[53].getCoords()
    ser_053_dist = dist2Plane(ser_053, top_leaf_plane)

    if ser_053_dist<4: 
        ser_053_touch= 1 
    else: 
        ser_053_touch = 0 
    ft_vect.append(ser_053_touch)
    
    arg_054 = frame_fab_bb[54].getCoords()
    arg_054_dist = dist2Plane(arg_054, top_leaf_plane)

    if arg_054_dist<4: 
        arg_054_touch= 1 
    else: 
        arg_054_touch = 0 
    ft_vect.append(arg_054_touch)   
    
    pro_055 = frame_fab_bb[55].getCoords()
    pro_055_dist = dist2Plane(pro_055, top_leaf_plane)

    if pro_055_dist<4: 
        pro_055_touch= 1 
    else: 
        pro_055_touch = 0 
    ft_vect.append(pro_055_touch)   
    #CDRL3 loops residues 
        
    tyr_091 = frame_fab_bb[91].getCoords()
    tyr_091_dist = dist2Plane(tyr_091, top_leaf_plane)
        
    if tyr_091_dist<4: 
        tyr_091_touch= 1 
    else: 
        tyr_091_touch = 0 
    ft_vect.append(tyr_091_touch)   
    
    gly_092 = frame_fab_bb[92].getCoords()
    gly_092_dist = dist2Plane(gly_092, top_leaf_plane)
    
    if gly_092_dist<4: 
        gly_092_touch= 1 
    else: 
        gly_092_touch = 0 
    ft_vect.append(gly_092_touch)   
    
    
    gln_093 = frame_fab_bb[93].getCoords()
    gln_093_dist = dist2Plane(gln_093, top_leaf_plane)
   
    if gln_093_dist<4: 
        gln_093_touch= 1 
    else: 
        gln_093_touch = 0 
    ft_vect.append(gln_093_touch)   
    
    ser_094 = frame_fab_bb[94].getCoords()
    ser_094_dist = dist2Plane(ser_094, top_leaf_plane)   
    
    
    if ser_094_dist<4: 
        ser_094_touch= 1 
    else: 
        ser_094_touch = 0 
    ft_vect.append(ser_094_touch)
    
        
    leu_095 = frame_fab_bb[95].getCoords()
    leu_095_dist = dist2Plane(leu_095, top_leaf_plane) 
    
    if leu_095_dist<4: 
        leu_095_touch= 1 
    else: 
        leu_095_touch = 0 
    ft_vect.append(leu_095_touch)
    

    ser_096 = frame_fab_bb[96].getCoords()
    ser_096_dist = dist2Plane(ser_096, top_leaf_plane) 
    
        
    if ser_096_dist<4: 
        ser_096_touch= 1 
    else: 
        ser_096_touch = 0 
    ft_vect.append(ser_096_touch) 



    #CDRH1 loop residues 
    gly_239 = frame_fab_bb[239].getCoords()
    gly_239_dist = dist2Plane(gly_239, top_leaf_plane) 
    
    if gly_239_dist<4: 
        gly_239_touch= 1 
    else: 
        gly_239_touch = 0 
    ft_vect.append(gly_239_touch) 
    
    gly_240 = frame_fab_bb[240].getCoords()
    gly_240_dist = dist2Plane(gly_240, top_leaf_plane)
   
    if gly_240_dist<4: 
        gly_240_touch= 1 
    else: 
        gly_240_touch = 0 
    ft_vect.append(gly_240_touch) 
    
    
    ser_241 = frame_fab_bb[241].getCoords()
    ser_241_dist = dist2Plane(ser_241, top_leaf_plane)

    if ser_241_dist<4: 
        ser_241_touch= 1 
    else: 
        ser_241_touch = 0 
    ft_vect.append(ser_241_touch) 
    
    phe_242 = frame_fab_bb[242].getCoords()
    phe_242_dist = dist2Plane(phe_242, top_leaf_plane)
    
    if phe_242_dist<4: 
        phe_242_touch= 1 
    else: 
        phe_242_touch = 0 
    ft_vect.append(phe_242_touch) 
    
    ser_243 = frame_fab_bb[243].getCoords()
    ser_243_dist = dist2Plane(ser_243, top_leaf_plane)

    if ser_243_dist<4: 
        ser_243_touch= 1 
    else: 
        ser_243_touch = 0 
    ft_vect.append(ser_243_touch) 
    
    thr_244 = frame_fab_bb[244].getCoords()
    thr_244_dist = dist2Plane(thr_244, top_leaf_plane)
    
    if thr_244_dist<4: 
        thr_244_touch= 1 
    else: 
        thr_244_touch = 0 
    ft_vect.append(thr_244_touch) 
    
    tyr_245 = frame_fab_bb[245].getCoords()
    tyr_245_dist = dist2Plane(tyr_245, top_leaf_plane)
    
    if tyr_245_dist<4: 
        tyr_245_touch= 1 
    else: 
        tyr_245_touch = 0 
    ft_vect.append(tyr_245_touch)

    
    #CDRH2 loop residues 
    pro_266 = frame_fab_bb[266].getCoords()
    pro_266_dist = dist2Plane(pro_266, top_leaf_plane)
    
    if pro_266_dist<4: 
        pro_266_touch= 1 
    else: 
        pro_266_touch = 0 
    ft_vect.append(pro_266_touch) 
    
    leu_267 = frame_fab_bb[267].getCoords()
    leu_267_dist = dist2Plane(leu_267, top_leaf_plane)

    if leu_267_dist<4: 
        leu_267_touch= 1 
    else: 
        leu_267_touch = 0 
    ft_vect.append(leu_267_touch) 
    
    leu_268 = frame_fab_bb[268].getCoords()
    leu_268_dist = dist2Plane(leu_268, top_leaf_plane)

    if leu_268_dist<4: 
        leu_268_touch= 1 
    else: 
        leu_268_touch = 0 
    ft_vect.append(leu_268_touch) 
    
    
    thr_269 = frame_fab_bb[269].getCoords()
    thr_269_dist = dist2Plane(thr_269, top_leaf_plane)
    
    if thr_269_dist<4: 
        thr_269_touch= 1 
    else: 
        thr_269_touch = 0 
    ft_vect.append(thr_269_touch)
        
    #CDRH3 loop residues 
    gly_316 = frame_fab_bb[316].getCoords()
    gly_316_dist = dist2Plane(gly_316, top_leaf_plane)
    
    if gly_316_dist<4: 
        gly_316_touch= 1 
    else: 
        gly_316_touch = 0 
    ft_vect.append(gly_316_touch)
    
    trp_317 = frame_fab_bb[317].getCoords()
    trp_317_dist = dist2Plane(trp_317, top_leaf_plane)
    print(trp_317_dist)
    if trp_317_dist<4 and trp_317_dist>-15: 
        trp_317_touch = 1 
    else: 
        trp_317_touch = 0 
    print(trp_317_touch)
    ft_vect.append(trp_317_touch)
    
    gly_318 = frame_fab_bb[318].getCoords()
    gly_318_dist = dist2Plane(gly_318, top_leaf_plane)

    if gly_318_dist<4: 
        gly_318_touch = 1 
    else: 
        gly_318_touch = 0 
    ft_vect.append(gly_318_touch)
    
    trp_319 = frame_fab_bb[319].getCoords()
    trp_319_dist = dist2Plane(trp_319, top_leaf_plane)

    if trp_319_dist<4 and trp_319_dist>-15: 
        trp_319_touch = 1 
    else: 
        trp_319_touch = 0 
    ft_vect.append(trp_319_touch)
    
    leu_320 = frame_fab_bb[320].getCoords()
    leu_320_dist = dist2Plane(leu_320, top_leaf_plane)

    if leu_320_dist<4: 
        leu_320_touch = 1 
    else: 
        leu_320_touch = 0 
    ft_vect.append(leu_320_touch)
    
    gly_321 = frame_fab_bb[321].getCoords()
    gly_321_dist = dist2Plane(gly_321, top_leaf_plane)
    
    if gly_321_dist<4: 
        gly_321_touch = 1 
    else: 
        gly_321_touch = 0 
    ft_vect.append(gly_321_touch)
    
    lys_322 = frame_fab_bb[322].getCoords()
    lys_322_dist = dist2Plane(lys_322, top_leaf_plane)
    
    if lys_322_dist<4: 
        lys_322_touch = 1 
    else: 
        lys_322_touch = 0 
    ft_vect.append(lys_322_touch)
    
    pro_323 = frame_fab_bb[323].getCoords()
    pro_323_dist = dist2Plane(pro_323, top_leaf_plane)
    
    if pro_323_dist<4: 
        pro_323_touch = 1 
    else: 
        pro_323_touch = 0 
    ft_vect.append(pro_323_touch)
    
    
    ile_324 = frame_fab_bb[324].getCoords()
    ile_324_dist = dist2Plane(ile_324, top_leaf_plane)
    

    if ile_324_dist<4: 
        ile_324_touch = 1 
    else: 
        ile_324_touch = 0 
    ft_vect.append(ile_324_touch)
        
################ ******** FEATURE ******** CDR LOOP RESIDUES - CONTACT ########################   


    cen_loops = frame_fab_bb.select('resnum 40 or resnum 254')
        #[40 or 254] #.getCoords()
    #print(cen_loopL)
    #cen_loopH = frame_fab_bb[] #.getCoords()
    
    #print(cen_loopH)
    p2 = calcCenter(cen_loops) #.select(324 ).getCoords()
    
    #p2 = frame_fab_bb[213].getCoords() #L chain C termini 
    #res28 = fab_bb[213].getCoords()
    ##other point 
    #print(fab_bb[319].getResname()) 
    p3 = frame_fab_bb[319].getCoords() #Tip of CRDH3 loop 
    p1 = np.array([p2[0], p2[1], 0]) #point directly center loops 
    #ang1 = 90-getAngle(p1, p2, p3)
    #ft_vect.append(ang1)
    #print(ang1)

    features[i]=ft_vect 

######################### ANGEL OF FAB TO MEMBRANE ################################    
    #print(fab_bb[213].getResname())
    #p2 = frame_fab_bb[213].getCoords() #L chain C termini 
    #res28 = fab_bb[213].getCoords()
    ##other point 
    #print(fab_bb[319].getResname()) 
    #p3 = frame_fab_bb[319].getCoords() #Tip of CRDH3 loop 
    #p1 = np.array([p2[0], p2[1], 0]) #point directly below L chain termini
    #ang1 = 90-getAngle(p1, p2, p3)
    #print(p1)
    #print(p2)
    #print(p3)
    #print(ang1)
    #point three - intersection with top leaflet print(top_leaf_plane)

    #angle_by_frame[i]=ang1

    
    
    
    #ft_vect = [dist_H1, dist_H2, dist_H3, bound_potential*5, dist_L1, dist_L3, ang1]
    #ft_vect = [dist_H1_top, dist_H2_top, dist_H3_top, dist_L1_top, dist_L3_top, fab_cen_dist]
    #ang1,
    #[H1_loop_dist2mem_by_frame, H2_loop_dist2mem_by_frame, H3_loop_dist2mem_by_frame,
    #          L1_loop_dist2mem_BOT_by_frame, fab_cen_dist2mem_by_frame, angle_by_frame]
    #print(ft_vect)
    #features[i]=ft_vect 
    
    
    
######################### PHOSPHATE GROUP INTERACTIONS #########################
#calculate distances of phosphates from CDRH1 loop residues 
#     res28_2_phos_distances = [] 
#     res29_2_phos_distances = [] 
#     res30_2_phos_distances = [] 
#     phosphates = frame_mem.select("name P")
#     for p in phosphates:
#         p_point=p.getCoords()
#         res28_dist2phos = np.linalg.norm(p_point-res28)
#         res28_2_phos_distances.append(res28_dist2phos)
#         res29_dist2phos = np.linalg.norm(p_point-res29)
#         res29_2_phos_distances.append(res29_dist2phos)
#         res30_dist2phos = np.linalg.norm(p_point-res30)
#         res30_2_phos_distances.append(res29_dist2phos)
# #calculate how many phosphates are interacting with CDRH1 residues
#     counter = 0 
#     for l in res28_2_phos_distances:
#         if abs(l)<6:
#             counter = counter+1
#     res28_phos_contacts_by_frame[i] = counter

#     counter = 0 
#     for m in res29_2_phos_distances:
#         if abs(m)<6:
#             counter = counter+1
#     res29_phos_contacts_by_frame[i] = counter

#     counter = 0
#     for n in res30_2_phos_distances:
#         if abs(n)<6:
#             counter = counter+1
#     res30_phos_contacts_by_frame[i] = counter
    
#     if abs(fab_cen_dist)<40:
#         bound_potential = 0 #true
#     else:
#         bound_potential = 1 #false 
#     print(fab_cen_dist)
    
    
    


    #print("frame analysis complete: ", i)


# In[ ]:





# In[32]:


#print(len(features))
#print(features[1])


# In[33]:


init_T = time.time()
ssd_rmsd = []
for pairwise in combinations(features, 2):
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_rmsd.append(rms)
print (round( time.time() - init_T, 3), 's to complete rmsd calculation')
print ('mean distance:', round( np.mean(ssd_rmsd), 1), '| StDev: ', round( np.std(ssd_rmsd) , 1))


# In[34]:


## use mean value of RMSD matrix as cut-off to define clusters
cutoff = 2.5 #np.mean(ssd_rmsd)*8 #1.52 cutoff for traj 051 gives 8 clusters 
ssd_rmsd = np.array( ssd_rmsd )
init_T= time.time()

# Ward hierarchical clustering minimizes variance between clusters
# Complete linkage clustering makes sure all cluster members are within same RMSD cutoff to each other
linkMat= linkage( ssd_rmsd , method='ward', metric='euclidean')
print (round( time.time() - init_T, 3), 's to complete clustering')

h_clust= fcluster( linkMat, cutoff, criterion='distance')
numClust= len( set(h_clust) )
print ('RMS cutoff at %.2f, Unique clusters found:' % cutoff, numClust, '\n')


# In[9]:


print(features[56]) #clust01
print(features[63]) #clust02
print(features[2]) #clust03
print(features[0]) #clust04
print(features[4]) #clust05
print(features[150]) #clust06
print(features[1]) #clust07
print(features[477]) #clust08
print(features[475]) #clust09
print(features[187]) #clust10
print(features[174]) #clust11
print(features[38]) #clust12
print(features[39]) #clust13


# In[52]:


import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
colors = ['tomato', 'darkorange', 'gold', 'tab:purple', 'tab:green', 'powderblue', 'mediumaquamarine', 'slateblue', 'orchid', 'pink', 'lightseagreen', 'hotpink', 'darkblue']
shc.set_link_color_palette(colors)

dend = shc.dendrogram(shc.linkage(ssd_rmsd, method='ward'), color_threshold=2.5, above_threshold_color='grey')
plt.axhline(y=2.5, color='r', linestyle='--') 


# In[15]:


clust_01 = []
clust_01_frames = []
for i in range(len(features)):
    if h_clust[i] == 1:
        #print(i)
        clust_01_frames.append(i)
        clust_01.append(features[i])
        #clust_01.append( dcd[i].getCoords() )


# re-do kmedoids clustering to find cluster, including re-calculating dist matrix bc in isnt too long
ssd_rmsd = []
for n, pair in enumerate( combinations( clust_01, 2 ) ):
    #superpose( *pair )
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_rmsd.append(rms)


ssd_rmsd = np.array( ssd_rmsd )
k_clust, cost,num	= kmedoids( ssd_rmsd, nclusters=1, npass=100)
centroid_ID = list(set(k_clust))[0]
print("centroid ID: " + str(centroid_ID))
print('representative frame: ' + str(clust_01_frames[centroid_ID]))
print("cluster state: " + str(h_clust[clust_01_frames[centroid_ID]]))


# In[16]:



clust_02 = []
clust_02_frames = []
for i in range(len(features)):
    if h_clust[i] == 2:
        #print(i)
        clust_02_frames.append(i)
        clust_02.append(features[i])
        #clust_01.append( dcd[i].getCoords() )


# re-do kmedoids clustering to find cluster, including re-calculating dist matrix bc in isnt too long
ssd_rmsd = []
for n, pair in enumerate( combinations( clust_02, 2 ) ):
    #superpose( *pair )
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_rmsd.append(rms)


ssd_rmsd = np.array( ssd_rmsd )
k_clust, cost,num	= kmedoids( ssd_rmsd, nclusters=1, npass=100)
centroid_ID = list(set(k_clust))[0]
print("centroid ID: " + str(centroid_ID))
print('representative frame: ' + str(clust_02_frames[centroid_ID]))
print("cluster state: " + str(h_clust[clust_02_frames[centroid_ID]]))


# In[17]:



clust_03 = []
clust_03_frames = []
for i in range(len(features)):
    if h_clust[i] == 3:
        #print(i)
        clust_03_frames.append(i)
        clust_03.append(features[i])
        #clust_01.append( dcd[i].getCoords() )


# re-do kmedoids clustering to find cluster, including re-calculating dist matrix bc in isnt too long
ssd_rmsd = []
for n, pair in enumerate( combinations( clust_03, 2 ) ):
    #superpose( *pair )
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_rmsd.append(rms)


ssd_rmsd = np.array( ssd_rmsd )
k_clust, cost,num	= kmedoids( ssd_rmsd, nclusters=1, npass=100)
centroid_ID = list(set(k_clust))[0]
print("centroid ID: " + str(centroid_ID))
print('representative frame: ' + str(clust_03_frames[centroid_ID]))
print("cluster state: " + str(h_clust[clust_03_frames[centroid_ID]]))


# In[18]:


clust_04 = []
clust_04_frames = []
for i in range(len(features)):
    if h_clust[i] == 4:
        #print(i)
        clust_04_frames.append(i)
        clust_04.append(features[i])
        #clust_01.append( dcd[i].getCoords() )


# re-do kmedoids clustering to find cluster, including re-calculating dist matrix bc in isnt too long
ssd_rmsd = []
for n, pair in enumerate( combinations( clust_04, 2 ) ):
    #superpose( *pair )
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_rmsd.append(rms)


ssd_rmsd = np.array( ssd_rmsd )
k_clust, cost,num	= kmedoids( ssd_rmsd, nclusters=1, npass=100)
centroid_ID = list(set(k_clust))[0]
print("centroid ID: " + str(centroid_ID))
print('representative frame: ' + str(clust_04_frames[centroid_ID]))
print("cluster state: " + str(h_clust[clust_04_frames[centroid_ID]]))


# In[19]:


clust_05 = []
clust_05_frames = []
for i in range(len(features)):
    if h_clust[i] == 5:
        #print(i)
        clust_05_frames.append(i)
        clust_05.append(features[i])
        #clust_01.append( dcd[i].getCoords() )


# re-do kmedoids clustering to find cluster, including re-calculating dist matrix bc in isnt too long
ssd_rmsd = []
for n, pair in enumerate( combinations( clust_05, 2 ) ):
    #superpose( *pair )
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_rmsd.append(rms)


ssd_rmsd = np.array( ssd_rmsd )
k_clust, cost,num	= kmedoids( ssd_rmsd, nclusters=1, npass=100)
centroid_ID = list(set(k_clust))[0]
print("centroid ID: " + str(centroid_ID))
print('representative frame: ' + str(clust_05_frames[centroid_ID]))
print("cluster state: " + str(h_clust[clust_05_frames[centroid_ID]]))


# In[30]:


clust_06 = []
clust_06_frames = []
for i in range(len(features)):
    if h_clust[i] == 6:
        #print(i)
        clust_06_frames.append(i)
        clust_06.append(features[i])
        #clust_01.append( dcd[i].getCoords() )


# re-do kmedoids clustering to find cluster, including re-calculating dist matrix bc in isnt too long
ssd_rmsd = []
for n, pair in enumerate( combinations( clust_06, 2 ) ):
    #superpose( *pair )
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_rmsd.append(rms)


ssd_rmsd = np.array( ssd_rmsd )
k_clust, cost,num	= kmedoids( ssd_rmsd, nclusters=1, npass=100)
centroid_ID = list(set(k_clust))[0]
print("centroid ID: " + str(centroid_ID))
print('representative frame: ' + str(clust_06_frames[centroid_ID]))
print("cluster state: " + str(h_clust[clust_06_frames[centroid_ID]]))


# In[20]:


clust_07 = []
clust_07_frames = []
for i in range(len(features)):
    if h_clust[i] == 7:
        #print(i)
        clust_07_frames.append(i)
        clust_07.append(features[i])
        #clust_01.append( dcd[i].getCoords() )


# re-do kmedoids clustering to find cluster, including re-calculating dist matrix bc in isnt too long
ssd_rmsd = []
for n, pair in enumerate( combinations( clust_07, 2 ) ):
    #superpose( *pair )
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_rmsd.append(rms)


ssd_rmsd = np.array( ssd_rmsd )
k_clust, cost,num	= kmedoids( ssd_rmsd, nclusters=1, npass=100)
centroid_ID = list(set(k_clust))[0]
print("centroid ID: " + str(centroid_ID))
print('representative frame: ' + str(clust_07_frames[centroid_ID]))
print("cluster state: " + str(h_clust[clust_07_frames[centroid_ID]]))


# In[21]:


clust_08 = []
clust_08_frames = []
for i in range(len(features)):
    if h_clust[i] == 8:
        #print(i)
        clust_08_frames.append(i)
        clust_08.append(features[i])
        #clust_01.append( dcd[i].getCoords() )


# re-do kmedoids clustering to find cluster, including re-calculating dist matrix bc in isnt too long
ssd_rmsd = []
for n, pair in enumerate( combinations( clust_08, 2 ) ):
    #superpose( *pair )
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_rmsd.append(rms)


ssd_rmsd = np.array( ssd_rmsd )
k_clust, cost,num	= kmedoids( ssd_rmsd, nclusters=1, npass=100)
centroid_ID = list(set(k_clust))[0]
print("centroid ID: " + str(centroid_ID))
print('representative frame: ' + str(clust_08_frames[centroid_ID]))
print("cluster state: " + str(h_clust[clust_08_frames[centroid_ID]]))


# In[33]:


clust_09 = []
clust_09_frames = []
for i in range(len(features)):
    if h_clust[i] == 9:
        #print(i)
        clust_09_frames.append(i)
        clust_09.append(features[i])
        #clust_01.append( dcd[i].getCoords() )


# re-do kmedoids clustering to find cluster, including re-calculating dist matrix bc in isnt too long
ssd_rmsd = []
for n, pair in enumerate( combinations( clust_09, 2 ) ):
    #superpose( *pair )
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_rmsd.append(rms)


ssd_rmsd = np.array( ssd_rmsd )
k_clust, cost,num	= kmedoids( ssd_rmsd, nclusters=1, npass=100)
centroid_ID = list(set(k_clust))[0]
print("centroid ID: " + str(centroid_ID))
print('representative frame: ' + str(clust_09_frames[centroid_ID]))
print("cluster state: " + str(h_clust[clust_09_frames[centroid_ID]]))


# In[34]:


clust_10 = []
clust_10_frames = []
for i in range(len(features)):
    if h_clust[i] == 10:
        #print(i)
        clust_10_frames.append(i)
        clust_10.append(features[i])
        #clust_01.append( dcd[i].getCoords() )


# re-do kmedoids clustering to find cluster, including re-calculating dist matrix bc in isnt too long
ssd_rmsd = []
for n, pair in enumerate( combinations( clust_10, 2 ) ):
    #superpose( *pair )
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_rmsd.append(rms)


ssd_rmsd = np.array( ssd_rmsd )
k_clust, cost,num	= kmedoids( ssd_rmsd, nclusters=1, npass=100)
centroid_ID = list(set(k_clust))[0]
print("centroid ID: " + str(centroid_ID))
print('representative frame: ' + str(clust_10_frames[centroid_ID]))
print("cluster state: " + str(h_clust[clust_10_frames[centroid_ID]]))


# In[35]:


clust_11 = []
clust_11_frames = []
for i in range(len(features)):
    if h_clust[i] == 11:
        #print(i)
        clust_11_frames.append(i)
        clust_11.append(features[i])
        #clust_01.append( dcd[i].getCoords() )


# re-do kmedoids clustering to find cluster, including re-calculating dist matrix bc in isnt too long
ssd_rmsd = []
for n, pair in enumerate( combinations( clust_11, 2 ) ):
    #superpose( *pair )
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_rmsd.append(rms)


ssd_rmsd = np.array( ssd_rmsd )
k_clust, cost,num	= kmedoids( ssd_rmsd, nclusters=1, npass=100)
centroid_ID = list(set(k_clust))[0]
print("centroid ID: " + str(centroid_ID))
print('representative frame: ' + str(clust_11_frames[centroid_ID]))
print("cluster state: " + str(h_clust[clust_11_frames[centroid_ID]]))


# In[36]:


clust_12 = []
clust_12_frames = []
for i in range(len(features)):
    if h_clust[i] == 12:
        #print(i)
        clust_12_frames.append(i)
        clust_12.append(features[i])
        #clust_01.append( dcd[i].getCoords() )


# re-do kmedoids clustering to find cluster, including re-calculating dist matrix bc in isnt too long
ssd_rmsd = []
for n, pair in enumerate( combinations( clust_12, 2 ) ):
    #superpose( *pair )
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_rmsd.append(rms)


ssd_rmsd = np.array( ssd_rmsd )
k_clust, cost,num	= kmedoids( ssd_rmsd, nclusters=1, npass=100)
centroid_ID = list(set(k_clust))[0]
print("centroid ID: " + str(centroid_ID))
print('representative frame: ' + str(clust_12_frames[centroid_ID]))
print("cluster state: " + str(h_clust[clust_12_frames[centroid_ID]]))


# In[37]:


clust_13 = []
clust_13_frames = []
for i in range(len(features)):
    if h_clust[i] == 13:
        #print(i)
        clust_13_frames.append(i)
        clust_13.append(features[i])
        #clust_01.append( dcd[i].getCoords() )


# re-do kmedoids clustering to find cluster, including re-calculating dist matrix bc in isnt too long
ssd_rmsd = []
for n, pair in enumerate( combinations( clust_13, 2 ) ):
    #superpose( *pair )
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_rmsd.append(rms)


ssd_rmsd = np.array( ssd_rmsd )
k_clust, cost,num	= kmedoids( ssd_rmsd, nclusters=1, npass=100)
centroid_ID = list(set(k_clust))[0]
print("centroid ID: " + str(centroid_ID))
print('representative frame: ' + str(clust_13_frames[centroid_ID]))
print("cluster state: " + str(h_clust[clust_13_frames[centroid_ID]]))


# In[38]:


clusters=Counter( h_clust).most_common()
cluster_ids = []
cluster_cnts = []
for i in range(len(clusters)): 
    cluster_ids.append(clusters[i][0])
    cluster_cnts.append(clusters[i][1]) 
print(cluster_ids)
print(cluster_cnts)


# In[39]:


plt.bar(cluster_ids, cluster_cnts, )

plt.xticks(cluster_ids,  fontsize=14)
plt.xlabel("Cluster IDs", fontsize=18)

plt.ylabel("Counts", fontsize=18)

plt.title("Cluster Frequency Counts", fontsize=20 )
for i in range(len(cluster_cnts)):
    plt.text(x = cluster_ids[i]-.5 , y = cluster_cnts[i]+1, s = 'n='+str(cluster_cnts[i]), size = 9)


# In[1]:


clust_cnts = Counter( h_clust).most_common()[0][0]
print (clust_cnts)
print (Counter( h_clust).most_common())


clust_01 = []


for i in range(len(features)):
    if h_clust[i] == 1:
        print(i)
        clust_01.append( dcd[i].getCoords() )


# do kmedoids clustering to find cluster, including re-calculationg matrix bc in isnt too long
ssd_array = []
for n, pair in enumerate( combinations( main_clust, 2 ) ):
    superpose( *pair )
    ssd_array.append( calcRMSD( *pair ) )

ssd_array = np.array( ssd_array )


# In[141]:



#from Bio.Cluster.cluster import kmedoids
k_clust, cost,num	= kmedoids( ssd_array, nclusters=1, npass=100)

#print (k_clust)
#print(set( k_clust ) )
centroid_ID = list( set( k_clust ) )[0]
#centroid  	= TM_structs[centroid_ID]


print ('centroid ID (w/in cluster_IDS): ' +  str(centroid_ID))

clust_01_ids = []
for i in range(len(features)):

    
    if h_clust[i] == 1:
        clust_01_ids.append(i)
#print(clust_01_ids)
print("representative frame:" + str(clust_01_ids[centroid_ID]))
print('cluster ID: ' + str(h_clust[clust_01_ids[centroid_ID]]))
#print(h_clust[clust_01_ids[0]])


# In[144]:



main_clust = []


for i in range(len(features)):
    if h_clust[i] == 4:
        main_clust.append( dcd[i].getCoords() )


# re-do kmedoids clustering to find cluster, including re-calculationg matrix bc in isnt too long
ssd_array 	= []
for n, pair in enumerate( combinations( main_clust, 2 ) ):
    superpose( *pair )
    ssd_array.append( calcRMSD( *pair ) )

ssd_array = np.array( ssd_array )




# In[159]:


#from Bio.Cluster.cluster import kmedoids
k_clust, cost,num	= kmedoids( ssd_array, nclusters=1, npass=100)

#print (k_clust)
#print(set( k_clust ) )
centroid_ID = list( set( k_clust ) )[0]
#centroid  	= TM_structs[centroid_ID]


print ('centroid ID (w/in cluster_IDS): ' +  str(centroid_ID))

clust_04_ids = []
for i in range(len(features)):
    if h_clust[i] == 1:
        clust_04_ids.append(i)
print(clust_04_ids[200])
print("representative frame:" + str(clust_04_ids[200]))
print('cluster ID: ' + str(h_clust[200]))


# In[70]:


#find representative frame for each cluster with kmedoids

clust_01 = [] 
#collect frames for clust_01 
for i in range(len(features)):
    #print(h_clust[i])
    if h_clust[i] == 1:
        clust_01.append(dcd[i])
#print(len(clust_01))
#calc dist matrix for clust_01 

ssd_array_01 = []
for n, pair in enumerate( combinations( clust_01, 2 ) ):
    #superpose( *pair )
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_array_01.append( rms)

#do kmedoids on clus_01 frame to find representative 
ssd_array_01 	= np.array( ssd_array_01 )
k_clust, cost,num	= kmedoids( ssd_array_01, nclusters=1, npass=100)

#for i in k_clust:
#    print(i)


# In[67]:


print(set(k_clust))
centroid_ID = list( set( k_clust ) )[0]
centroid    = features[centroid_ID]


# In[39]:


plt.plot(h_clust)
for i in range(len(h_clust)):
    if h_clust[i]==1:
        print("frame: " + str(i) + " state: " + str(h_clust[i]))
    
    
    


# In[36]:


from numpy import array, dot, amax, amin
from Bio.Cluster import pca


# In[56]:


#print(np.shape(ssd_rmsd))
print(np.shape(np.cov(features)))
print(np.trace(np.cov(features)))
features1 = np.linalg.norm(np.array(features))

#print(features)
#print(features1)
columnmean, coordinates, pc, eig_val = pca(features/features1)
#print(coordinates)
#print(pc)
# print(np.sum(eig_val))
print(eig_val[0])
print(eig_val[1])
# print(eig_val[2])
# print(eig_val[3])
print(eig_val[0]/np.sum(eig_val))
print(eig_val[1]/np.sum(eig_val))
#print(eig_val[2]/np.sum(eig_val))
#print(eig_val[3]/np.sum(eig_val))
#print(np.trace(np.array(ssd_rmsd)))

columnmean, coordinates, pc, eig_val = pca(features)
#print(coordinates)
#print(pc)
# print(np.sum(eig_val))
print(eig_val[0])
print(eig_val[1])
# print(eig_val[2])
# print(eig_val[3])
print(eig_val[0]/480.30634278002697)
print(eig_val[1]/480.30634278002697)
#print(d)


# In[47]:


ids=list(range(0,39))
plt.scatter(ids,eig_val)
plt.title("Scree Plot")
plt.ylabel("Eigenvalues")


# In[48]:


#print(pc[0])

plt.scatter(pc[0], pc[1])
plt.ylabel("PC1")
plt.xlabel("PC2")
plt.ylim([-.4, .5])
plt.xlim([-.4, .5])


# In[47]:





# import scipy.cluster.hierarchy as shc
# plt.figure(figsize=(10, 7))  
# plt.title("Dendrograms")  
# #colors = ['tomato', 'darkorange', 'gold', 'tab:purple', 'tab:green', 'powderblue', 'mediumaquamarine', 'slateblue', 'orchid', 'pink', 'lightseagreen', 'hotpink', 'darkblue']

# #shc.set_link_color_palette(colors)

# dend = shc.dendrogram(shc.linkage(ssd_rmsd, method='ward'), color_threshold=2.5, above_threshold_color='grey')
# plt.axhline(y=2.5, color='r', linestyle='--')


# In[57]:


plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(features, method='ward'))
plt.axhline(y=1.52, color='r', linestyle='--')


# In[42]:


plt.plot(h_clust) #traj 009 - no angle in ft vect,  5x weight on bound vs unvound cutoff of 3.75 


# In[ ]:





# In[50]:


plt.plot(h_clust) #traj 009 - w/ angle  & 10x weight on bound vs nonbound 


# In[56]:


plt.plot(h_clust) #traj 051 - w/ angle  & 10x weight on bound vs nonbound (cutoff 38.18 )


# In[73]:


plt.plot(h_clust) #traj 009, all loop contacts, no angle, thresh 3.0 
print(h_clust[100])
print(h_clust[600])


# In[78]:


plt.plot(h_clust)
print(h_clust[100])
print(h_clust[200])
print(h_clust[400])


# In[ ]:


feature_ids = ['bound', '27', '28', '29', '30', '31', '49',
              '50', '51', '52', '53', '54', '91', '92', '93',
               '94', '95', '96', '239', '240', '241', '242', '243',
               '244', '245', '266', '267', '268', '269', '316',
               '317', '318', '319', '320', '321', '322', '232',
               '324']
data= 

