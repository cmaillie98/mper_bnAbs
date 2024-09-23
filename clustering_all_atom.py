#!/usr/bin/env python
# coding: utf-8

# In[9]:


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


# In[10]:


def calcRMS(x, axis=None):
    return np.sqrt(np.nanmean(x**2, axis=axis))


# In[12]:


#read in 4E10 ft vectors 

with open('4E10_ppm_traj_fts.npy', 'rb') as f:
    ppm_4e10 = np.load(f)
f.close()
with open('4E10_p15_traj_fts.npy', 'rb') as f:
    p15_4e10 = np.load(f)
f.close()
with open('4E10_n15_traj_fts.npy', 'rb') as f:
    n15_4e10 = np.load(f)
f.close()
with open('4E10_ppm_traj_fts.npy', 'rb') as f:
    ppm_rep_4e10 = np.load(f)
f.close()
#aggregate 4e10 ft vectors 
aggregate_4e10 = [] 
for i in ppm_4e10:
    aggregate_4e10.append(i)
for i in p15_4e10:
    aggregate_4e10.append(i)
for i in n15_4e10:
    aggregate_4e10.append(i)
for i in ppm_rep_4e10:
    aggregate_4e10.append(i)
    
aggregate_4e10 = np.array(aggregate_4e10)
# print(len(ppm_4e10))
# print(len(p15_4e10))
# print(len(ppm_rep_4e10))
# print(len(n15_4e10))
# print(len(aggregate_4e10[0]))

#cluster aggregate 4e10 AA features 

init_T = time.time()
ssd_rmsd_4e10_aggregate = []
for pairwise in combinations((aggregate_4e10), 2):
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_rmsd_4e10_aggregate.append(rms)
print (round( time.time() - init_T, 3), 's to complete rmsd calculation')
print ('mean distance:', round( np.mean(ssd_rmsd_4e10_aggregate), 1), '| StDev: ', round( np.std(ssd_rmsd_4e10_aggregate) , 1))
#save as np array

with open('4e10_aggregate_aa_ssd_rmsd.npy', 'wb') as f:
    np.save(f, ssd_rmsd_4e10_aggregate)
f.close()


# In[ ]:


#read in pgzl1 ft vectors 

with open('pgzl1_ppm_traj_fts.npy', 'rb') as f:
    ppm_pgzl1 = np.load(f)
f.close()
with open('pgzl1_p15_traj_fts.npy', 'rb') as f:
    p15_pgzl1 = np.load(f)
f.close()
with open('pgzl1_n15_traj_fts.npy', 'rb') as f:
    n15_pgzl1 = np.load(f)
f.close()
with open('pgzl1_ppm_traj_fts.npy', 'rb') as f:
    ppm_rep_pgzl1 = np.load(f)
f.close()
#aggregate pgzl1 ft vectors 
aggregate_pgzl1 = [] 
for i in ppm_pgzl1:
    aggregate_pgzl1.append(i)
for i in p15_pgzl1:
    aggregate_pgzl1.append(i)
for i in n15_pgzl1:
    aggregate_pgzl1.append(i)
for i in ppm_rep_pgzl1:
    aggregate_pgzl1.append(i)
    
# print(len(ppm_pgzl1))
# print(len(p15_pgzl1))
# print(len(ppm_rep_pgzl1))
# print(len(n15_pgzl1))
# print(len(aggregate_pgzl1[0]))

init_T = time.time()
ssd_rmsd_pgzl1_aggregate = []
for pairwise in combinations((aggregate_pgzl1), 2):
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_rmsd_pgzl1_aggregate.append(rms)
print (round( time.time() - init_T, 3), 's to complete rmsd calculation')
print ('mean distance:', round( np.mean(ssd_rmsd_pgzl1_aggregate), 1), '| StDev: ', round( np.std(ssd_rmsd_pgzl1_aggregate) , 1))
#save as np array

with open('pgzl1_aggregate_aa_ssd_rmsd.npy', 'wb') as f:
    np.save(f, ssd_rmsd_pgzl1_aggregate)
f.close()


# In[ ]:


#read in 10e8 ft vectors 

with open('10e8_ppm_traj_fts.npy', 'rb') as f:
    ppm_10e8 = np.load(f)
f.close()
with open('10e8_p15_traj_fts.npy', 'rb') as f:
    p15_10e8 = np.load(f)
f.close()
with open('10e8_n15_traj_fts.npy', 'rb') as f:
    n15_10e8 = np.load(f)
f.close()
with open('10e8_ppm_traj_fts.npy', 'rb') as f:
    ppm_rep_10e8 = np.load(f)
f.close()
#aggregate 10e8 ft vectors 
aggregate_10e8 = [] 
for i in ppm_10e8:
    aggregate_10e8.append(i)
for i in p15_10e8:
    aggregate_10e8.append(i)
for i in n15_10e8:
    aggregate_10e8.append(i)
for i in ppm_rep_10e8:
    aggregate_10e8.append(i)
    
# print(len(ppm_10e8))
# print(len(p15_10e8))
# print(len(ppm_rep_10e8))
# print(len(n15_10e8))
# print(len(aggregate_10e8[0]))

init_T = time.time()
ssd_rmsd_10e8_aggregate = []
for pairwise in combinations((aggregate_10e8), 2):
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_rmsd_10e8_aggregate.append(rms)
print (round( time.time() - init_T, 3), 's to complete rmsd calculation')
print ('mean distance:', round( np.mean(ssd_rmsd_10e8_aggregate), 1), '| StDev: ', round( np.std(ssd_rmsd_10e8_aggregate) , 1))
#save as np array

with open('10e8_aggregate_aa_ssd_rmsd.npy', 'wb') as f:
    np.save(f, ssd_rmsd_10e8_aggregate)
f.close()


# In[ ]:


#combine aggregate 4e10,pgzl1,10e8 systems (3 us of each to not bias it & cut time )
all_atom_aggregate = [] 
for i in aggregate_4e10:
    all_atom_aggregate.append(i)
for i in aggregate_pgzl1:
    all_atom_aggregate.append(i)
for i in aggregate_10e8:
    all_atom_aggregate.append(i)
    
init_T = time.time()
ssd_rmsd_all_atom_aggregate = []
for pairwise in combinations((all_atom_aggregate), 2):
    a, b = pairwise[0], pairwise[1]
    rms = calcRMS(np.array(a)-np.array(b))
    ssd_rmsd_all_atom_aggregate.append(rms)
print (round( time.time() - init_T, 3), 's to complete rmsd calculation')
print ('mean distance:', round( np.mean(ssd_rmsd_all_atom_aggregate), 1), '| StDev: ', round( np.std(ssd_rmsd_all_atom_aggregate) , 1))
#save as np array

with open('ssd_rmsd_all_atom_aggregate.npy', 'wb') as f:
    np.save(f, ssd_rmsd_all_atom_aggregate)
f.close()


# In[ ]:


cutoff =  250 
ssd_rmsd_4e10_aggregate = np.array( ssd_rmsd_4e10_aggregate )
init_T= time.time()

# Ward hierarchical clustering minimizes variance between clusters
# Complete linkage clustering makes sure all cluster members are within same RMSD cutoff to each other
linkMat= linkage( ssd_rmsd_4e10_aggregate , method='ward', metric='euclidean')
print (round( time.time() - init_T, 3), 's to complete clustering')

clusters_4e10= fcluster( linkMat, cutoff, criterion='distance')
num_clusters_4e10 = len( set(clusters_4e10) )
print ('RMS cutoff at %.2f, Unique clusters found:' % cutoff, num_clusters_4e10, '\n')


# In[ ]:


cutoff =  250 
ssd_rmsd_pgzl1_aggregate = np.array( ssd_rmsd_pgzl1_aggregate )
init_T= time.time()

# Ward hierarchical clustering minimizes variance between clusters
# Complete linkage clustering makes sure all cluster members are within same RMSD cutoff to each other
linkMat= linkage( ssd_rmsd_pgzl1_aggregate , method='ward', metric='euclidean')
print (round( time.time() - init_T, 3), 's to complete clustering')

clusters_pgzl1= fcluster( linkMat, cutoff, criterion='distance')
num_clusters_pgzl1 = len( set(clusters_pgzl1) )
print ('RMS cutoff at %.2f, Unique clusters found:' % cutoff, num_clusters_pgzl1, '\n')


# In[ ]:


cutoff =  250 
ssd_rmsd_10e8_aggregate = np.array( ssd_rmsd_10e8_aggregate )
init_T= time.time()

# Ward hierarchical clustering minimizes variance between clusters
# Complete linkage clustering makes sure all cluster members are within same RMSD cutoff to each other
linkMat= linkage( ssd_rmsd_10e8_aggregate , method='ward', metric='euclidean')
print (round( time.time() - init_T, 3), 's to complete clustering')

clusters_10e8= fcluster( linkMat, cutoff, criterion='distance')
num_clusters_10e8 = len( set(clusters_10e8) )
print ('RMS cutoff at %.2f, Unique clusters found:' % cutoff, num_clusters_10e8, '\n')


# In[ ]:


cutoff =  250 
ssd_rmsd_all_atom_aggregate = np.array( ssd_rmsd_all_atom_aggregate )
init_T= time.time()

# Ward hierarchical clustering minimizes variance between clusters
# Complete linkage clustering makes sure all cluster members are within same RMSD cutoff to each other
linkMat= linkage( ssd_rmsd_all_atom_aggregate , method='ward', metric='euclidean')
print (round( time.time() - init_T, 3), 's to complete clustering')

clusters_all_atom= fcluster( linkMat, cutoff, criterion='distance')
num_clusters_all_atom = len( set(clusters_all_atom) )
print ('RMS cutoff at %.2f, Unique clusters found:' % cutoff, num_clusters_all_atom, '\n')


# In[ ]:




