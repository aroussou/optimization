#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 10:10:54 2021

@author: alexandra
"""

# The following code is an optimization code of the 1COMP case
# of N bosons confined in a ring through ED

# Some similar .py file that I developed are the following:
# 
# 1) "edfullhyst.py" which reads the H1 and H2 matrices (originally obtained 
# from Matlab), store the Hamiltonian matrix after computing the g/2*H2 
# and taking the sum H = H1 + g/2*H2, and finally calculates the eigenergy E0
# and eigenvalue V0 of the problem for diff. values of L, which are relevant
# to the hysteresis phenomenon in the 1COMP system (i.e. L=0,1,N-1,N)
#
# 2) "1COMP_Exact_Diagonalization_Optimization" Jupyter Notebook,
# coming from "1COMP_Python_OptimizationCode_ConstructingH2matrix_FINAL_ANDREAS_CODE!!!"
# Jupyter Notebook and with results given in "1COMP_Python_OptimizationCode_ConstructingH2matrix_RESULTS!!!"
# Jupyter Notebook

# -------------
# Import libraries
# -------------
from itertools import combinations
#from scipy.sparse.linalg import eigsh
#from scipy.linalg import eig
from scipy.sparse import diags
from scipy import sparse
##from scipy import linalg
import numpy as np
#import os, psutil; print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2) # shows how many MBytes 
import math
import time
#import re
#import os
#import os.path
import sys
#import timeit

float_formatter = "{:.16f}".format
#np.set_printoptions(formatter={'float_kind':float_formatter})
np.set_printoptions(threshold=sys.maxsize,formatter={'float_kind':float_formatter})
#np.set_printoptions(precision=16)

# -------------
# Give inputs
# -------------   
# HERE #           
N = 9
L = 0        
g = 1    
mmin = -7
mmax = 7
print("\n")        

# -------------
# Define class
# -------------
class ed_1COMP_opt():
    def __init__(self,N,L,mmin,mmax):
        self.N = N
        self.L = L
        #self.g = g
        self.mmin = mmin
        self.mmax = mmax
        
    ''' Helper functions '''
    def multmatnumberdiv2(self,mat,g):
        '''
        Multiply a number divides by 2 with a numpy array
    
        Parameters
        ----------
        mat : numpy array
        
        g : number
            
            g is the interaction strength of the N bosons in a ring
    
        Returns
        -------
        mat : Resulting numpy array after multiplication with number g
    
        '''
        mat = g/2.*mat
        return mat
    
    # Find all combinations for the basis states of specific N and L
    def nsumk(self,n,k):
        allcomb = []
        m = math.comb(k+n-1,n-1)
        xNchooseK = list(range(1,k+n-1+1)) 
    
        # Get all combinations of a list of given length
        # E.g. Get all combinations of [1, 2, 3] and length 2
        comb = combinations(xNchooseK, n-1)

        # Print the obtained combinations
        for i in list(comb):
            allcomb.append(i)
            
        allcombs = np.array(allcomb)
        div_array = (np.diff(np.hstack([np.zeros((m,1),int),allcombs,np.ones((m,1),int)*(k+n)]),1)-1).astype(int)
        return div_array
    
    def save_nparray_as_txt(self, path, mat):
        np.savetxt(path, mat)
        return mat
    
    # Get the power (p) of all elements in a list (my_list) 
    def power(self,my_list,p):
        return [ x**p for x in my_list ]
    
    ''' Step 1: Create the basis states'''
    def edstates(self,N,L,mmin,mmax):
        # Truncation 
        mlist = list(range(mmin,mmax+1))
        
        # Find all_states
        #max_comb = math.comb(N+(mmax - mmin + 1)-1,(mmax - mmin + 1)-1)
        #list_size = max_comb*(mmax - mmin + 1)*8/2**20
        #print('Permutation list_size = %0.2f ' % list_size, 'Mb\n')
        all_states = ed_1COMP_opt(self.N,self.L,self.mmin,self.mmax).nsumk(mmax - mmin + 1,N) # numofallcombs, all_states
        
        # Find the Lvalues of every state in all_states
        #mvalues = np.array(mlist)
        #Lvalues = np.dot(all_states,np.array(mlist).T)
        
        # Find all indices in vector Lvalues where the Lvalue-element == L
        ##Lidx = np.where(Lvalues==L)
        #print(Lidx)
        #nLidx = np.size(Lidx)
        #print('Basis dimension =',nLidx,'\n')
        #print('\n')
        
        # Extract all states with index Lidx=np.where(Lvalues==L), and transpose to mimic previous version
        states = np.squeeze(all_states[np.where(np.dot(all_states,np.array(mlist).T)==L),:])
        #states = np.squeeze(states)
        return states
    
    ''' Step 2: Create the H1 (kinetic energy operator) acting on the 1COMP basis states'''
    def get_H1(self, N, L, mmin, mmax):
        # Get all basis states 
        initial_states = ed_1COMP_opt(self.N,self.L,self.mmin,self.mmax).edstates(N,L,mmin,mmax)
        # Truncation 
        mlist = list(range(mmin,mmax+1))
        # Truncation for kinetic energy
        m2list = ed_1COMP_opt(self.N,self.L,self.mmin,self.mmax).power(mlist,2)
        
        H1 = np.inner(m2list,initial_states)
        H1s = diags(H1,0)#.toarray() # convert to numpy array

        #H1 = np.diagflat(H1)
        return H1s
    
    ''' Step 3: Create the H2 (int. energy operators) acting on the 1COMP basis states'''

    ''' Functions that we need for H2 : '''
    def count_excitations(self,bra,ket):
        delta_state = bra-ket
        return sum(x for x in delta_state if x > 0), delta_state

    def locate_excitations(self,delta_state):
        return np.where(delta_state>0)[0], np.where(delta_state<0)[0]
    
    def tbme_d(self,bra,g,mmin,mmax):
        nnstate = np.nonzero(bra)
        
        diagonal_number = 0
        
        # When i == j
        for x in np.nditer(nnstate):
            diagonal_number += (1/2)*bra[x]*(bra[x]-1)*2*g
    
        # When i != j
        for xi in np.nditer(nnstate): 
            for xj in np.nditer(nnstate):
                if xi != xj:
                    diagonal_number += bra[xi]*bra[xj]*2*g
        return diagonal_number
     
    def tbme_od(self,bra, ket):
        n_excitations, delta = ed_1COMP_opt(self.N,self.L,self.mmin,self.mmax).count_excitations(bra, ket)
        #print(n_excitations,delta)
        if n_excitations > 2:
            return 0
        else:
            bra_side, ket_side = ed_1COMP_opt(self.N,self.L,self.mmin,self.mmax).locate_excitations(delta)
            na = bra[bra_side[0]]
            try:
                nb = bra[bra_side[1]]
                delta_ab = 0
                fac_ab = 1
            except IndexError:
                nb = np.copy(na)
                delta_ab = 1
                fac_ab = 0.5
            nc = ket[ket_side[0]]
            try:
                nd = ket[ket_side[1]]
                delta_cd = 0
                fac_cd = 1
            except IndexError:
                nd = np.copy(nc)
                delta_cd = 1
                fac_cd = 0.5
            return fac_ab*fac_cd*4*np.sqrt(na*(nb-delta_ab)*nc*(nd-delta_cd))
        
    ''' Step 3 (completed): Create H2 acting on the 1COMP basis states'''
    def get_H2(self,finalstates, L, g):
        H2 = sparse.coo_matrix((np.size(finalstates,0),np.size(finalstates,0)))
        H2 = H2.todok() # convert to dok   
        #count_NZME = 0
            
        for bra_pr in range(np.size(finalstates,0)):
            state_row = np.copy(finalstates[bra_pr]) 
            diag_num = ed_1COMP_opt(self.N,self.L,self.mmin,self.mmax).tbme_d(state_row,g,mmin,mmax)
            H2[bra_pr,bra_pr] = diag_num
            for ket_pr in range(bra_pr,np.size(finalstates,0)):
                state_col = np.copy(finalstates[ket_pr]) 
                n_excitations, delta = ed_1COMP_opt(self.N,self.L,self.mmin,self.mmax).count_excitations(state_row,state_col)
                 
                if np.all(delta == 0):
                    continue
                
                #start_od_NZME = time.time()
                tbme_new = ed_1COMP_opt(self.N,self.L,self.mmin,self.mmax).tbme_od(state_row,state_col)
                #end_od_NZME = time.time()
                #print("Elapsed time for the calculation of 1 od NZME of H2 is = ")
                #print(end_od_NZME-start_od_NZME)
                #print('\n')
                
                H2[bra_pr,ket_pr] = tbme_new 
                #print(bra_pr)
                #print(ket_pr)
                #print(tbme_new)
                #print('--------')

                
        H2 = H2.tocoo() # convert back to coo 
        #H2 = H2.toarray() # convert to numpy array       
        ## H2 = H2 + H2.T - np.diag(np.diag(H2)) 
        return H2#, count_NZME   
    
    def timing_and_sparsity_mmax(self,g):
        basis_dim = []
        
        # timing for increasing model spaces abs(mmax)
        time_setup_basis = []
        time_setup_H1 = []
        time_setup_H2 = []
        time_setup_H_total = []
        
        # sparsity for increasing model spaces abs(mmax)
        NZME_H1 = []
        NZME_H2 = []
        NZME_H_total = []
        
        #m = 7
        
        for m in range(1,self.mmax+1):
            start_H_total = time.time()
            start_states = time.time()
            states = ed_1COMP_opt(self.N,self.L,self.mmin,self.mmax).edstates(self.N,self.L,-m,m)
            end_states = time.time()
                
            start_H1 = time.time()
            H1 = ed_1COMP_opt(self.N,self.L,self.mmin,self.mmax).get_H1(self.N, self.L, -m,m)
                #print(H1)
                #print('------end of H1------')
            end_H1 = time.time()
            
            start_H2 = time.time()
            H2 = ed_1COMP_opt(self.N,self.L,self.mmin,self.mmax).get_H2(states, self.L, g)
            end_H2 = time.time()
                
                ##start_H_total = time.time()
            H2g = ed_1COMP_opt(self.N,self.L,self.mmin,self.mmax).multmatnumberdiv2(H2,g)
            H_total = H1 + H2g
                #print(H2g)
                #print('----')
                #print(H_total)
                #print('-------')
                #print('------------')
                #H_total = H1.toarray() + H2g.toarray()
                #H_total = (H1.tocsr() + H2g.tocsr()).tolil()
            end_H_total = time.time()
                
            basis_dim.append(np.size(states,0))
                
            time_setup_basis.append(end_states-start_states)
            time_setup_H1.append(end_H1-start_H1)
            time_setup_H2.append(end_H2-start_H2)
            time_setup_H_total.append(end_H_total-start_H_total)
                
            NZME_H1.append(int(np.count_nonzero(H1.toarray())))
            NZME_H2.append(int(np.count_nonzero(H2.toarray())+np.count_nonzero(np.triu(H2.toarray(),1))))
            NZME_H_total.append(int(np.count_nonzero(H_total.toarray())+np.count_nonzero(np.triu(H_total.toarray(),1))))

        return basis_dim, time_setup_basis, time_setup_H1, time_setup_H2, time_setup_H_total, NZME_H1, NZME_H2, NZME_H_total

# ----------------------------------------------------------
# Timing and sparsity outputs: The 1COMP case
# ----------------------------------------------------------

basis_dim, time_setup_basis, time_setup_H1, time_setup_H2, time_setup_H_total, NZME_H1, NZME_H2, NZME_H_total = ed_1COMP_opt(N,L,mmin,mmax).timing_and_sparsity_mmax(g)

print("Basis Dimensions for increasing mmax are = ")
print(basis_dim)
print('\n')

print("Elapsed times for the BASIS CONSTRUCTION for increasing mmax are = ")
print(time_setup_basis)
print('\n')

print("Elapsed times for the setup of H1 for increasing mmax are = ")
print(time_setup_H1)
print('\n')

print("Elapsed times for the setup of H2 for increasing mmax are = ")
print(time_setup_H2)
print('\n')

print("Elapsed times for the setup of H_total for increasing mmax are = ")
print(time_setup_H_total)
print('\n')

print("NZME of H1 for increasing mmax are = ")
print(NZME_H1)
print('\n')

print("NZME of H2 for increasing mmax are = ")
print(NZME_H2)
print('\n')

print("NZME of H_total for increasing mmax are = ")
print(NZME_H_total)
print('\n')

# ----------------------------------------------------------
# Step 1 (completed): The 1COMP basis states
# ----------------------------------------------------------
# -------------
# Start clock for code timing
# -------------
#start = time.time()

#states = ed_1COMP_opt(N,L,mmin,mmax).edstates(N,L,mmin,mmax)

# -------------
# Stop clock
# -------------
#end = time.time()
#print("Elapsed time for the BASIS CONSTRUCTION is :  %s seconds " % (end - start))
#print('\n')
#print("Basis Dimension is = {:}".format(int(np.size(states,0))))
#print('\n')
#print(states)

# ----------------------------------------------------------
# Step 2 (completed): H1 matrix
# ----------------------------------------------------------
# -------------
# Start clock for code timing
# -------------
##start_setup_H = time.time()  

# -------------
# Start clock for code timing
# -------------
##start = time.time()

##H1 = ed_1COMP_opt(N,L,mmin,mmax).get_H1(N, L, mmin, mmax)

# -------------
# Stop clock
# -------------
##end = time.time()
##print("Elapsed time for the setup of H1 is :  %s seconds " % (end - start))
##print('\n')
##print("NZME of H1 is = {:}".format(int(np.count_nonzero(H1.toarray()))))

# ----------------------------------------------------------
# Step 3.1 (completed): H2 matrix
# ----------------------------------------------------------
# -------------
# Start clock for code timing
# -------------
##start = time.time()

##H2 = ed_1COMP_opt(N,L,mmin,mmax).get_H2(states, L, g)

# -------------
# Stop clock
# -------------
##end = time.time()
##print("Elapsed time for the setup of H2 is :  %s seconds " % (end - start))
##print('\n')

# TESTS:

##print(states)
##print('\n')

#i = 0
#bra = np.copy(states[i])
#print(bra)
#tbme_diag = ed_1COMP_opt(N,L,mmin,mmax).tbme_d(bra,g,mmin,mmax) 
#print(tbme_diag)
#print('\n')

##print(H1)
##print('\n')


##H2g = ed_1COMP_opt(N,L,mmin,mmax).multmatnumberdiv2(H2,g)
##print(H2g)
##print('\n')

##print("NZME of H2 is = {:}".format(int(np.count_nonzero(H2.toarray())+np.count_nonzero(np.triu(H2.toarray(),1)))))
# -------------
# Stop clock
# -------------
##end = time.time()
##print("Elapsed time for the setup of H_total is :  %s seconds " % (end - start))
##print('\n')