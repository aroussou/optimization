#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 16:56:35 2021

@author: alexandra
"""
# -------------
# Import libraries
# -------------
from scipy.sparse.linalg import eigsh
from scipy.linalg import eig
import numpy as np
import re

# -------------
# Give inputs
# -------------   
# N = 5
# L = 0
# g = 0.5
# mmax = 6
# print("\n") 

# -------------
# Define class
# -------------
class edhyst():
    def __init__(self,N,L,g,mmax):
        self.N = N
        self.L = L
        self.g = g
        self.mmax = mmax
        # L_Omega1 = [0,1]  # limits of the angular momentum for Omega1
        # L_Omega2 = [self.N -1,self.N] # limits of the angular momentum for Omega2
        # self.L_Omega1 = L_Omega1
        # self.L_Omega2 = L_Omega2
    
    def diagonalize(self,H,Hnorm):
        if np.size(H) == 1:
           E0 = float(H.real)
           V0 = 1
           #print("Just 1 state")
        else:
           if self.mmax <= 2:
                D,V = eig(H,Hnorm)
                s = np.argsort(D)
                E0 =  np.real(D[s[0]])
                V0 =  np.real(V[:,s[0]])         
              #print("linalg.eig")
           else:
                D,V = eigsh(H,k=1,M=Hnorm,sigma=None,which='SM') #which='SA'
                E0 = np.real(D[0])   
                # V0 =  np.real(V[:,0])         

                V0str = str(re.sub('[\[\]]', '', str(np.real(V))))
                V0 = np.fromstring(V0str, dtype=np.float64, sep=' ') 
                #print("sparse.eigsh")
        return E0, V0

    def get_E0_Omegas_ED(self):
        root_path = ("/home/alexandra/Documents/Matlab_coding/Exact_Diagonalization/ED_Matlab_AErevision/ED_results_AErevision/ED_matrices_from_Matlab")
        
        dynamic_path_H1_L0 = root_path + ('/ED_H1_AErevision/H1_for_N%d_L%d_mmin_minus%d_mmax%d.txt' % (self.N,0,self.mmax,self.mmax))
        dynamic_path_H1_L1 = root_path + ('/ED_H1_AErevision/H1_for_N%d_L%d_mmin_minus%d_mmax%d.txt' % (self.N,1,self.mmax,self.mmax))
        dynamic_path_H1_LNminus1 = root_path + ('/ED_H1_AErevision/H1_for_N%d_L%d_mmin_minus%d_mmax%d.txt' % (self.N,self.N-1,self.mmax,self.mmax))
        dynamic_path_H1_LN = root_path + ('/ED_H1_AErevision/H1_for_N%d_L%d_mmin_minus%d_mmax%d.txt' % (self.N,self.N,self.mmax,self.mmax))
        
        dynamic_path_H2_L0 = root_path + ('/ED_H2_AErevision/H2_for_N%d_L%d_mmin_minus%d_mmax%d.txt' % (self.N,0,self.mmax,self.mmax))
        dynamic_path_H2_L1 = root_path + ('/ED_H2_AErevision/H2_for_N%d_L%d_mmin_minus%d_mmax%d.txt' % (self.N,1,self.mmax,self.mmax))
        dynamic_path_H2_LNminus1 = root_path + ('/ED_H2_AErevision/H2_for_N%d_L%d_mmin_minus%d_mmax%d.txt' % (self.N,self.N-1,self.mmax,self.mmax))
        dynamic_path_H2_LN = root_path + ('/ED_H2_AErevision/H2_for_N%d_L%d_mmin_minus%d_mmax%d.txt' % (self.N,self.N,self.mmax,self.mmax))

        H1_L0 = np.loadtxt(dynamic_path_H1_L0, dtype=float)
        H1_L1 = np.loadtxt(dynamic_path_H1_L1, dtype=float)
        H1_LNminus1 = np.loadtxt(dynamic_path_H1_LNminus1, dtype=float)
        H1_LN = np.loadtxt(dynamic_path_H1_LN, dtype=float)
        
        H2_L0 = np.loadtxt(dynamic_path_H2_L0, dtype=float)
        H2_L1 = np.loadtxt(dynamic_path_H2_L1, dtype=float)
        H2_LNminus1 = np.loadtxt(dynamic_path_H2_LNminus1, dtype=float)
        H2_LN = np.loadtxt(dynamic_path_H2_LN, dtype=float)
        
        H2_L0 = self.g/2.*H2_L0
        H2_L1 = self.g/2.*H2_L1
        H2_LNminus1 = self.g/2.*H2_LNminus1
        H2_LN = self.g/2.*H2_LN
    
        H_L0 = H1_L0 + H2_L0
        H_L1 = H1_L1 + H2_L1
        H_LNminus1 = H1_LNminus1 + H2_LNminus1
        H_LN = H1_LN + H2_LN
        
        E0_L0, V0_L0 = edhyst(self.N,0,self.g,self.mmax).diagonalize(H_L0,None)
        E0_L1, V0_L1 = edhyst(self.N,1,self.g,self.mmax).diagonalize(H_L1,None)
        E0_LNminus1, V0_LNminus1 = edhyst(self.N,self.N -1,self.g,self.mmax).diagonalize(H_LNminus1,None)
        E0_LN, V0_LN = edhyst(self.N,self.N,self.g,self.mmax).diagonalize(H_LN,None)
        
        Omega1 = E0_L1 - E0_L0
        Omega2 = E0_LN - E0_LNminus1
        
        return E0_L0, E0_L1, E0_LNminus1, E0_LN, Omega1, Omega2

    def E0varymmax(self):
        E0mmax = []
        for m in range(1,self.mmax+1):
            E0m,_,_,_,_,_ = edhyst(self.N,self.L,self.g,m).get_E0_Omegas_ED()
            E0mmax.append(E0m)
        return E0mmax
    
    def Omega1varymmax(self):      
        Omega1mmax = []
        for m in range(1,self.mmax+1):  
            #Omega1m = EDclass(self.N,self.L,self.g,m).get_Omega1()
            _,_,_,_,Omega1m,_ = edhyst(self.N,self.L,self.g,m).get_E0_Omegas_ED()
            Omega1mmax.append(Omega1m)
        return Omega1mmax
    
    def Omega2varymmax(self):      
        Omega2mmax = []
        for m in range(1,self.mmax+1):  
            #Omega2m = EDclass(self.N,self.L,self.g,m).get_Omega2()
            _,_,_,_,_,Omega2m = edhyst(self.N,self.L,self.g,m).get_E0_Omegas_ED()
            Omega2mmax.append(Omega2m)
        return Omega2mmax
    
    def save_plots_to_files(self,pdf_directory,dynamic_name):
        self.pdf_directory = pdf_directory
        self.dynamic_name = dynamic_name
        root_path = ("/home/alexandra/Documents/Python_coding/Exact_Diagonalization/plots_pdfs")
        plot_path = root_path + pdf_directory  + dynamic_name
        return plot_path       

# E0_L0, E0_L1, E0_LNminus1, E0_LN, Omega1_ED, Omega2_ED = edhyst(N,L,g,mmax).get_E0_Omegas_ED()

# print("E0 for N =",N,", L = 0, g =",g,", mmax =",mmax,"is : \n",E0_L0)
# print("\n")
# print("E0 for N =",N,", L = 1, g =",g,", mmax =",mmax,"is : \n",E0_L1)
# print("\n")
# print("E0 for N =",N,", L = ",N-1,", g =",g,", mmax =",mmax,"is : \n",E0_LNminus1)
# print("\n")
# print("E0 for N =",N,", L = ",N,", g =",g,", mmax =",mmax,"is : \n",E0_LN)
# print("\n")
# print("Omega1_ED for N =",N,", g =",g,", mmax =",mmax,"is : \n",Omega1_ED)
# print("\n")
# print("Omega2_ED for N =",N,", g =",g,", mmax =",mmax,"is : \n",Omega2_ED)
# print("\n")


# E0mmax = edhyst(N,L,g,mmax).E0varymmax()
# print("E0mmax for N =",N,", L =",L,", g =",g,", vary mmax until",mmax,"is : \n",E0mmax)
# print("\n")

# Omega1mmax = edhyst(N,L,g,mmax).Omega1varymmax()
# print("Omega1mmax for Nmax =",N,"g =",g," vary mmax until",mmax,"is : \n",Omega1mmax)
# print("\n")
# Omega2mmax = edhyst(N,L,g,mmax).Omega2varymmax()
# print("Omega2mmax for Nmax =",N,"g =",g," vary mmax until",mmax,"is : \n",Omega2mmax)
# print("\n")

# # -------------
# # Plots
# # -------------

# # -----------------------------------------------------------------------------
# # Lowest eigenvalue (E0) convergence vs truncation (mmax) 
# # -----------------------------------------------------------------------------
# mlist = range(1,mmax+1)
# E0mmax = edhyst(N,L,g,mmax).E0varymmax() 
     
# plt.plot(mlist,E0mmax,color='tab:blue',marker='o',linestyle='dashed',label=('g=%.4f' %g))
# plt.xlabel('Symmetric truncated spaces (mmax)')
# plt.ylabel('Energy Convergence (E0) (a.u.)')
# plt.title('E0 for N%d, L%d, mmax%d and specific g' % (N,L,mmax))
# plt.legend()
# plt.yscale('log')
# plt.tight_layout()  

# pdf_directory = "/E0vsmmax_pdfs"
# dynamic_name = ('/E0vsmmax_for_N%d_L%d_g%.4f_mmax%d.pdf' % (N,L,g,mmax))

# plot_pathE0 = edhyst(N,L,g,mmax).save_plots_to_files(pdf_directory,dynamic_name)
# plt.savefig(plot_pathE0)
# plt.show()

# # Test plot_path, where the plot is saved!
# # plot_pathE0 = EDclass(N,L,g,mmax).save_plots_to_files(pdf_directory,dynamic_name)
# # print(plot_pathE0)

# # -----------------------------------------------------------------------------
# # Omega1 convergence vs truncation (mmax) 
# # -----------------------------------------------------------------------------
# Omega1mmax = edhyst(N,L,g,mmax).Omega1varymmax()

# plt.plot(mlist,Omega1mmax,color='tab:orange',marker='o',linestyle='dashed',label= ('Omega1 for g=%.4f' %g))   
# plt.xlabel('Symmetric truncated spaces (mmax)')
# plt.ylabel('E0(L=1)-E0(L=0) (Omega1) (a.u.)')
# plt.title('Omega1 for N%d, mmax%d and specific g' % (N,mmax))
# plt.legend()
# plt.yscale('log')
# plt.tight_layout()

# pdf_directory = "/Omega1vsmmax_pdfs"
# dynamic_name = ('/Omega1vsmmax_for_N%d_g%.4f_mmax%d.pdf' % (N,g,mmax))

# plot_pathOmega1 = edhyst(N,L,g,mmax).save_plots_to_files(pdf_directory,dynamic_name)
# plt.savefig(plot_pathOmega1)
# plt.show()

# # -----------------------------------------------------------------------------
# # Omega2 convergence vs truncation (mmax) 
# # -----------------------------------------------------------------------------
# Omega2mmax = edhyst(N,L,g,mmax).Omega2varymmax()

# plt.plot(mlist,Omega2mmax,color='tab:red',marker='o',linestyle='dashed',label= ('Omega2 for g=%.4f' %g))   
# plt.xlabel('Symmetric truncated spaces (mmax)')
# plt.ylabel('E0(L=N)-E0(L=N-1) (Omega2) (a.u.)')
# plt.title('Omega2 for N%d, mmax%d and specific g' % (N,mmax))
# plt.legend()
# plt.yscale('log')
# plt.tight_layout()

# pdf_directory = "/Omega2vsmmax_pdfs"
# dynamic_name = ('/Omega2vsmmax_for_N%d_g%.4f_mmax%d.pdf' % (N,g,mmax))

# plot_pathOmega2 = edhyst(N,L,g,mmax).save_plots_to_files(pdf_directory,dynamic_name)
# plt.savefig(plot_pathOmega2)
# plt.show()

# # -----------------------------------------------------------------------------
# # Comparison of E0 and Omega1 convergence vs truncation (mmax) [2 y SCALES PLOT]
# # -----------------------------------------------------------------------------

# fig,ax = plt.subplots()
# ax.plot(mlist,Omega1mmax,color='tab:orange',marker='o',linestyle='dashed',label='Omega1mmax')
# ax.set_xlabel("mmax",fontsize=14)
# ax.set_ylabel("Omega1mmax",fontsize=14)
# plt.title('E0 and Omega1 for N%d, mmax%d and specific g%.4f' % (N,mmax,g))
# ax.legend(loc='lower left')

# ax2=ax.twinx()
# ax2.plot(mlist,E0mmax,color='tab:blue',marker='o',linestyle='dashed',label='E0mmaxL0')
# ax2.set_ylabel("E0mmaxL0",fontsize=14)
# ax2.legend(loc='upper right')

# pdf_directory = "/E0_and_Omega1vsmmax_pdfs"
# dynamic_name = ('/E0_and_Omega1vsmmax_for_N%d_g%.4f_mmax%d.pdf' % (N,g,mmax))

# plot_pathE0_and_Omega1 = edhyst(N,L,g,mmax).save_plots_to_files(pdf_directory,dynamic_name)
# fig.savefig(plot_pathE0_and_Omega1,format='pdf',dpi=100,bbox_inches='tight')
# plt.show()

# # -----------------------------------------------------------------------------
# # Comparison of E0 and Omega2 convergence vs truncation (mmax) [2 y SCALES PLOT]
# # -----------------------------------------------------------------------------

# fig,ax = plt.subplots()
# ax.plot(mlist,Omega2mmax,color='tab:red',marker='o',linestyle='dashed',label='Omega2mmax')
# ax.set_xlabel("mmax",fontsize=14)
# ax.set_ylabel("Omega2mmax",fontsize=14)
# plt.title('E0 and Omega2 for N%d, mmax%d and specific g%.4f' % (N,mmax,g))
# ax.legend(loc='lower left')

# ax2=ax.twinx()
# ax2.plot(mlist,E0mmax,color='tab:blue',marker='o',linestyle='dashed',label='E0mmaxL0')
# ax2.set_ylabel("E0mmaxL0",fontsize=14)
# ax2.legend(loc='upper right')

# pdf_directory = "/E0_and_Omega2vsmmax_pdfs"
# dynamic_name = ('/E0_and_Omega2vsmmax_for_N%d_g%.4f_mmax%d.pdf' % (N,g,mmax))

# plot_pathE0_and_Omega1 = edhyst(N,L,g,mmax).save_plots_to_files(pdf_directory,dynamic_name)
# fig.savefig(plot_pathE0_and_Omega1,format='pdf',dpi=100,bbox_inches='tight')
# plt.show()

# # -----------------------------------------------------------------------------
# # Lowest eigenvalue differences (DE0) convergence vs truncation (mmax) 
# # -----------------------------------------------------------------------------
# DE0mmax = [x - E0mmax[i - 1] for i, x in enumerate(E0mmax)][1:]
        
# plt.plot(mlist[:-1],np.abs(DE0mmax),color='tab:brown',marker='o',linestyle='dashed',label=('DE0 for g=%.4f' %g))
# plt.xlabel('Symmetric truncated spaces (mmax)')
# plt.ylabel('Energy Differencies Convergence (DE0)')
# plt.title('DE0 for N%d, L%d, mmax%d and specific g' % (N,L,mmax-1))
# plt.legend(loc='upper right')
# #plt.yscale('log')
# plt.tight_layout()

# pdf_directory = "/E0vsmmax_pdfs"
# dynamic_name = ('/DE0vsmmax_for_N%d_L%d_g%.4f_mmax%d.pdf' % (N,L,g,mmax-1))

# plot_pathDE0 = edhyst(N,L,g,mmax).save_plots_to_files(pdf_directory,dynamic_name)
# plt.savefig(plot_pathDE0)
# plt.show()

# # -----------------------------------------------------------------------------
# # Comparison of E0 and DE0 convergence vs truncation (mmax) [2 y SCALES PLOT]
# # -----------------------------------------------------------------------------
# # Single LEGEND in 2 y SCALES PLOT
# # -----------------------------------------------------------------------------

# fig,ax = plt.subplots()
# l1, = ax.plot(mlist[:-1],E0mmax[:-1],color='tab:blue',marker='o',linestyle='dashed',label='E0mmaxL0')
# ax.set_xlabel("mmax",fontsize=14)
# ax.set_ylabel("E0mmaxL0",fontsize=14)
# plt.title('E0 and DE0 for N%d, L%d, mmax%d and specific g%.4f' % (N,L,mmax-1,g))

# ax2=ax.twinx()
# l2, = ax2.plot(mlist[:-1],np.abs(DE0mmax),color='tab:brown',marker='o',linestyle='dashed',label='DE0')
# ax2.set_ylabel("DE0mmax",fontsize=14)

# fig.legend([l1, l2], ["E0mmaxL0","DE0"],loc='center')
# pdf_directory = "/E0vsmmax_pdfs"
# dynamic_name = ('/E0_and_DE0vsmmax_for_N%d_L%d_g%.4f_mmax%d.pdf' % (N,L,g,mmax-1))
# plot_pathE0_and_DE0 = edhyst(N,L,g,mmax).save_plots_to_files(pdf_directory,dynamic_name)
# fig.savefig(plot_pathE0_and_DE0,format='pdf',dpi=100,bbox_inches='tight')
# plt.show()

# # -----------------------------------------------------------------------------
# # Comparison of E0, Omega1 and DE0 convergence vs truncation (mmax) [3 y SCALES PLOT]
# # -----------------------------------------------------------------------------
# # Single LEGEND in 3 y SCALES PLOT
# # -----------------------------------------------------------------------------

# fig,ax = plt.subplots()
# l3, = ax.plot(mlist[:-1],E0mmax[:-1],color='tab:blue',marker='o',linestyle='dashed',label='E0mmaxL0')
# ax.set_xlabel("mmax",fontsize=14)
# ax.set_ylabel("E0mmaxL0",fontsize=14)
# plt.title('E0, Omega1 and DE0 for N%d, L%d, mmax%d and g%.4f' % (N,L,mmax-1,g))

# ax2=ax.twinx()
# l4, = ax2.plot(mlist[:-1],Omega1mmax[:-1],color='tab:orange',marker='o',linestyle='dashed',label='Omega1mmax')
# ax2.set_ylabel("Omega1mmax",fontsize=14)

# ax3=ax.twinx()
# l5, = ax3.plot(mlist[:-1],np.abs(DE0mmax),color='tab:brown',marker='o',linestyle='dashed',label='DE0mmax')
# ax3.set_ylabel("DE0mmax",fontsize=14)

# fig.legend([l3, l4, l5],["E0mmaxL0", "Omega1","DE0"],loc='center')
# pdf_directory = "/E0vsmmax_pdfs"
# dynamic_name = ('/E0_Omega1_and_DE0vsmmax_for_N%d_L%d_g%.4f_mmax%d.pdf' % (N,L,g,mmax-1))

# plot_pathE0_Omega1_and_DE0 = edhyst(N,L,g,mmax).save_plots_to_files(pdf_directory,dynamic_name)
# fig.savefig(plot_pathE0_Omega1_and_DE0,format='pdf',dpi=100,bbox_inches='tight')
# plt.show()