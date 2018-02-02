# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 04:20:30 2017

@author: Mihir
"""
#0.1964285714285715

import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.integrate import tplquad
import inspect
from sympy import *
from sympy import Matrix

def get_derivatives(func):
    arg_symbols = symbols(inspect.getargspec(func).args)
    sym_func = func(*arg_symbols)

    return [lambdify(arg_symbols, sym_func.diff(a)) for a in arg_symbols]
    
psi1 = lambda p,q,r: 1/8*(1+p)*(1-q)*(1-r)
psi2 = lambda p,q,r: 1/8*(1+p)*(1+q)*(1-r) 
psi3 = lambda p,q,r: 1/8*(1-p)*(1+q)*(1-r)
psi4 = lambda p,q,r: 1/8*(1-p)*(1-q)*(1-r)
psi5 = lambda p,q,r: 1/8*(1+p)*(1-q)*(1+r)   
psi6 = lambda p,q,r: 1/8*(1+p)*(1+q)*(1+r)
psi7 = lambda p,q,r: 1/8*(1-p)*(1+q)*(1+r)
psi8 = lambda p,q,r: 1/8*(1-p)*(1-q)*(1+r)

psis = np.array([psi1,psi2,psi3,psi4,psi5,psi6,psi7,psi8])

v = 0.39285714

u = np.array([0,0,0,0,v,v,v,v])

def uh(p,q,r):
    sumu = 0
    for i in range(8):
        sumu += u[i]*psis[i](p,q,r)
    return sumu
    
[uhp,uhq,uhr] = get_derivatives(uh)

p = Symbol('p')
q = Symbol('q')
r = Symbol('r')

xc = np.array([0,2,2,0,0.5,1.5,1.5,0.5])
yc = np.array([0,0,2,2,0.5,0.5,1.5,1.5])
zc = np.array([0,0,0,0,1,1,1,1])
'''
def x1(p,q,r):
    sumx = 0
    for i in range(8):        
        sumx += xc[i]*psis[i](p,q,r)
    return sumx
    
def y1(p,q,r):
    sumx = 0
    for i in range(8):        
        sumx += yc[i]*psis[i](p,q,r)
    return sumx   
'''    
def z1(p,q,r):
    sumx = 0
    for i in range(8):        
        sumx += zc[i]*psis[i](p,q,r)
    return sumx
    
zeta = np.linspace(-1,1)
z_sp = z1(1,0.5,zeta)
u_z = uh(1,0.5,zeta)

plt.figure()
plt.plot(z_sp,u_z)
plt.xlabel('z')
plt.ylabel('u')
plt.title('Variation of Temperature (u) along z-axis')
plt.grid()