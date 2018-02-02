# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 00:56:00 2017

@author: Mihir
"""

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

[psi1p,psi1q,psi1r] = get_derivatives(psi1)
[psi2p,psi2q,psi2r] = get_derivatives(psi2)
[psi3p,psi3q,psi3r] = get_derivatives(psi3)
[psi4p,psi4q,psi4r] = get_derivatives(psi4)
[psi5p,psi5q,psi5r] = get_derivatives(psi5)
[psi6p,psi6q,psi6r] = get_derivatives(psi6)
[psi7p,psi7q,psi7r] = get_derivatives(psi7)
[psi8p,psi8q,psi8r] = get_derivatives(psi8)

xc = np.array([0,2,2,0,0.5,1.5,1.5,0.5])
yc = np.array([0,0,2,2,0.5,0.5,1.5,1.5])
zc = np.array([0,0,0,0,1,1,1,1])

psis = np.array([psi1,psi2,psi3,psi4,psi5,psi6,psi7,psi8])
psips = np.array([psi1p,psi2p,psi3p,psi4p,psi5p,psi6p,psi7p,psi8p])
psiqs = np.array([psi1q,psi2q,psi3q,psi4q,psi5q,psi6q,psi7q,psi8q])
psirs = np.array([psi1r,psi2r,psi3r,psi4r,psi5r,psi6r,psi7r,psi8r])

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
    
def z1(p,q,r):
    sumx = 0
    for i in range(8):        
        sumx += zc[i]*psis[i](p,q,r)
    return sumx

'''    
xi = np.linspace(-1,1,6)
eta = np.linspace(-1,1,6)
zeta = np.linspace(-1,1,6)

[Xm,Ym,Zm] = np.meshgrid(xi,eta,zeta)
xsp = x1(Xm,Ym,Zm)
ysp = y1(Xm,Ym,Zm)
zsp = z1(Xm,Ym,Zm)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xsp,ysp,zsp)
#X, Y, Z = axes3d.get_test_data(0.05)
#cset = ax.contour(X, Y, Z, cmap=cm.coolwarm)
#ax.clabel(cset, fontsize=9, inline=1)

plt.contour3D(xsp)
#x1(xi,eta,zeta)
'''
    
[x1p,x1q,x1r] = get_derivatives(x1)
[y1p,y1q,y1r] = get_derivatives(y1)
[z1p,z1q,z1r] = get_derivatives(z1)

J1 = lambda p,q,r: x1p(p,q,r)*(y1q(p,q,r)*z1r(p,q,r) - y1r(p,q,r)*z1q(p,q,r))
J2 = lambda p,q,r: -x1q(p,q,r)*(y1p(p,q,r)*z1r(p,q,r) - y1r(p,q,r)*z1p(p,q,r))
J3 = lambda p,q,r: x1r(p,q,r)*(y1p(p,q,r)*z1q(p,q,r) - y1q(p,q,r)*z1p(p,q,r))

J_det = lambda p,q,r: J1(p,q,r) + J2(p,q,r) + J3(p,q,r)

#J_mat = 

p = Symbol('p')
q = Symbol('q')
r = Symbol('r')

J_mat = Matrix([[x1p(p,q,r),x1q(p,q,r),x1r(p,q,r)],[y1p(p,q,r),y1q(p,q,r),y1r(p,q,r)],[z1p(p,q,r),z1q(p,q,r),z1r(p,q,r)]])
J_inv = J_mat.inv()

p1x = lambdify((p,q,r),J_inv[0,0])
p1y = lambdify((p,q,r),J_inv[0,1])
p1z = lambdify((p,q,r),J_inv[0,2])

q1x = lambdify((p,q,r),J_inv[1,0])
q1y = lambdify((p,q,r),J_inv[1,1])
q1z = lambdify((p,q,r),J_inv[1,2])

r1x = lambdify((p,q,r),J_inv[2,0])
r1y = lambdify((p,q,r),J_inv[2,1])
r1z = lambdify((p,q,r),J_inv[2,2])

K1 = -123*np.ones([8,8])
P1 = 12334*np.ones([8,1])

for i in range(8):
    for j in range(8):
        
        psiix = lambda p,q,r: psips[i](p,q,r)*p1x(p,q,r) + psiqs[i](p,q,r)*q1x(p,q,r) + psirs[i](p,q,r)*r1x(p,q,r)
        psijx = lambda p,q,r: psips[j](p,q,r)*p1x(p,q,r) + psiqs[j](p,q,r)*q1x(p,q,r) + psirs[j](p,q,r)*r1x(p,q,r)
        
        psiiy = lambda p,q,r: psips[i](p,q,r)*p1y(p,q,r) + psiqs[i](p,q,r)*q1y(p,q,r) + psirs[i](p,q,r)*r1y(p,q,r)
        psijy = lambda p,q,r: psips[j](p,q,r)*p1y(p,q,r) + psiqs[j](p,q,r)*q1y(p,q,r) + psirs[j](p,q,r)*r1y(p,q,r)
        
        psiiz = lambda p,q,r: psips[i](p,q,r)*p1z(p,q,r) + psiqs[i](p,q,r)*q1z(p,q,r) + psirs[i](p,q,r)*r1z(p,q,r)
        psijz = lambda p,q,r: psips[j](p,q,r)*p1z(p,q,r) + psiqs[j](p,q,r)*q1z(p,q,r) + psirs[j](p,q,r)*r1z(p,q,r)
        
        integrand = lambda p,q,r: (psiix(p,q,r)*psijx(p,q,r) + psiiy(p,q,r)*psijy(p,q,r) + psiiz(p,q,r)*psijz(p,q,r))*J_det(p,q,r)
        
        K1[i][j] = tplquad(integrand, -1,1, lambda x: -1, lambda x: 1, lambda x,y: -1, lambda x,y: 1)[0]
    
    P1[i] = tplquad(lambda p,q,r: psis[i](p,q,r)*J_det(p,q,r), -1,1, lambda x: -1, lambda x: 1, lambda x,y: -1, lambda x,y: 1)[0]

K_ebc = K1[4:,4:]
p_ebc = P1[4:]

u_ebc = np.linalg.solve(K_ebc,p_ebc)

u = np.zeros([8,1])
u[4:] = u_ebc

def uh(m,n,o):
    sumu = 0
    for i in range(8):
        sumu += u[i]*psis[i](m,n,o)
    return sumu
    
#[uhp,uhq,uhr] = get_derivatives(uh[0])

uhp = lambdify((p,q,r),uh(p,q,r)[0].diff(p))
uhq = lambdify((p,q,r),uh(p,q,r)[0].diff(q))
uhr = lambdify((p,q,r),uh(p,q,r)[0].diff(r))
















    
    