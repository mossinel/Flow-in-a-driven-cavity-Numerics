# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 10:30:16 2022

@author: danie
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rc('font', size=18)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rc('figure', titlesize=18)  # fontsize of the figure title

vel_ref= np.array([0, -0.05, -0.099, -0.15, -0.18, -0.2, -0.191, -0.11, 0.091, 0.48, 1])
x_ref= np.array([0, 0.11, 0.21, 0.31, 0.40, 0.51, 0.6, 0.71, 0.81, 0.91, 1 ])

#### PLOT BURGRAFF ####
fig1 = plt.figure()
plt.plot(vel_ref, x_ref,'-o',label='reference Burggraf',markersize=8)
plt.legend()
plt.grid()
plt.title('Velocity results - comparison with reference')
plt.xlabel('horizontal velocity $u_1$')
plt.ylabel('vertical position')
fig1.set_figheight(8)
fig1.set_figwidth(14)
# plt.savefig('velocity_ref.eps', format='eps')
# plt.savefig('velocity_ref.png', format='png')

velocity=np.loadtxt('velocity_line.txt')
pos=np.loadtxt('pos_line.txt')

#### PLOT COMPARISON ####
fig1 = plt.figure()
plt.plot(vel_ref, x_ref,'-o',label='reference Burggraf',markersize=8)
plt.plot(velocity, pos, '-', label=' numerical results', markersize=8)
plt.legend()
plt.grid()
plt.title('Velocity results - Comparison with reference', weight='bold')
plt.xlabel('horizontal velocity $u_1$')
plt.ylabel('vertical position $Y$')
fig1.set_figheight(8)
fig1.set_figwidth(14)
plt.savefig('velocity_ref.eps', format='eps')
plt.savefig('velocity_ref.png', format='png')