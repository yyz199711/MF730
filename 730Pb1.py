#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 23:18:27 2020

@author: shousakai
"""

import numpy as np
from math import log, exp, sqrt
import matplotlib.pyplot as plt

def Simulation(s, kappa, Sbar, sigma, gamma, t, T, N, M):
    Y_t = log(s)
    h = (T - t) / N
    Y = np.zeros((N+1, M))
    
    for i in range(M):
        for j in range(N+1):
            if j == 0 :
                Y[j][i] = Y_t
            else:
                Y[j][i] = Y[j-1][i] + h * (kappa * (Sbar - Y[j-1][i]) \
                        - 0.5 * sigma ** 2 * exp(2 * gamma * Y[j-1][i])) \
                        + sigma * exp(gamma * Y[j-1][i]) * np.random.normal(0, sqrt(h), 1)[0]
    
    Y_T = Y[N,:]
    
    E = sum(Y_T) / M
    
    return Y, E

def plot(s, kappa, Sbar, sigma, gamma, t, T, N, M):
    h = (T-t)/N
    x = np.array([t + h * j for j in range(N+1)])
    
    y, E = Simulation(s, kappa, Sbar, sigma, gamma, t, T, N, M)
    
    plt.plot(x, y)
    plt.xlabel("Time")
    plt.ylabel("Log-price")
    plt.title("simulation")
    plt.show()
    
def plot_err(s, kappa, Sbar, sigma, gamma, t, T, N, M):
    exact_ex = log(5)*exp(-1.5)+(log(10)-0.01/3)*(1-exp(-1.5))
    err = np.zeros(N)
    x = np.array([i for i in range(1, N + 1)])
    for i in range(1, N+1):
        y, err[i-1] = Simulation(s, kappa, Sbar, sigma, gamma, t, T, i, M)
    
    err = err - exact_ex
    
    plt.plot(x, err)
    
    plt.xlabel("N")
    plt.ylabel("ERROR")
    plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

    