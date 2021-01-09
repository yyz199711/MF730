#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 19:44:26 2020

@author: Yuyang Zhang & Zehao Dong
"""
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
from scipy.linalg import expm
from IPython.display import Audio
plt.style.use('seaborn')

data = pd.read_excel('Desktop/Final730data.xlsx')
data = data.iloc[1:,:].copy()
data.index = data['Date']
data = data.iloc[:, 1:].copy()
data = data.iloc[::-1]

r_f = 0.01
ret = np.log(data/data.shift(1))
ret.dropna(inplace = True)

mean_year = 12 * ret.mean(axis = 0)

mu = np.array(mean_year,dtype = float)

covariance_year = 12 * ret.cov()

cov = np.array(covariance_year, dtype = float)

def MVF(mu, cov, r_f):
    size = len(mu)
    vec_ones = np.ones(size)
    cov_inv = np.linalg.inv(cov)
    deno = np.dot(vec_ones.T, np.dot(cov_inv, vec_ones)) #denominator for minimum_variance portfolio
    mv_port = np.dot(cov_inv, vec_ones)/deno
    mu_mv = sum(mv_port * mu)
    sigma_mv = np.sqrt(np.dot(mv_port, np.dot(cov, mv_port)))
    excess_return_norf = mu - vec_ones * mu_mv
    excess_return_rf = mu - vec_ones * r_f
    
    m_port_norf = np.dot(cov_inv, excess_return_norf)   #mean-variance portfolio without riskfree asset
    m_port_rf = np.dot(cov_inv, excess_return_rf)  #mean-variance portfolio with riskfree asset
    
    mu_m_port_norf = sum(m_port_norf * mu) #mu and sigma
    sigma_m_port_norf = np.sqrt(np.dot(m_port_norf, np.dot(cov, m_port_norf)))

    
    mu_m_port_rf = sum(m_port_rf * mu) + (1 - sum(m_port_rf)) * r_f
    sigma_m_port_rf = np.sqrt(np.dot(m_port_rf, np.dot(cov, m_port_rf)))
    mu_target = np.linspace(0, 0.4, 100)
    
    port_front = np.zeros((100,size))
    sigma_front = np.zeros(100)
    for i in range(100):
        target_port = mv_port + (mu_target[i] - mu_mv)/mu_m_port_norf * m_port_norf
        port_front[i,:] = target_port
        sigma_front[i] = np.sqrt(sigma_mv**2 + (mu_target[i] - mu_mv)**2 / mu_m_port_norf)
    
    tangency_port = m_port_rf/sum(m_port_rf)
    mu_tangency = sum(tangency_port * mu)
    sigma_tangency = sigma_m_port_rf / abs(sum(m_port_rf))
    
    
    sigma_target = np.linspace(0, max(sigma_front), 100)
    SR_tangency = (mu_tangency - r_f) / sigma_tangency # sharpe ratio, slope of SML
    mu_SML = SR_tangency * sigma_target + r_f
    
    #print(sigma_tangency, mu_tangency)
    plt.plot(sigma_front, mu_target, color = 'r')
    plt.plot(sigma_target, mu_SML, color = 'b')
    plt.plot(sigma_tangency, mu_tangency, 'x', color = 'y')
    plt.legend(['Efficient Frontier', 'Market Security Line', 'Tangency Portfolio'])
    plt.xlabel('Volatility of Portfolio')
    plt.ylabel('Expected Return of Portfolio')
    plt.show()

MVF(mu, cov, 0.01)

state_var = pd.read_csv('Desktop/730FACTOR DATA.csv')

state_var['Date'] = pd.to_datetime(state_var.Date)

state_var.index = state_var['Date']
state_var = state_var.iloc[:,1:].copy()

state_var = state_var[::-1]

d_state_var = state_var['DIV YIELD'] - state_var['DIV YIELD'].shift(1)
d_state_var = pd.DataFrame(d_state_var)
d_state_var = d_state_var.rename(columns = {'DIV YIELD': 'd_div'})

d_state_var['d_eps'] = state_var['EARNINGS YIELD'] - state_var['EARNINGS YIELD'].shift(1)
d_state_var['div'] = state_var['DIV YIELD']
d_state_var['eps'] = state_var['EARNINGS YIELD']

d_state_var = d_state_var.dropna()

model = VAR(d_state_var[['d_div', 'd_eps']])
results = model.fit(1)

results.summary()

d_state_var['fitted_d_div'] = 0.000027 + 0.230516 * d_state_var['d_div'].shift(1) - 0.072953 * d_state_var['d_eps'].shift(1)

d_state_var['fitted_d_eps'] = 0.000069 - 1.291066 * d_state_var['d_div'].shift(1) + 0.595099 * d_state_var['d_eps'].shift(1)

d_state_var['residue_1'] = d_state_var['d_div'] - d_state_var['fitted_d_div']
d_state_var['residue_2'] = d_state_var['d_eps'] - d_state_var['fitted_d_eps']

residue = d_state_var[['residue_1', 'residue_2']]

Y_var = residue.cov()

w, v = np.linalg.eig(Y_var)
sqrt_w = np.sqrt(w)
Y_sigma = np.dot(v, np.diag(sqrt_w))
orthnormal_vec = ortho_group.rvs(4)
Y_sigma_ex = orthnormal_vec[:2]
Y_diffusion = np.dot(Y_sigma, Y_sigma_ex)

Y_diffusion = np.sqrt(12) * Y_diffusion

delta_t = 1/12 # one month represents delta t = 1/12
A = [0.000027, 0.000069]
B = np.array([[0.230516,-0.072953], [-1.291066, 0.595099]])
kappa = (np.identity(2) - B)/delta_t
C = np.array(state_var)
state_var['shift_div'] = state_var['DIV YIELD'].shift(1)
state_var['shift_eps'] = state_var['EARNINGS YIELD'].shift(1)
C1 = np.array(state_var[['shift_div','shift_eps']])
sigma = np.array([[ 1.59486340e-03, -4.47540973e-04, -1.75332075e-04,
        -4.68266104e-04],
       [ 3.64089128e-03, -9.31531131e-05,  4.60180196e-04,
        -2.20741477e-03]])

C2 = C[1:].T - np.dot(B,C1[1:].T)

theta = np.dot(np.linalg.inv(np.identity(2)-B), np.mean(C2,axis=1))

mu_S0 = np.array([0.08600382, 0.05108251, 0.04710213, 0.05375621])

cov_S = np.array([[0.02508057, 0.01317813, 0.02878345, 0.0182023 ],
       [0.01317813, 0.01248853, 0.02102624, 0.01195126],
       [0.02878345, 0.02102624, 0.06003099, 0.02485158],
       [0.0182023 , 0.01195126, 0.02485158, 0.02044259]])

sigma_S = np.linalg.cholesky(cov_S)
kappa = np.array([[ 9.233808,  0.875436],
       [15.492792,  4.858812]])
theta = np.array([0.01980579, 0.05130033])
sigma_Y = np.array([[ 1.59486340e-03, -4.47540973e-04, -1.75332075e-04,
        -4.68266104e-04],
       [ 3.64089128e-03, -9.31531131e-05,  4.60180196e-04,
        -2.20741477e-03]])
sigma_S_inv = np.linalg.inv(sigma_S)
r_f = 0.01
beta = 0.01
R = 4
rho = 1 - 1/R
eta = 2/3
h_bar = 1
T = 60
T_r = 45
mu_w = 0.01
sigma_w = np.array([0.03, 0.03, 0.03, 0.03])/2
wage_0 = 40000
Y0 = np.array([0.0198, 0.0483])
A = np.array([[-1.20615929,  -0.49445039 ],
       [0.51792404  ,  0.21231669],
       [0.52187949, 0.21393818],
       [0.86857626,  0.35606232]])
B = mu_S0 - np.dot(A, Y0)
fi = 20 ** 4

d = 30
frequency = 1 / 4
nodes = int(60*d/frequency+60/frequency+1)
Y_list_0 = np.zeros((2, nodes))
wage_list_0 = np.zeros(nodes)
Y_list_0[:, 0] = Y0
wage_list_0[0] = wage_0
dw_0 = np.random.normal(0, np.sqrt(1/(d+1)*frequency), size = (nodes-1,4))

for i in range(nodes-1):
    Y_list_0[:, i+1] = Y_list_0[:, i] +np.dot(kappa, (theta - Y_list_0[:, i])) /(d+1)*frequency+ np.dot(sigma_Y, dw_0[i])
    wage_list_0[i+1] = wage_list_0[i] + wage_list_0[i]*mu_w/(d+1)*frequency + wage_list_0[i] * np.dot(sigma_w, dw_0[i])

index = [int((d+1)*i) for i in range(int(60/frequency+1))]

wage_month_0 = wage_list_0[index]
Y_month_0 = Y_list_0[:, index]

mu_total_0 = np.dot(A, Y_list_0) + np.repeat(B.reshape(4,1), nodes, axis = 1)
mu_month_0 = mu_total_0[:,index]
theta_total_0 = np.dot(sigma_S_inv, mu_total_0-r_f)
theta_month_0 = np.dot(sigma_S_inv, mu_month_0-r_f)

log_xi_month_0 = np.zeros(int(60/frequency+1))
log_xi_month_0[0] = 0
for i in range(int(60/frequency)):
    index_i_next = index[i+1]
    index_i_before = index[i]
    dw_i = dw_0[index_i_before:index_i_next, :]
    theta_i = theta_total_0[:, index_i_before:index_i_next]
    stoc = np.trace(np.dot(theta_i.T, dw_i.T))
    deter = np.trace(np.dot(theta_i.T, theta_i))
    log_xi_month_0[i+1] = log_xi_month_0[i] -r_f/(d+1)*frequency - stoc - 0.5 * deter/(d+1)*frequency
xi_month_0 = np.exp(log_xi_month_0)

wl = wage_list_0[:5580]
plt.plot(tl,wl)
tl = np.array([45*i/5580 for i in range(5580)])
plt.show()


def get_values(t, dw, y=0):
    index_t = index[t]
    dw_t = dw[:, index_t:]
    Y_t = Y_month_0[:, t]
    wage_t = wage_month_0[t]
    mu_t = mu_month_0[:, t]
    theta_t = theta_month_0[:, t]
    xi_t = xi_month_0[t]

    Y_list = np.zeros((M, 2, nodes - index_t))
    wage_list = np.zeros((M, nodes - index_t))
    Y_list[:, :, 0] = Y_t
    wage_list[:, 0] = wage_t

    for i in range(nodes - index_t - 1):
        Y_list[:, :, i+1] = Y_list[:, :, i] +np.dot(kappa, (theta - Y_list[:, :, i]).T).T /(d+1)*frequency+ np.dot(sigma_Y, dw_t[:, i].T).T
        wage_list[:, i+1] = wage_list[:, i] + wage_list[:, i]*mu_w/(d+1)*frequency + wage_list[:, i] * np.dot(sigma_w, dw_t[:, i].T)

    mu_total = np.zeros((M, 4, nodes - index_t))
    for i in range(M):
        mu_total[i] = np.dot(A, Y_list[i]) + np.repeat(B.reshape(4,1), nodes - index_t, axis = 1)

    theta_total = np.zeros((M, 4, nodes - index_t))
    for i in range(M):
        theta_total[i] = np.dot(sigma_S_inv, mu_total[i]-r_f)

    log_xi_total = np.zeros((M, nodes - index_t))
    log_xi_total[:, 0] = np.log(xi_t)
    for i in range(nodes - index_t - 1):
        dw_i = dw_t[:, i, :]
        theta_i = theta_total[:, :, i]
        stoc = np.diagonal(np.dot(theta_i, dw_i.T))
        deter = np.diagonal(np.dot(theta_i, theta_i.T))
        log_xi_total[:, i+1] = log_xi_total[:, i] -r_f/(d+1)*frequency - stoc - 0.5 * deter/(d+1)*frequency
    xi_total = np.exp(log_xi_total)
    xi_i = xi_total[:, : index[int(45/frequency)] - index_t + 1]   
    wage_i = wage_list[:, : index[int(45/frequency)] - index_t + 1]
    part1 = np.diagonal(np.dot(xi_i, wage_i.T))
    H_t = part1 * h_bar / (d+1)*frequency / xi_t
    H_t = np.mean(H_t)

    t_total = np.array([i * frequency for i in range(nodes - index_t)])

    xi_i = xi_total[:, :index[int(45/frequency)] - index_t + 1]   
    wage_i = wage_list[:, :index[int(45/frequency)] - index_t + 1]
    t_i = t_total[ : index[int(45/frequency)] - index_t + 1]
    part1 = np.diagonal(np.dot((np.exp(-beta / R * t_i) * xi_i ** rho), (wage_i ** ((1 - eta) * rho)).T))
    L_t = part1 / (d+1)*frequency / (xi_t ** rho * wage_t ** ((1 - eta) * rho))
    L_t = 2.5*np.mean(L_t)

    xi_i = xi_total[:, index[int(45/frequency)] - index_t:]
    t_i = t_total[index[int(45/frequency)] - index_t :]
    part1 = np.sum((xi_i ** rho * np.exp(-beta / R * t_i)), axis=1)
    M_t = 4*np.mean(part1 / (xi_t ** rho)) / (d+1)*frequency

    if y == 0:
        y = ((H_t + 100000)/ (1/eta * (eta/(1-eta)) ** ((1-eta)*rho)* wage_t**((1 - eta) * rho) * L_t + fi ** (1/R) * M_t)) ** (-R)

    Nat = (1/eta * (eta/(1-eta)) ** ((1-eta)*rho) * y ** (-1/R) * xi_t ** (-1/R) \
                    * wage_t ** ((1 - eta) * rho) * np.exp(-beta / R * t * frequency)) * L_t

    Nrt = fi ** (1/R) * (y ** (-1/R) * xi_t ** (-1/R) * np.exp(-beta / R * t * frequency)) * M_t

    N_t = Nat + Nrt
    fi_t = Nat / N_t

    c_t = np.exp(-beta / R * t * frequency) * wage_t ** ((1 - eta) * rho) * (eta/(1 - eta)) ** ((1 - eta) * rho) \
          * y ** (-1/R) * xi_t ** (-1/R)

    l_t = c_t * (1 - eta) / (eta * wage_t)

    H_theta_total = np.zeros((M, 4, nodes - index_t))
    H_theta_total[:,:,0] = np.zeros(4)
    part1 = np.dot(np.dot(np.dot(sigma_S_inv, A), expm(- kappa/(d+1)*frequency)), sigma_Y)
    for i in range(nodes - index_t - 1):
        dw_i = dw_t[:, i, :]
        theta_i = theta_total[:, :, i]
        H_theta_total[:,:,i+1] = H_theta_total[:,:,i] + np.dot((dw_i + theta_i/(d+1)*frequency), part1)

    DL_list = np.zeros((M, 4)) 
    xi_i = xi_total[:, :index[int(45/frequency)] - index_t + 1]
    wage_i = wage_list[:, :index[int(45/frequency)] - index_t + 1]
    t_i = t_total[:index[int(45/frequency)] - index_t + 1]
    part1 = np.exp(-beta / R * t_i) * xi_i ** rho * wage_i ** ((1 - eta) * rho)
    for i in range(4):
        H_theta_i = H_theta_total[:,i,:index[int(45/frequency)] - index_t + 1]   
        part2 = np.diagonal(np.dot(part1, H_theta_i.T))
        DL_list[:, i] = -rho * part2 / (d+1)*frequency / xi_t ** rho / wage_t ** ((1 - eta) * rho)
    DL_t = 2.5*np.mean(DL_list, axis=0)

    DM_list = np.zeros((M, 4))
    xi_i = xi_total[:, index[int(45/frequency)] - index_t:]
    t_i = t_total[index[int(45/frequency)] - index_t :]
    part1 = np.exp(-beta / R * t_i) * xi_i ** rho
    for i in range(4):
        H_theta_i = H_theta_total[:,i,index[int(45/frequency)] - index_t:]
        part2 = np.diagonal(np.dot(part1, H_theta_i.T))
        DM_list[:, i] = -rho * part2 / (d+1)*frequency / xi_t ** rho
    DM_t = 4*np.mean(DM_list, axis=0)

    pi_H_theta_list = np.zeros((M, 4))
    xi_i = xi_total[:, :index[int(45/frequency)] - index_t + 1]
    wage_i = wage_list[:, :index[int(45/frequency)] - index_t + 1]
    part1 = xi_i * wage_i * h_bar
    for i in range(4):   
        H_theta_i = H_theta_total[:,i,:index[int(45/frequency)] - index_t + 1]  
        part2 = np.diagonal(np.dot(part1, H_theta_i.T))
        pi_H_theta_list[:, i] = part2 / (d+1)*frequency / xi_t 
    pi_H_theta = np.dot(sigma_S_inv.T, (np.mean(pi_H_theta_list, axis=0) / H_t).T)

    pi_m = np.dot(sigma_S_inv.T, (theta_t))
    pi_w = np.dot(sigma_S_inv.T, sigma_w)
    pi_L_theta = np.dot(sigma_S_inv.T, DL_t.T) / L_t / - rho
    pi_M_theta = np.dot(sigma_S_inv.T, DM_t.T) / M_t / - rho
    pi_H = H_t / N_t * (np.dot(sigma_S_inv.T, sigma_w) - pi_H_theta)

    pi = 1 / R * pi_m + (1 - eta) * rho * fi_t * pi_w - rho * fi_t * pi_L_theta - rho * (1 - fi_t) * pi_M_theta - H_t / N_t * np.dot(sigma_S_inv.T, sigma_w)

    return H_t, L_t, M_t, N_t, c_t, l_t, pi_m, pi_w, pi_H, pi_L_theta, pi_M_theta, pi_H_theta, pi, y, Nat, Nrt

import time

def get_values2(t, dw, y):
    index_t = index[t]
    dw_t = dw[:, index_t:]
    Y_t = Y_month_0[:, t]
    wage_t = wage_month_0[t]
    mu_t = mu_month_0[:, t]
    theta_t = theta_month_0[:, t]
    xi_t = xi_month_0[t]

    Y_list = np.zeros((M, 2, nodes - index_t))
    Y_list[:, :, 0] = Y_t
    for i in range(nodes - index_t - 1):
        Y_list[:, :, i+1] = Y_list[:, :, i] +np.dot(kappa, (theta - Y_list[:, :, i]).T).T /(d+1)*frequency+ np.dot(sigma_Y, dw_t[:, i].T).T

    mu_total = np.zeros((M, 4, nodes - index_t))
    for i in range(M):
        mu_total[i] = np.dot(A, Y_list[i]) + np.repeat(B.reshape(4,1), nodes - index_t, axis = 1)

    theta_total = np.zeros((M, 4, nodes - index_t))
    for i in range(M):
        theta_total[i] = np.dot(sigma_S_inv, mu_total[i]-r_f)

    log_xi_total = np.zeros((M, nodes - index_t))
    log_xi_total[:, 0] = np.log(xi_t)
    for i in range(nodes - index_t - 1):
        dw_i = dw_t[:, i, :]
        theta_i = theta_total[:, :, i]
        stoc = np.diagonal(np.dot(theta_i, dw_i.T))
        deter = np.diagonal(np.dot(theta_i, theta_i.T))
        log_xi_total[:, i+1] = log_xi_total[:, i]-r_f/(d+1)*frequency - stoc - 0.5 * deter/(d+1)*frequency
    xi_total = np.exp(log_xi_total)

    c_t = fi ** (1/R) * np.exp(-beta / R * t * frequency) * (y * xi_t) ** (-1/R)

    t_total = np.array([i * frequency for i in range(nodes - index_t)])

    xi_i = xi_total
    t_i = t_total
    part1 = np.dot(xi_i ** rho,  np.exp(-beta / R * t_i).T)
    M_t = 4*np.mean(part1 / (xi_t ** rho)) / (d+1)*frequency

    N_t = fi ** (1/R) * (y * xi_t) ** (-1/R) * np.exp(-beta / R * t * frequency) * M_t

    H_theta_total = np.zeros((M, 4, nodes - index_t))
    H_theta_total[:,:,0] = np.zeros(4)
    part1 = np.dot(np.dot(np.dot(sigma_S_inv, A), expm(- kappa/(d+1)*frequency)), sigma_Y)
    for i in range(nodes - index_t - 1):
        dw_i = dw_t[:, i, :]
        theta_i = theta_total[:, :, i]
        H_theta_total[:,:,i+1] = H_theta_total[:,:,i] + np.dot((dw_i + theta_i/(d+1)*frequency), part1)

    DM_list = np.zeros((M, 4))
    xi_i = xi_total
    t_i = t_total
    part1 = np.exp(-beta / R * t_i) * xi_i ** rho
    for i in range(4):
        H_theta_i = H_theta_total[:,i,:]
        part2 = np.diagonal(np.dot(part1, H_theta_i.T))
        DM_list[:, i] = -rho * part2 / (d+1)*frequency / xi_t ** rho
    DM_t = 4*np.mean(DM_list, axis=0)
    
    pi_m = np.dot(sigma_S_inv.T, (theta_t))
    pi_M_theta = np.dot(sigma_S_inv.T, DM_t.T) / M_t / - rho

    pi = 1 / R * pi_m - rho * pi_M_theta
    
    return M_t, N_t, c_t, pi_m, pi_M_theta, pi

M = 100
dw = np.random.normal(0, np.sqrt(1/(d+1)*frequency), size = (M, nodes-1,4))

H_t_list = []
L_t_list = []
M_t_list = []
N_t_list = []
Nat_t_list = []
Nrt_t_list = []
c_t_list = []
l_t_list = []
pi_m_list = []
pi_w_list = []
pi_H_list = []
pi_L_theta_list = []
pi_M_theta_list = []
pi_H_theta_list = []
pi_list = []
start_time = time.time()
for i in range(int(45/frequency)):
    if i == 0:
        res = get_values(i, dw)
        H_t_list.append(res[0])
        L_t_list.append(res[1])
        M_t_list.append(res[2])
        N_t_list.append(res[3])
        c_t_list.append(res[4])
        l_t_list.append(res[5])
        pi_m_list.append(res[6])
        pi_w_list.append(res[7])
        pi_H_list.append(res[8])
        pi_L_theta_list.append(res[9])
        pi_M_theta_list.append(res[10])
        pi_H_theta_list.append(res[11])
        pi_list.append(res[12])
        y = res[13]
        Nat_t_list.append(res[14])
        Nrt_t_list.append(res[15])
    else:
        res = get_values(i, dw, y)
        H_t_list.append(res[0])
        L_t_list.append(res[1])
        M_t_list.append(res[2])
        N_t_list.append(res[3])
        c_t_list.append(res[4])
        l_t_list.append(res[5])
        pi_m_list.append(res[6])
        pi_w_list.append(res[7])
        pi_H_list.append(res[8])
        pi_L_theta_list.append(res[9])
        pi_M_theta_list.append(res[10])
        pi_H_theta_list.append(res[11])
        pi_list.append(res[12])
        Nat_t_list.append(res[14])
        Nrt_t_list.append(res[15])
    #print(i, time.time() - start_time)

M_t_list2 = []
N_t_list2 = []
c_t_list2 = []
pi_m_list2 = []
pi_M_theta_list2 = []
pi_list2 = []
for i in range(int(45/frequency)+1, int(60/frequency)+1):
    res2 = get_values2(i, dw, y)
    M_t_list2.append(res2[0])
    N_t_list2.append(res2[1])
    c_t_list2.append(res2[2])    
    pi_m_list2.append(res2[3])       
    pi_M_theta_list2.append(res2[4])      
    pi_list2.append(res2[5])

x = np.array([i * frequency for i in range(int(60/frequency))])
fig = plt.figure(figsize=(20,10),dpi=200)
ax1 = fig.add_subplot(2,2,1)
ax1.set_title('SPY')
ax1.plot(x, np.array(pi_list + pi_list2)[:,0], c='black')
ax1.plot(x, np.array(pi_m_list + pi_m_list2)[:,0])
ax1.plot(x, np.append(np.array(pi_w_list)[:,0], np.zeros(int(15/frequency))))
ax1.plot(x, np.append(np.array(pi_H_list)[:,0], np.zeros(int(15/frequency))))
ax1.plot(x, np.append(np.array(pi_L_theta_list)[:,0], np.zeros(int(15/frequency))))
ax1.plot(x, np.array(pi_M_theta_list + pi_M_theta_list2)[:,0])
ax1.plot(x, np.append(np.array(pi_H_theta_list)[:,0], np.zeros(int(15/frequency))))
ax1.legend(['pi', 'pi_m', 'pi_w', 'pi_H', 'pi_L_theta', 'pi_M_theta', 'pi_H_theta'])
ax2 = fig.add_subplot(2,2,2)
ax2.set_title('HYG')
ax2.plot(x, np.array(pi_list + pi_list2)[:,1], c='black')
ax2.plot(x, np.array(pi_m_list + pi_m_list2)[:,1])
ax2.plot(x, np.append(np.array(pi_w_list)[:,1], np.zeros(int(15/frequency))))
ax2.plot(x, np.append(np.array(pi_H_list)[:,1], np.zeros(int(15/frequency))))
ax2.plot(x, np.append(np.array(pi_L_theta_list)[:,1], np.zeros(int(15/frequency))))
ax2.plot(x, np.array(pi_M_theta_list + pi_M_theta_list2)[:,1])
ax2.plot(x, np.append(np.array(pi_H_theta_list)[:,1], np.zeros(int(15/frequency))))
ax2.legend(['pi', 'pi_m', 'pi_w', 'pi_H', 'pi_L_theta', 'pi_M_theta', 'pi_H_theta'])
ax3 = fig.add_subplot(2,2,3)
ax3.set_title('VNQ')
ax3.plot(x, np.array(pi_list + pi_list2)[:,2], c='black')
ax3.plot(x, np.array(pi_m_list + pi_m_list2)[:,2])
ax3.plot(x, np.append(np.array(pi_w_list)[:,2], np.zeros(int(15/frequency))))
ax3.plot(x, np.append(np.array(pi_H_list)[:,2], np.zeros(int(15/frequency))))
ax3.plot(x, np.append(np.array(pi_L_theta_list)[:,2], np.zeros(int(15/frequency))))
ax3.plot(x, np.array(pi_M_theta_list + pi_M_theta_list2)[:,2])
ax3.plot(x, np.append(np.array(pi_H_theta_list)[:,2], np.zeros(int(15/frequency))))
ax3.legend(['pi', 'pi_m', 'pi_w', 'pi_H', 'pi_L_theta', 'pi_M_theta', 'pi_H_theta'])
ax4 = fig.add_subplot(2,2,4)
ax4.set_title('PGUAX')
ax4.plot(x, np.array(pi_list + pi_list2)[:,3], c='black')
ax4.plot(x, np.array(pi_m_list + pi_m_list2)[:,3])
ax4.plot(x, np.append(np.array(pi_w_list)[:,3], np.zeros(int(15/frequency))))
ax4.plot(x, np.append(np.array(pi_H_list)[:,3], np.zeros(int(15/frequency))))
ax4.plot(x, np.append(np.array(pi_L_theta_list)[:,3], np.zeros(int(15/frequency))))
ax4.plot(x, np.array(pi_M_theta_list + pi_M_theta_list2)[:,3])
ax4.plot(x, np.append(np.array(pi_H_theta_list)[:,3], np.zeros(int(15/frequency))))
ax4.legend(['pi', 'pi_m', 'pi_w', 'pi_H', 'pi_L_theta', 'pi_M_theta', 'pi_H_theta'])
plt.show()

x = np.array([i * frequency for i in range(int(60/frequency))])
fig = plt.figure(figsize=(20,10),dpi=200)
ax1 = fig.add_subplot(2,2,1)
ax1.set_title('SPY')
ax1.plot(x, np.array(pi_list + pi_list2)[:,0])
ax2 = fig.add_subplot(2,2,2)
ax2.set_title('HYG')
ax2.plot(x, np.array(pi_list + pi_list2)[:,1])
ax3 = fig.add_subplot(2,2,3)
ax3.set_title('VNQ')
ax3.plot(x, np.array(pi_list + pi_list2)[:,2])
ax4 = fig.add_subplot(2,2,4)
ax4.set_title('PGUAX')
ax4.plot(x, np.array(pi_list + pi_list2)[:,3])
plt.show()

plt.figure(figsize=(10,6),dpi=100)
plt.plot(x, N_t_list + N_t_list2)
plt.plot(x, Nat_t_list + [0 for i in range(int(15/frequency))])
plt.plot(x, Nrt_t_list + N_t_list2)
plt.plot(x, H_t_list + [0 for i in range(int(15/frequency))])
plt.plot(x, np.append(np.array(N_t_list) - np.array(H_t_list), np.array(N_t_list2) - np.zeros(int(15/frequency))))
plt.legend(['Total wealth', 'Accumalation wealth', 'Pension plan', 'Human capital', 'Financial wealth'])
plt.show()

plt.figure(figsize=(10,6),dpi=100)
plt.plot(x, Nrt_t_list + N_t_list2)
plt.legend(['Pension plan'])
plt.show()

plt.figure(figsize=(10,6),dpi=100)
plt.plot(x, c_t_list + c_t_list2)
plt.legend(['Optimal consumption'])
plt.show()
