import numpy as np
import random
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d,make_interp_spline
import pandas as pd
import seaborn as sns

def chip_novss(path):
    chip_files = ['chiplet1_ac.subckt', 'chiplet2_ac.subckt', 'chiplet3_ac.subckt', 'chiplet4_ac.subckt', 'chiplet5_ac.subckt', 'chiplet6_ac.subckt']
    for chip in chip_files:
        lines = ''   
        with open(path + chip, 'r') as f:
            f_lines = f.readlines()
        strs = ['rs', '+ns', 'rvs', 'is', 'cs', 'xs']
        for line in f_lines:
            flag = 1
            for s in strs:
                if s in line:
                    flag = 0
                    break
            if flag:
                lines += line   
        
        f.close()
        
        with open(path + chip[:-7] + '_novss.subckt', 'w') as f1:
            f1.writelines(lines)
        f1.close()
    cur_list = []
    a = []
    with open(path + chip[:-7] + '_novss.subckt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'ns' in line:
                line = line.replace('ns','')
            if ')' in line:
                line = line.replace(')', ' ')
            if '(' in line:
                line = line.replace('(', ' ')
            if 'pwl' in line:
                cur_list.append(line)

    for i in cur_list:
        a1 = i.split(' ')
        a2 = [strs for strs in a1 if strs != '']
        a.append(a2[4:-1])                              # 提取出电流波形时间和对应的值

    b = []
    for j in a:
        d = []
        for data in j:
            d.append(int(float(data)*10000))
        b.append(d)

    y_val = [0]*10000
    x_time = [i for i in range(1,10001)]
    for currents in b:
        aa = [0] * 10000
        x = currents[::2]
        y = currents[1::2]
    #print(x,y)
        for n in range(len(x)-1):
            for node in range(x[n], x[n+1]):
                aa[node] = (y[n+1]-y[n])/(x[n+1]-x[n])*(node - x[n]) + y[n]
        y_val = [x+y for x, y in zip(y_val, aa)]
    plt.plot(x_time, y_val)
    
def int_ac_novss(path):
    f = open(path + 'interposer_ac.sp', 'r')
    f1 = open(path + 'interposer_ac_novss.sp', 'w')
    strs = ['rs', '+ns', 'xs',]
    for line in f:
        flag = 1
        for s in strs:
            if s in line:
                flag = 0
                break
        if flag:
            f1.write(line)    

    f.close()
    f1.close()

def int_tr_novss(path):
    f = open(path + 'interposer_tr.sp', 'r')
    f1 = open(path + 'interposer_tr_temp.sp', 'w')
    strs = ['rs', '+ns', 'xs',]
    for line in f:
        flag = 1
        for s in strs:
            if s in line:
                flag = 0
                break
        if flag:
            f1.write(line)    

    f.close()
    f1.close()


if __name__ == '__main__':
    path = './'
    chip_novss(path)
    int_ac_novss(path)
    #int_tr_novss(path)


