import numpy as np
import random
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d,make_interp_spline
import pandas as pd
import seaborn as sns

def copy_files(path, num):
    for j in range(num-1):
        writelines = []
        with open(path + 'interposer_ac_novss1.sp', 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if '*-- pcb instances' in line:
                    line_index = i
            for i, line in enumerate(lines):
                if i > line_index:
                    line = line.replace('port1', 'port%d' % (j + 2))
                writelines.append(line)
        with open(path + 'interposer_ac_novss%d.sp'%(j+2), 'w') as f2:
            f2.writelines(writelines)

if __name__ == '__main__':
    path = './'
    copy_files(path, 4)


