import numpy as np
import os 

def intp_sp_change(path):
    rm_lines = "*-- external power source"
    add_lines = [".include 'int_param_dcap.txt'\n",
    ".include 'full-int-decap.txt'\n\n\n",
    "*-- external power source\n",
    'ivdd 0 port1 0 ac 1.0\n',
    '.ac dec 100 100meg 20g\n',
    '.save vm(port1)\n\n',
    'vdd vdd 0 0.9\n\n',
    '.control\n',
    'set filetype=ascii\n',
    'run\n',
    'wrdata port1_impeval.txt vm(port1)\n',
    'quit\n',
    'plot vm(port1) xlog ylog\n',
    '.endc\n',
    '.end\n',]
    with open(path+'interposer_ac_novss1.sp', 'w') as f1:
        with open(path+'interposer_ac_novss.sp','r') as f2:
            lines = f2.readlines()
            for line in lines:
                if rm_lines in line:
                    break
                elif '_ac.subckt' in line:
                    line2 = line.replace('_ac.subckt', '_ac_novss.subckt')
                    f1.write(line2)
                else:
                    f1.write(line)
        for line in add_lines:
            f1.write(line)

if __name__ == '__main__':
    path = './'
    intp_sp_change(path)












