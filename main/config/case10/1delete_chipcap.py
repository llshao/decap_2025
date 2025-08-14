import numpy as np


chip_files = ['chiplet1_ac.subckt', 'chiplet2_ac.subckt', 'chiplet3_ac.subckt', 'chiplet4_ac.subckt', 'chiplet5_ac.subckt', 'chiplet6_ac.subckt']


def delete_chipcap(path):

    for chip in chip_files:
        with open(path + chip, 'r') as f:
            lines = f.readlines()

        with open(path + chip, 'w') as f2:
            for line in lines:
                if 'cd_decap_0' not in line and 'cs_decap_0' not in line and 'cd_add' not in line and 'rd_add' not in line and 'moscap_esr.txt' not in line and 'int_param_dcap.txt' not in line:
                    f2.write(line)


def add_chipcap(path):
    mos = 0
    dcap = 121
    for chip in chip_files:
        with open(path + chip, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if 'W/H:' in line:
                    parts = line.replace('\n', '').split(': ')
                    wh = parts[-1].split('/')
                    w = int(float(wh[0])/1e-3)
                    h = int(float(wh[1])/1e-3)

        with open(path + chip, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if '*-- intentional decap' in line:
                    for j in range(w * h):
                        lines.insert(i + 1 + j, "cd_add_cap%d nd_1_0_X_X nd%d 'dcap_int_val%d'\n" % (j, j, dcap+1))
                        dcap += 1
                    for k in range(w * h):
                        lines.insert(i + 1 + w * h + k, "rd_add%d nd%d 0 'esr%d'\n" % (k, k, mos+1))
                        mos += 1
                    lines.insert(i + 1 + 2 * w * h, "\n.include 'moscap_esr.txt'\n")
                    lines.insert(i + 2 + 2 * w * h, ".include 'int_param_dcap.txt'\n\n")

        with open(path + chip, 'w') as f:
            f.writelines(lines)

if __name__ == '__main__':
    path = './'
    delete_chipcap(path)
    add_chipcap(path)
        
        
        
        
        
