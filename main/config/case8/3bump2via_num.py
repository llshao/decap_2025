import numpy as np

def modify_ubump2via(path):
    files = ['interposer_ac_novss.sp']
    chip_num = 6

    for ckt in files:
        with open(path + ckt, 'r') as f1:
            lines = f1.readlines()
            n = len(lines)
            for i in range(n):
                for idx in range(chip_num):
                    s = ' nd_chiplet%d_pad' % (idx + 1)
                    if s in lines[i]:
                        lines[i] = lines[i].replace('rd_ubump2via', 'rd%d_ubump2via' % (idx + 1))
            f1.close()

        with open(path + ckt, 'w') as f2:
            f2.writelines(lines)
            f2.close()


if __name__ == '__main__':
    path = './'
    modify_ubump2via(path)