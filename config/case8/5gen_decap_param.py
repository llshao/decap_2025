
intp_x = [0, 909, 1818, 2727, 3636, 4545, 5454, 6363, 7272, 8181, 9090]
intp_y = [0, 909, 1818, 2727, 3636, 4545, 5454, 6363, 7272, 8181, 9090]

def full_int_name(path, nodecap):			# interposer MIM cap location
    s = ''
    a = 0
    for i in range(len(intp_x)):
        for j in range(len(intp_y)):
            if (i, j) not in nodecap:
                a += 1
                single_s = '\'dcap_int_val%d\'' % a
                s += 'cd_int_%d_%d ndint_xy_%d_%d 0 %s\n' % (i, j, intp_x[j], intp_y[i], single_s)

    f = open(path + 'full-int-decap.txt', 'w')
    f.write(s)
    f.close()

def init_esr_val(path, num_mos):
    s = ''
    cap_num = num_mos
    for k in range(1, cap_num + 1):
        s += '.param esr%d=0\n' % k

    f = open(path + 'moscap_esr.txt', 'w')
    f.write(s)
    f.close()

def init_cap_val(path, NCAP):		# initial mim cap val
    s = ''
    cap_num = NCAP
    for k in range(1, cap_num+1):
        s += '.param dcap_int_val%d=0p\n' % k
    
    f = open(path + 'int_param_dcap.txt', 'w')
    f.write(s)
    f.close()

if __name__ == '__main__':
    path = './'
    # nodecap = [(1,5),(2,5),(3,5),(4,4),(4,6),(5,4),(5,5),(5,6),(6,4),(6,6),(7,5),(8,5),(9,5)]
    nodecap = []
    num_mim = 121
    num_mos = 42
    NCAP = num_mim + num_mos
    full_int_name(path, nodecap)
    init_esr_val(path, num_mos)
    init_cap_val(path, NCAP)
    
