import os

if __name__ == '__main__':
    path = './'
    target_file = ['pwr', 'vdd.ipulse', 'vss.decap', 'vss.ipulse', '.cpd', 'vdd.decap']


    for root, dirs, files in os.walk(path):
        for f in files:
            for strs in target_file:
                if strs in f:
                    file_path = os.path.join(root, f)
                    os.remove(file_path)
                    break
