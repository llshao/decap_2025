import os
import numpy as np

filepath = r'./'

file = os.listdir(filepath)

for pic in file:
    new_id = os.path.join(filepath,pic)
    #print(new_id)
    if pic[0].isdigit():
        os.remove(new_id)
    #else:
        #print('baoliu')
