import numpy as np
import os
from sklearn.neighbors import KDTree

import pickle

def load_features(src):
    print("[+] Load data....")
    data = []
    for folder in os.listdir(src):
        folder_path = os.path.join(src, folder)    
        print('folder_path : ', folder_path)
        for file in os.listdir(folder_path):
            data.append(np.load(os.path.join(folder_path, file))[0])
            print('file : ', file)
    print("[+] Load data finished")
    return data


# extracted features so only run one time

data = load_features('./features/train_set')
X = np.array(data)
kdt = KDTree(X, leaf_size=33, metric='euclidean')

filehandler = open('kdtree.pickle', 'wb')
save = pickle.dump(kdt, filehandler)
