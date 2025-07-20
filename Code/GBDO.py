import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.io import loadmat

from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from GB_generation import GB_Gen


def calculate_center_and_radius(gb):    
    center = gb.mean(axis=0)
    radius = None
    return center, radius

def assign_points_to_closest_gb(data, gb_centers):
    assigned_gb_indices = np.zeros(data.shape[0])
    for idx, sample in enumerate(data):
        t_idx = np.argmin(np.sqrt(np.sum((sample - gb_centers) ** 2, axis=1)))
        assigned_gb_indices[idx] = t_idx
    
    return assigned_gb_indices.astype('int')

def GB_Generate(data):
    n, m = data.shape
    gb_list = GB_Gen(data)
    n_gb = len(gb_list)
    gb_center = np.zeros((n_gb, m))
    for idx, gb in enumerate(gb_list):
        gb_center[idx], _ = calculate_center_and_radius(gb)

    point_to_gb = assign_points_to_closest_gb(data, gb_center)
    center_dist = cdist(gb_center, gb_center)
    fuzzy_similarity = 1 - center_dist / m
    return gb_list, point_to_gb, center_dist, None, fuzzy_similarity


def GBLOF(gb_list, point_to_gb, center_dist, gb_radius, fuzzy_similarity, k):
    n_gb, n = len(gb_list), len(point_to_gb)
    fuzzy_idxs = np.argsort(-fuzzy_similarity, axis=1)
    k_fuzzy = fuzzy_similarity[np.arange(n_gb), fuzzy_idxs[:, k]]

    count_temp = fuzzy_similarity >= k_fuzzy[:, np.newaxis]
    count = np.sum(count_temp, axis=1)

    k_fuzzy_temp = fuzzy_similarity.copy()
    k_fuzzy_temp[k_fuzzy_temp < k_fuzzy[:, np.newaxis]] = 0
    k_fuzzy_card = np.sum(k_fuzzy_temp, axis=1)
    
    reach_similarity = np.zeros((n_gb, n_gb))
    for i in range(n_gb):
        for j in range(n_gb):
            reach_similarity[i, j] = min(k_fuzzy[j], fuzzy_similarity[i, j])
            
    lrd = np.zeros(n_gb)
    for i in range(n_gb):
        s = 0
        for j in range(count[i]):
            j_p = fuzzy_idxs[i, j]
            s += reach_similarity[i, j_p]
        lrd[i] = s / k_fuzzy_card[i]

    LOF = np.zeros(n)
    for i in range(n):
        t = 0
        i_gb = point_to_gb[i]
        cnt = count[i_gb]
        for j in range(cnt):
            j_gb = fuzzy_idxs[i_gb, j]
            t += lrd[j_gb] / lrd[i_gb]
        LOF[i] = t / k_fuzzy_card[i_gb]

    return LOF


if __name__ == '__main__':
    load_data = loadmat('./breast_cancer_variant1.mat')
    trandata = load_data['trandata']
    trandata = trandata.astype(float)
    ID = (trandata >= 1).all(axis=0) & (
        trandata.max(axis=0) != trandata.min(axis=0))
    scaler = MinMaxScaler()
    if any(ID):
        trandata[:, ID] = scaler.fit_transform(trandata[:, ID])
    X = trandata[:, 0:-1]
    labels = trandata[:, -1]
    gb_list, point_to_gb, center_dist, gb_radius, fuzzy_similarity = GB_Generate(X)
    print(len(gb_list))
    OF = GBLOF(gb_list, point_to_gb, center_dist, gb_radius, fuzzy_similarity, 10)
    print(roc_auc_score(labels, OF))
