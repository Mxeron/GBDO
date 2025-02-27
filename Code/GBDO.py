import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.cluster import k_means
from scipy.io import loadmat
from scipy.spatial.distance import cdist


def calculate_center_and_radius(gb):    
    center = gb.mean(axis=0)
    radius = np.mean(np.sqrt(np.sum((gb - center) ** 2, axis=1)))
    return center, radius


def splits(gb_list, num, k=2):
    gb_list_new = []
    for gb in gb_list:
        p = gb.shape[0]
        if p < num:
            gb_list_new.append(gb)
        else:
            gb_list_new.extend(splits_ball(gb, k))
    return gb_list_new


def splits_ball(gb, k):
    ball_list = []
    len_no_label = np.unique(gb, axis=0)
    if len_no_label.shape[0] < k:
        k = len_no_label.shape[0]
    label = k_means(X=gb, n_clusters=k, n_init=1, random_state=8)[1]

    for single_label in range(0, k):
        ball_list.append(gb[label == single_label, :])
    return ball_list

def assign_points_to_closest_gb(data, gb_centers):
    assigned_gb_indices = np.zeros(data.shape[0])
    for idx, sample in enumerate(data):
        t_idx = np.argmin(np.sqrt(np.sum((sample - gb_centers) ** 2, axis=1)))
        assigned_gb_indices[idx] = t_idx
    return assigned_gb_indices.astype('int')

def GB_Generate(data):
    n, m = data.shape
    
    gb_list = [data]
    num = np.ceil(0.3*(n ** 0.5))
    while True:
        ball_number_1 = len(gb_list)
        gb_list = splits(gb_list, num=num, k=2)
        ball_number_2 = len(gb_list)
        if ball_number_1 == ball_number_2:
            break

    n_gb = len(gb_list)
    gb_center = np.zeros((n_gb, m))
    gb_radius = np.zeros(n_gb)
    for idx, gb in enumerate(gb_list):
        gb_center[idx], gb_radius[idx] = calculate_center_and_radius(gb)

    point_to_gb = assign_points_to_closest_gb(data, gb_center)
    center_dist = cdist(gb_center, gb_center)
    fuzzy_similarity = 1 - center_dist / m
    return gb_list, point_to_gb, center_dist, gb_radius, fuzzy_similarity

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

    OF = np.zeros(n)
    for i in range(n):
        t = 0
        i_gb = point_to_gb[i]
        cnt = count[i_gb]
        for j in range(cnt):
            j_gb = fuzzy_idxs[i_gb, j]
            t += lrd[j_gb] / lrd[i_gb]
        OF[i] = t / k_fuzzy_card[i_gb]

    return OF

if __name__ == '__main__':
    df = pd.read_csv("./Datasets/cardio.csv")
    cur_data = df.values

    X = cur_data[:, :-1]
    label = cur_data[:, -1]
    gb_list, point_to_gb, center_dist, gb_radius, fuzzy_similarity = GB_Generate(X)
    OF = GBLOF(gb_list, point_to_gb, center_dist, gb_radius, fuzzy_similarity, 15)
