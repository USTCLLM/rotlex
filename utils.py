import torch
import os
import random
import numpy as np
import math
from joblib import Parallel, delayed

def calculate_cluster_centers(X, bucket_order, bucket_num, num_processes=os.cpu_count()):
    def calculate_cluster_center(i):
        indices = np.where(bucket_order == i)[0]
        return np.mean(X[indices], axis=0)

    centers = Parallel(n_jobs=num_processes)(delayed(calculate_cluster_center)(i) for i in range(bucket_num))
    cluster_centers = np.array(centers)
    return cluster_centers

def calculate_cluster_centers_slow(X, bucket_order, bucket_num):

    cluster_centers = np.zeros((bucket_num, X.shape[1]))

    for i in range(bucket_num):
        indices = np.where(bucket_order == i)[0]
        cluster_centers[i] = np.mean(X[indices], axis=0)
    return cluster_centers

def set_seed(seed = 2000, deterministic = True):
    """
    links: https://github.com/pytorch/pytorch/issues/7068
    :param seed: random seed
    :return: None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def l2(x, y):
    return math.sqrt(((x-y)**2).sum())