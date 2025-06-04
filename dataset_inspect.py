from dataset import *
from tqdm.contrib import tenumerate
from tqdm import tqdm
import numpy as np
import multiprocessing

mono_type = {"ip":"desc", "l2":"asc"}

def vvs_dist(v, v_s, metric):
    if metric == "ip":
        dist = np.squeeze(v.reshape(1,-1) @ v_s.T)
    elif metric == "l2":
        dist = ((v.reshape(1,-1) - v_s)**2).sum(axis = -1).reshape(-1) # l2 sqr
    return dist

def check_monotonic_single(i ,v, v_gts, metric, eps):
    dist = vvs_dist(v, v_gts, metric)    
    tp = mono_type[metric] 
    diff = np.diff(dist)
    if tp == "desc":
        sig = diff <= eps # desc
    elif tp == "asc":
        sig = diff >= -eps # asc
    if not np.all(sig):
        print(f"check fail at {i}: ", end = " ")
        # print(dist)
        pos = np.where(~sig)[0]
        # print(diff, pos, diff[pos])
        print(f"not {tp}, diff(len {len(pos)}) {diff[pos]}, eps {eps}")
        if len(pos)>1 or np.fabs(diff[pos[0]]).item() > 5*eps:
            print(dist)
            exit()
    return

def unpacked_check_monotonic_single(args):
    return check_monotonic_single(*args)

def check_monotonic(vec, gts, data, metric, eps = 1e-5, para = True):
    print(f"Metric is {metric}, para is {para} !!!", flush=True)
    if not para:
        for i, (v, g) in tqdm(enumerate(zip(vec, gts)), total=vec.shape[0]):
            check_monotonic_single(i, v, data[g], metric, eps)
    else:
        def args_generator():
            for i, (v, g) in enumerate(zip(vec, gts)):
                yield (i, v, data[g], metric, eps)
        num_processes = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=num_processes)
        
        for _ in tqdm(pool.imap(unpacked_check_monotonic_single, args_generator()), total=vec.shape[0]):
            pass

        pool.close()
        pool.join()
        
def check_dataset_monotonic(ds: AnnDatasetSelfTrain, metric = None, para = True):
    """Check if the dataset is correct"""
    if not metric or metric == "":
        metric = ds.metric
    print(f"Metric is {metric}, para is {para}")
    print("Test mono checking ...")
    # print(ds.test_gts[0])
    check_monotonic(ds.test_queries, ds.test_gts, ds.data, metric, para = para)
    # return
    if ds.train_queries is not None:
        print("Train mono checking ...")
        check_monotonic(ds.train_queries, ds.train_gts, ds.data, metric, para = para)
    else:
        print("No train_queries or not loaded")
    if isinstance(ds, AnnDatasetSelfTrain):
        if ds.self_train_gts is not None:
            print("Train_self mono  checking ...")
            check_monotonic(ds.data[:ds.self_train_gts.shape[0]], ds.self_train_gts, ds.data, metric, para = para)
        else:
            print("No self_train_gts or not loaded")

def is_normed(matrix, eps=1e-3):
    row_l2sqr = (matrix**2).sum(axis = -1)
    # row_l2 = np.linalg.norm(matrix, ord = ord, axis=1)
    print("avg of norm: ", np.average(row_l2sqr))
    print("max of norm: ", np.max(row_l2sqr))
    print("min of norm: ", np.min(row_l2sqr))
    print("std of norm : ", np.std(row_l2sqr))
    delta = np.fabs(row_l2sqr - np.ones_like(row_l2sqr))
    print("avg of err : ", np.average(delta))
    return np.all(delta <= eps)

def check_dataset_l2norm(ds: AnnDatasetSelfTrain):
    
    print("test query is normed:", is_normed(ds.test_queries))
    print("base is normed:", is_normed(ds.data))
    if ds.train_queries is not None:
        print("train query is normed:", is_normed(ds.train_queries))
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset", type=str, default="")
    
    parser.add_argument("--mono", action="store_true")
    parser.add_argument("--para", action="store_true")
    parser.add_argument("-m","--metric", type=str, default="")
    parser.add_argument("-l","--load", type=str, default="all", choices= ["all", "only_test", "query", "self", "only_vecs"])
    
    parser.add_argument("--norm", action="store_true")
    
    args = parser.parse_args()
    
    ds = dataset_factory(args.dataset, read_mode=args.load)
    if args.norm:
        check_dataset_l2norm(ds)
        
    if args.mono:
        check_dataset_monotonic(ds, args.metric, args.para)
    