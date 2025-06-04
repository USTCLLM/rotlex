from dataset import *
import faiss
from utils_faiss import to_faiss_metric
import os
import time
from rwutils import *
import gc
from tqdm import tqdm
import argparse

BATCH_SZ = 1000

def calc_gt(index, path, metric, mode, ds_name, queries, batch_sz = BATCH_SZ):
    print("*"*50, mode, "*"*50,)
    sub_dir = os.path.join(path, f"{mode}_{metric}_gt")
    os.makedirs(sub_dir, exist_ok=True)
    print(f"Output to {sub_dir}, query shape :{queries.shape}")
    bs = batch_sz
    xq = queries
    bnum = (xq.shape[0]-1)//bs + 1
    print(bnum, flush=True)
    for i in range(bnum):
        print("batch ",i,":calc ",i*bs," to ",(i+1)*bs,flush=True)
        t0 = time.time()
        dist, gt = index.search(xq[i*bs:(i+1)*bs], 100)
        name_str = "{}.{}.{}".format(ds_name, mode, str(i*bs)+"-"+str((i+1)*bs))
        ivecs_write(os.path.join(sub_dir, name_str+".ivecs"), gt.astype('int32'))
        del gt
        gc.collect()
        # fvecs_write(os.path.join(sub_dir, name_str+".fvecs"), dist.astype('float32'))
        del dist
        gc.collect()
        print(str((i+1)*bs), " OK", time.time() - t0,flush=True)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset", type=str, default="")
    parser.add_argument("--mode", type=str, default=["train"], choices=["train", "test", "self_train", "self_test"], nargs="+")
    parser.add_argument("-g", "--gpu", type=int, default=-1)
    parser.add_argument("-m","--metric", type=str, default="ip")
    parser.add_argument("-p","--path", type=str, default="./")
    
    parser.add_argument("-n", "--merge_num", type=int, default=0)
    args = parser.parse_args()
    
    if args.merge_num > 0:
        ds = dataset_factory(args.dataset, read_mode="no_read")
        print(ds.dataset_name)
        bnum = (args.merge_num-1)//BATCH_SZ + 1
        with open(f"{ds.dataset_name}.{args.mode[0]}_{args.metric}.ivecs","wb") as w:
            for i in tqdm(range(0, bnum)):
                ph = os.path.join(args.path, os.path.join(args.path, f"{args.mode[0]}_{args.metric}_gt"))
                with open(os.path.join(ph, "{}.{}.{}.ivecs".format(ds.dataset_name, args.mode[0], str(i*BATCH_SZ)+"-"+str((i+1)*BATCH_SZ))),"rb") as f:
                    w.write(f.read())
        exit(0)
    
    ds = dataset_factory(args.dataset, read_mode="only_test")
    if args.path == "":
        args.path = ds.path
    
    xb = ds.data
    nb, d = xb.shape
    print("base shape", xb.shape,flush=True)
    
    index = faiss.IndexFlat(d, to_faiss_metric(args.metric))
    
    if args.gpu >= 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, args.gpu, index)
    index.add(xb)
    
    print("base added")
    
    if "self" not in args.mode:
        del ds.data
        del xb
    
    for x in args.mode:
        if "self_train" == x:
            calc_gt(index, args.path, args.metric, x, ds.dataset_name, xb)
            del ds.data
            del xb
        if "train" == x:
            calc_gt(index, args.path, args.metric, x, ds.dataset_name, ds.train_queries)
            del ds.train_queries
        if "test" == x:
            calc_gt(index, args.path, args.metric, x, ds.dataset_name, ds.test_queries)
        if "self_test" == x:
            calc_gt(index, args.path, args.metric, x, ds.dataset_name, ds.self_test_queries)
        
    