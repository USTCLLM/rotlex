import sys
import os
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)

import argparse
import faiss
from dataset import *
import time
from faiss import IndexIVFStats
from metric import recall
import json
import numpy as np
from faiss import write_index, read_index
from loguru import logger
import datetime
from utils_faiss import to_faiss_metric
from utils import calculate_cluster_centers
from faiss.contrib.ivf_tools import add_preassigned
# os.environ["OMP_NUM_THREADS"] = "16"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset","-d", type=str, default="")
    parser.add_argument("--load","-l", type=str, default="")
    parser.add_argument("--rebuild","-r", type=str, default="")
    parser.add_argument("--nlist","-n", type=int, default=65536)
    parser.add_argument("--build","-b", action="store_true")
    parser.add_argument("--device","-i", type=str, default="cpu")
    parser.add_argument("--cuda_lst","-c", type=int, default=[], nargs="+")
    parser.add_argument("-m","--metric", type=str, default="ip")
    lst = list(range(50, 550, 50))
    lst.extend((1,5,10,25))
    lst.sort()
    # lst = [47,48,49,50,51,52,53,95,96,97,98,99,100,101,102,103,104,105]
    parser.add_argument("-x","--points", type=int, default=lst, nargs="+")
    args = parser.parse_args()
    
    time_str = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    
    if args.dataset in dataset_dict:
        ds_class = dataset_dict[args.dataset]
        if issubclass(ds_class, AnnDatasetSelfTrain):
            ds:AnnDatasetSelfTrain = ds_class()
        else:
            ds:AnnDataset = ds_class()
    else:
        raise NotImplementedError(args.dataset)
    ds.path = os.path.join(ROOT_PATH, ds.path)
    
    if args.load == "":
        if args.build:
            logdir = f"baselines_logs/ivfflat_{ds.dataset_name}_nlist{args.nlist}"
            
        elif args.rebuild != "":
            logdir = os.path.dirname(args.rebuild)
        else:
            print("do nothing")
            exit()
    else:
        logdir = os.path.dirname(args.load)
    
    logger.add(logdir + "/{time}.log")
    
    ds.read(False)
    
    d = ds.data.shape[1]
    if args.build:
        st = time.time()
        quantizer = faiss.IndexFlat(d, to_faiss_metric(args.metric)) 
        index = faiss.IndexIVFFlat(quantizer, d, args.nlist, to_faiss_metric(args.metric))
        index.verbose = True
        index.cp.niter = 10
        index.train(ds.data)
        logger.debug("train time: {}", time.time() - st)
        write_index(index, f"{logdir}/{ds.dataset_name}_ivfflat_nlist{args.nlist}_noadd.index")
        index.add(ds.data)
        try:
            write_index(index, f"{logdir}/{ds.dataset_name}_ivfflat_nlist{args.nlist}.index")
        except Exception as e:
            logger.debug(str(e))
        logger.debug("build time: {}", time.time() - st)

    if args.rebuild != "":
        st = time.time()
        bucket_order = np.fromfile(args.rebuild, dtype=np.int64)
        bucket_num = int(bucket_order.max() +1) 
        logger.debug("bucket num {}", bucket_num)
        centroids_data = calculate_cluster_centers(ds.data, bucket_order, bucket_num)
        logger.debug("calc centor time: {}", time.time() - st)
        quantizer = faiss.IndexFlat(d, to_faiss_metric(args.metric))
        quantizer.add(centroids_data)
        index = faiss.IndexIVFFlat(quantizer, d, bucket_num, to_faiss_metric(args.metric))
        index.is_trained = True
        index.verbose = True
        if False:
            index.add(ds.data)
        else:
            add_preassigned(index, ds.data, bucket_order)    
        logger.debug("rebuild time: {}", time.time() - st)
        try:
            print(f"{logdir}/{ds.dataset_name}_ivfflat_nlist{bucket_num}.index")
            write_index(index, f"{logdir}/{ds.dataset_name}_ivfflat_nlist{bucket_num}.index")
        except Exception as e:
            logger.debug(str(e))
    
    if args.load != "":
        index = read_index(args.load)
        
    if args.device.startswith("cuda"):
        if len(args.cuda_lst) <=1:
            if len(args.cuda_lst) == 0:
                idx = int(args.device.split(":")[1])
            else:
                idx = args.cuda_lst[0]
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, idx, index)
        else:
            res = faiss.StandardGpuResources()
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            index = faiss.index_cpu_to_gpus_list(index, co, gpus=args.cuda_lst)

        print(index)
        
    stats:IndexIVFStats = faiss.cvar.indexIVF_stats
    test_queries = ds.test_queries
    test_gts = ds.test_gts
    self_test_queries = ds.self_test_queries
    self_test_gts = ds.self_test_gts
    i = 0
    bs = os.cpu_count()
    with open(f"{logdir}/{ds.dataset_name}_ivfflat_{time_str}.json", "w") as f:
        time_cost = {}
        search_cost = {}
        retrieve_cost = {}
        ndis = {}
        res = {}
            
        time_cost2 = {}
        search_cost2 = {}
        retrieve_cost2 = {}
        ndis2 = {}
        res2 = {}

        time_tot = {}
        recall_tot = {}
        retrieve_tot = {}
        search_tot = {}
        ndis_tot = {}

        try:
            # First loop: process test_queries
            for i in args.points:
                if i == 0:
                    i = 1
                
                if isinstance(index, faiss.IndexShardsIVF):
                    for j in range(len(args.cuda_lst)):
                        faiss.downcast_index(index.at(j)).nprobe = i
                else:
                    index.nprobe = i
                D, I = index.search(test_queries, 10)
                st = time.time()
                stats.reset()
                I = []
                bnum = (ds.test_queries.shape[0]-1)//bs +1
                for j in range(bnum):
                    Dmow, Inow = index.search(ds.test_queries[bs*j:bs*(j+1)], 10)
                    I.append(Inow)
                I = np.concatenate(I)
                tm = time.time() - st
                r = recall(I, test_gts[:, :10])
                res[i] = r
                time_cost[i] = tm * 1000 / test_queries.shape[0]
                retrieve_cost[i] = stats.quantization_time / test_queries.shape[0]
                search_cost[i] = stats.search_time / test_queries.shape[0]
                ndis[i] = stats.ndis / test_queries.shape[0]

                # Debug output
                perf = str(i) + ":" + str(r) + "," + str(tm * 1000 / test_queries.shape[0]) + "," + str(ndis[i]) + "\n"
                print(perf)

            # Second loop: process self_test_queries
            for i in args.points:
                if i == 0:
                    i = 1
                if isinstance(index, faiss.IndexShardsIVF):
                    for j in range(len(args.cuda_lst)):
                        faiss.downcast_index(index.at(j)).nprobe = i
                else:
                    index.nprobe = i
                
                D, I = index.search(self_test_queries, 10)
                st = time.time()
                stats.reset()
                
                I = []
                bnum = (ds.self_test_queries.shape[0]-1)//bs +1
                for j in range(bnum):
                    Dmow, Inow = index.search(ds.self_test_queries[bs*j:bs*(j+1)], 10)
                    I.append(Inow)
                I = np.concatenate(I)
                tm = time.time() - st
                r = recall(I, self_test_gts[:, :10])
                res2[i] = r
                time_cost2[i] = tm * 1000 / self_test_queries.shape[0]
                retrieve_cost2[i] = stats.quantization_time / self_test_queries.shape[0]
                search_cost2[i] = stats.search_time / self_test_queries.shape[0]
                ndis2[i] = stats.ndis / self_test_queries.shape[0]

                # Debug output
                perf = str(i) + ":" + str(r) + "," + str(tm * 1000 / self_test_queries.shape[0]) + "," + str(ndis2[i]) + "\n"
                print(perf)

            # Calculate weighted averages (time_tot, recall_tot, etc.)
            total_queries = test_queries.shape[0] + self_test_queries.shape[0]
            for i in args.points:
                if i == 0:
                    i = 1
                time_tot[i] = (
                    time_cost[i] * test_queries.shape[0] +
                    time_cost2[i] * self_test_queries.shape[0]
                ) / total_queries
                recall_tot[i] = (
                    res[i] * test_queries.shape[0] +
                    res2[i] * self_test_queries.shape[0]
                ) / total_queries
                retrieve_tot[i] = (
                    retrieve_cost[i] * test_queries.shape[0] +
                    retrieve_cost2[i] * self_test_queries.shape[0]
                ) / total_queries
                search_tot[i] = (
                    search_cost[i] * test_queries.shape[0] +
                    search_cost2[i] * self_test_queries.shape[0]
                ) / total_queries
                ndis_tot[i] = (
                    ndis[i] * test_queries.shape[0] +
                    ndis2[i] * self_test_queries.shape[0]
                ) / total_queries

            # Save results to JSON
            json.dump(
                {
                    "time": time_cost,
                    "recall": res,
                    "retrieve_time": retrieve_cost,
                    "search_time": search_cost,
                    "ndis": ndis,
                    "time_id": time_cost2,
                    "recall_id": res2,
                    "retrieve_time_id": retrieve_cost2,
                    "search_time_id": search_cost2,
                    "ndis_id": ndis2,
                    "time_tot": time_tot,
                    "recall_tot": recall_tot,
                    "retrieve_time_tot": retrieve_tot,
                    "search_time_tot": search_tot,
                    "ndis_tot": ndis_tot,
                },
                f
            )

        except KeyboardInterrupt as e:
            json.dump(
                {
                    "time": time_cost,
                    "recall": res,
                    "retrieve_time": retrieve_cost,
                    "search_time": search_cost,
                    "ndis": ndis,
                    "time_id": time_cost2,
                    "recall_id": res2,
                    "retrieve_time_id": retrieve_cost2,
                    "search_time_id": search_cost2,
                    "ndis_id": ndis2,
                    "time_tot": time_tot,
                    "recall_tot": recall_tot,
                    "retrieve_time_tot": retrieve_tot,
                    "search_time_tot": search_tot,
                    "ndis_tot": ndis_tot,
                },
                f
            )