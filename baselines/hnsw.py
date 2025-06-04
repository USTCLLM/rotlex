import sys
import os
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)
from utils_faiss import to_faiss_metric
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
# os.environ["OMP_NUM_THREADS"] = "16"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset","-d", type=str, default="")
    parser.add_argument("--load","-l", type=str, default="")
    parser.add_argument("-M", type=int, default=20)
    parser.add_argument("--efc","-c", type=int, default=200)
    parser.add_argument("--build","-b", action="store_true")
    parser.add_argument("-m","--metric", type=str, default="l2")
    parser.add_argument("-x","--points", type=int, default=[16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 5120, 6144], nargs="+")
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
    
    ds.read(False, False)
    if args.load == "":
        logdir = f"baselines_logs/hnsw_{ds.dataset_name}_efc{args.efc}_m{args.M}"
        logger.add(logdir + "/{time}.log")
    else:
        logdir = os.path.dirname(args.load)
        
    logger.add(logdir + "/{time}.log")
    d = 200 # ds.data.shape[1]
    bs = 100000 # 1e7 
    if args.build:
        st = time.time()
        index = faiss.IndexHNSWFlat(d, args.M, to_faiss_metric(args.metric))
        logger.debug("metric {}(L2 {}, IP {})", index.metric_type, to_faiss_metric("l2"), to_faiss_metric("ip"))
        index.hnsw.efConstruction = args.efc
        index.verbose = True
        with open(ds.path+"/"+ds.base_fn, "rb") as f_in:
            cnt = 0
            f_in.seek(8)
            while True:
                batch_data = np.fromfile(f_in, dtype=np.float32, count=bs * d)
            
                # If no more data, break the loop
                if batch_data.size == 0:
                    break
                print(batch_data.shape)

                index.add(batch_data.reshape(-1, d))
                print(cnt)
        try:
            write_index(index, f"{logdir}/{ds.dataset_name}_hnsw_efc{args.efc}_m{args.M}.index")
        except Exception as e:
            logger.debug(str(e))
        logger.debug("build time: {}", time.time() - st)

    if args.load != "":
        index:faiss.IndexHNSWFlat = read_index(args.load)
        logger.debug("metric load {}(L2 {}, IP {})", index.metric_type, to_faiss_metric("l2"), to_faiss_metric("ip"))
        # index.metric_type = to_faiss_metric(args.metric)
        # logger.debug("metric {}", index.metric_type)
    # exit(0)    
    stats:faiss.HNSWStats = faiss.cvar.hnsw_stats
    test_queries = ds.test_queries
    test_gts = ds.test_gts
    self_test_queries = ds.self_test_queries
    self_test_gts = ds.self_test_gts
    bs = os.cpu_count()
    nns = 10
    with open(f"{logdir}/{ds.dataset_name}_hnsw_efc{args.efc}_m{args.M}_{time_str}.json", "w") as f:
        time_cost = {}
        ndis = {}
        res = {}
        time_cost2 = {}
        ndis2 = {}
        res2 = {}
        time_tot = {}
        recall_tot = {}
        res_tot = {}
        for efSearch in args.points: #  
            index.hnsw.efSearch = efSearch
            i = efSearch
            st = time.time()
            stats.reset()
            # D, I = index.search(test_queries, nns)
            I = []
            bnum = (test_queries.shape[0]-1)//bs +1
            for j in range(bnum):
                Dmow, Inow = index.search(test_queries[bs*j:bs*(j+1)], nns)
                I.append(Inow)
            I = np.concatenate(I)
            tm = time.time() - st
            r = recall(I,test_gts[:,:nns])
            res[i] = r
            time_cost[i] = tm*1000/test_queries.shape[0]
            # retrieve_cost[i] = stats.quantization_time/test_queries.shape[0]
            # search_cost[i] = stats.search_time/test_queries.shape[0]
            ndis[i] = stats.ndis/test_queries.shape[0]
            perf = str(r)+","+str(tm*1000/test_queries.shape[0])+","+str(ndis[i])+"\n"
            print(perf)
            # break
        for efSearch in args.points:
            index.hnsw.efSearch = efSearch
            i = efSearch
            st = time.time()
            stats.reset()
            # D, I = index.search(self_test_queries, nns)
            I = []
            bnum = (self_test_queries.shape[0]-1)//bs +1
            for j in range(bnum):
                Dmow, Inow = index.search(self_test_queries[bs*j:bs*(j+1)], nns)
                I.append(Inow)
            I = np.concatenate(I)
            tm = time.time() - st
            r = recall(I, self_test_gts[:, :nns])
            res2[i] = r
            time_cost2[i] = tm * 1000 / self_test_queries.shape[0]
            ndis2[i] = stats.ndis / self_test_queries.shape[0]

            # Debug output
            perf = str(r) + "," + str(tm * 1000 / self_test_queries.shape[0]) + "," + str(ndis2[i]) + "\n"
            print(perf)
        total_queries = test_queries.shape[0] + self_test_queries.shape[0]
        for efSearch in args.points:
            time_tot[efSearch] = (
                time_cost[efSearch] * test_queries.shape[0] +
                time_cost2[efSearch] * self_test_queries.shape[0]
            ) / total_queries
            recall_tot[efSearch] = (
                res[efSearch] * test_queries.shape[0] +
                res2[efSearch] * self_test_queries.shape[0]
            ) / total_queries
        
        
        json.dump(
            {
                "time": time_cost,
                "recall": res,
                "ndis": ndis,
                "time_id": time_cost2,
                "recall_id": res2,
                "ndis_id": ndis2,
                "time_tot": time_tot,
                "recall_tot": recall_tot,
            },
            f
        )