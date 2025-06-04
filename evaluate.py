
from tree_anns import TreeANNs
import time
from metric import precision, recall
import gc

def val_test(trainer:TreeANNs, test_queries, test_gts, id_test_queries, id_test_gts, start = 0, stop  = 150, interval = 20, extra_points = (), topk = 10):
    res = {}
    time_cost = {}
    retrieve_cost = {}
    ivf_cost = {}
    ndis = {}
    res2 = {}
    time_cost2 = {}
    retrieve_cost2 = {}
    ivf_cost2 = {}
    ndis2 = {}
    resm = {}
    time_costm = {}
    retrieve_costm = {}
    ivf_costm = {}
    ndism = {}
    lst = list(range(start, stop, interval))
    lst.extend(extra_points)
    lst.sort()
    distri1_len = test_queries.shape[0]
    if not (id_test_queries is None):
        distri2_len = id_test_queries.shape[0]
    else:
        distri2_len = -1
    tot_len = distri1_len + distri2_len
    for i in lst:
        if i == 0:
            i = 1
        print(i,":")
        try:
            st = time.time()
            result_history, retrieve_time, ivf_time, ndis_now = trainer.predict(test_queries,topm=i,num_beams=i,topk=topk,retrieve_batch_size=1000,mode = "layer_wise", distri=0)
            tm = time.time() - st
            st2 = time.time()
            if not (id_test_queries is None):
                result_history_id, retrieve_time_id, ivf_time_id, ndis_now_id = trainer.predict(id_test_queries,topm=i,num_beams=i,topk=topk,retrieve_batch_size=1000,mode = "layer_wise", distri=1)
            else:
                result_history_id, retrieve_time_id, ivf_time_id, ndis_now_id = 0,0,0,0
            tm2 = time.time() - st2
            gc.collect()
        except KeyboardInterrupt as e:
            return res, time_cost, retrieve_cost, ivf_cost, ndis, res2, time_cost2, retrieve_cost2, ivf_cost2, ndis2, resm, time_costm, retrieve_costm, ivf_costm, ndism
        
        r = recall(result_history, test_gts[:,:topk])
        if not (id_test_queries is None):
            r_id = recall(result_history_id, id_test_gts[:,:topk])
        else:
            r_id = 0
        # metrics[f"{i}_recall_10"] = r
        res[i] = r
        res2[i] = r_id
        resm[i] = (r*distri1_len + r_id*distri2_len)/tot_len
        ndis[i] = ndis_now
        ndis2[i] = ndis_now_id
        ndism[i] = (ndis_now*distri1_len + ndis_now_id*distri2_len)/tot_len
        time_cost[i] = tm*1000/distri1_len
        time_cost2[i] = tm2*1000/distri2_len
        time_costm[i] = (tm + tm2)*1000/tot_len
        retrieve_cost[i] = retrieve_time*1000/distri1_len
        retrieve_cost2[i] = retrieve_time_id*1000/distri2_len
        retrieve_costm[i] = (retrieve_time+retrieve_time_id)*1000/tot_len
        ivf_cost[i] = ivf_time*1000/distri1_len
        ivf_cost2[i] = ivf_time_id*1000/distri2_len
        ivf_costm[i] = (ivf_time + ivf_time_id)*1000/tot_len
        perf_str = "recall: "+str(r)+", time: "+str(time_cost[i])+", retrieve_cost: "+str(retrieve_cost[i])+", ivf_cost:"+ str(ivf_cost[i])+", ndis:"+ str(ndis[i])+"\n"
        print(perf_str)
        perf_str = "id: recall: "+str(r_id)+", time: "+str(time_cost2[i])+", retrieve_cost: "+str(retrieve_cost2[i])+", ivf_cost:"+ str(ivf_cost2[i])+ str(ivf_cost[i])+", ndis:"+ str(ndis2[i])+"\n"
        print(perf_str)
        perf_str = "tot: recall: "+str(resm[i])+", time: "+str(time_costm[i])+", retrieve_cost: "+str(retrieve_costm[i])+", ivf_cost:"+ str(ivf_costm[i])+ str(ivf_cost[i])+", ndis:"+ str(ndism[i])+"\n"
        print(perf_str)
    return res, time_cost, retrieve_cost, ivf_cost, ndis, res2, time_cost2, retrieve_cost2, ivf_cost2, ndis2, resm, time_costm, retrieve_costm, ivf_costm, ndism
