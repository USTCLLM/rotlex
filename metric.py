import numpy as np

def precision(result_list,gt_list,top_k):
    count=0.0
    for r,g in zip(result_list,gt_list):
        count+=len(set(r).intersection(set(g)))
    return count/(top_k*len(result_list))

def recall(result_list,gt_list):
    t=0.0
    for r,g in zip(result_list,gt_list):
        # print(r,g)
        t+=1.0*len(set(r).intersection(set(g)))/len(set(g))
    return t/len(result_list)

def f_measure(result_list,gt_list,top_k,eps=1.0e-9):
    f=0.0
    for r,g in zip(result_list,gt_list):
        recc=1.0*len(set(r).intersection(set(g)))/len(g)
        pres=1.0*len(set(r).intersection(set(g)))/top_k
        if recc+pres<eps:
            continue
        f+=(2*recc*pres)/(recc+pres)
    return f/len(result_list)

def novelty(result_list,s_u,top_k):
    count=0.0
    for r,g in zip(result_list,s_u):
        count+=len(set(r)-set(g))
    return count/(top_k*len(result_list))

def hit_ratio(result_list,gt_list):
    intersetct_set=[len(set(r)&set(g)) for r,g in zip(result_list,gt_list)]
    return 1.0*sum(intersetct_set)/sum([len(gts) for gts in gt_list])

def MAP(result_list,gt_list,topk):
    t=0.0
    for re,gt in zip(result_list,gt_list):
        setgt=set(gt)
        indicator=np.asfarray([1 if r in setgt else 0 for r in re])
        t+=np.mean([indicator[:i].sum(-1)/i for i in range(1,topk+1)],axis=-1)
    return t/len(gt_list)

def jaccard(r1, r2):
    t = 0
    for r,g in zip(r1,r2):
        t+=1.0*len(set(r).intersection(set(g)))/len(set(r).union(set(g)))
    return t/len(r1)
