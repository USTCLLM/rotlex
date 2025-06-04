import faiss
import math
import torch
from collections import deque
import numpy as np
from utils import calculate_cluster_centers
from loguru import logger
import time
from utils_faiss import to_faiss_metric

def pairwise_distance_function(data1, data2,metric='l2'):
    #data1 [m,dim],data2 [k,dim]
    if metric=='l2':
        return  ((data1.unsqueeze(dim=1)-data2) ** 2.0).sum(-1)#[m,k]
    elif metric=='cosine':
        ip=(data1@data2.transpose(0,1))
        data1_norm=torch.norm(data1,p=2,dim=-1).unsqueeze(1)#[m,1]
        data2_norm=torch.norm(data2,p=2,dim=-1).unsqueeze(0)#[1,k]
        return 1.0 - ip/(torch.matmul(data1_norm,data2_norm))
    elif metric=='ip':
        return -(data1.unsqueeze(dim=1)*data2).sum(-1)#[m,k]
    else:
        assert False,'error!! wrong distance metric'
        
def initialize(X, num_clusters):
    indices = torch.randperm(X.shape[0], device=X.device)
    return X[indices].view(num_clusters, -1, X.shape[-1]).mean(dim=1)#[num_clusters,dim]

def kmeans_equal(
        X,
        num_clusters=2,
        cluster_size=10,
        max_iters=100,
        initial_state=None,
        update_centers=True,
        tol=1e-6,
        metric='cosine'):
    assert X.shape[0]==num_clusters*cluster_size,'data point size should be the product of num_clusters and cluster_size'
    if initial_state is None:
        # randomly group vectors to clusters (forgy initialization)
        initial_state = initialize(X, num_clusters)##[num_clusters,dim]
    iteration = 0

    final_choice=torch.full((X.shape[0],),-1,dtype=torch.int64,device=X.device)
    left_index=torch.full((X.shape[0],),True,dtype=torch.bool,device=X.device)
    all_ins_ids=torch.arange(X.shape[0],device=X.device)
    while True:
        #choices is [num_sample,num_cluster],remark the cluster rank
        #start_t = time.time()
        choices = torch.argsort(pairwise_distance_function(X, initial_state,metric=metric), dim=-1)
        #print(time.time()-start_t)
        
        initial_state_pre = initial_state.clone()
        left_index[:]=True
        for index in torch.randperm(num_clusters):
            cluster_positions = torch.argmax((choices[left_index] == index).to(torch.long), dim=-1)#cluster_positions is [left_num_sample]

            #choose the most colse cluster_size samples to cluster index,selected_ind is [cluster_size]
            selected_ind=all_ins_ids[left_index].gather(dim=0,index=torch.argsort(cluster_positions, dim=-1)[:cluster_size])
            #print(selected_ind)

            final_choice.scatter_(0, selected_ind, value=index)
            left_index.scatter_(0,selected_ind,value=False)
            # update cluster center

            if update_centers:#initial_state is [num_clusters,dim]
                initial_state[index] = torch.gather(X, 0, index=selected_ind.view(-1,1).expand(cluster_size,X.shape[1])).mean(dim=0)
        center_shift =torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)).sum()

        # increment iteration
        iteration = iteration + 1

        if center_shift ** 2 < tol:
            break
        if iteration >= max_iters:
            break

    return final_choice, initial_state



class Tree:
    def __init__(self, data, k = 64, tree_height=2, max_iters=100, record_center = False, init = True, init_method = "kmeans", metric = "ip", spherical = False):
        self.tree_height = tree_height
        self.k = k
        self.bucket_num = k**tree_height
        self.max_iters = max_iters
        self.metric = metric
        self.bucket_to_path=torch.empty((self.bucket_num,self.tree_height), dtype=torch.int64) # bucket order to path
        self.card = torch.zeros(self.tree_height,dtype=torch.int64)
        for i in range(self.tree_height):
            self.card[i] = self.k ** (self.tree_height - i - 1)
        self.layer_st_ed = []
        for i in range(tree_height):
            st = self.card[tree_height-i-1:].sum().item()
            ed = st + self.card[tree_height-i-1].item()*k
            self.layer_st_ed.append((st, ed))
        self.num_code = (self.bucket_num*k-1)//(k-1)
        
        self.record_center = record_center
        if self.record_center:
            self.code2ct = torch.empty((self.num_code, data.shape[1]), dtype=torch.float32)
        
        for b_order, code in enumerate(range(self.card.sum(), self.card.sum()*k+1)):
            reverse_path = []
            for _ in range(self.tree_height):
                reverse_path.append((code-1)%self.k)
                code=(code-1)//self.k
            self.bucket_to_path[b_order] = torch.LongTensor(reverse_path[::-1])
        
        # to direct label
        for i in range(1, self.bucket_to_path.shape[1]):
            self.bucket_to_path[:,i] = ((self.bucket_to_path[:,:i+1]+1)*self.card[self.tree_height-i-1:]).sum(dim = 1) - self.layer_st_ed[i][0]

        if init:
            if init_method == "kmeans":
                if False and data.shape[0] < 10000000: # slower but a litte better
                    kmeans = faiss.Kmeans(data.shape[1], self.bucket_num, spherical = spherical, niter = max_iters, verbose = True)
                    logger.debug("niter : {}", kmeans.cp.niter)
                    logger.debug("spherical is set to: {}", kmeans.cp.spherical)
                    kmeans.train(data)
                    centroids_data = torch.from_numpy(kmeans.centroids)
                    _, data_bucket_id = kmeans.index.search(data,1)
                    data_bucket_id = data_bucket_id[:,0]
                else:
                    quantizer = faiss.IndexFlat(data.shape[1], to_faiss_metric(metric)) 
                    kmeans = faiss.IndexIVFFlat(quantizer, data.shape[1], self.bucket_num, to_faiss_metric(metric)) # if metric is spherical is open in faiss 
                    logger.debug("kmeans metric :{}, spherical : {}", metric, kmeans.cp.spherical)
                    kmeans.verbose = True
                    kmeans.cp.verbose = True
                    kmeans.cp.niter = max_iters
                    kmeans.cp.spherical = spherical
                    logger.debug("spherical is set to: {}", kmeans.cp.spherical)
                    logger.debug("niter : {}", kmeans.cp.niter)
                    st = time.time()
                    kmeans.train(data)
                    logger.debug("kmeans time : {}", time.time() - st)
                    st = time.time()
                    centroids_data = np.empty((self.bucket_num, data.shape[1]), dtype=np.float32)
                    kmeans.quantizer.reconstruct_n(0, self.bucket_num, centroids_data)
                    centroids_data = torch.from_numpy(centroids_data) # bucket id to center of bucket
                    logger.debug("reconstruct_n time : {}", time.time() - st)
                    st = time.time()
                    _, data_bucket_id = kmeans.quantizer.search(data,1)
                    logger.debug("search time : {}", time.time() - st)
                    data_bucket_id = torch.from_numpy(data_bucket_id).squeeze(1) # data index to bucket id
            
                self.tree_cluster(centroids_data, data_bucket_id)
            elif init_method == "rand":
                self.bucket_order = torch.from_numpy(np.random.randint(0,self.bucket_num, data.shape[0], dtype=np.int64))
            else:
                raise NotImplementedError
 
    def tree_cluster(self, centroids_data, data_bucket_id):
        index = torch.arange(self.bucket_num)
        queue = deque()
        queue.append((0, index))
        
        id_code_list = []
        max_code = 0
        while len(queue)>0:
            pcode,index=queue.pop()
            max_code = max(pcode, max_code)
            if self.record_center:
                self.code2ct[pcode] = centroids_data[index].mean(dim = 0)
            if len(index)<=self.k:
                id_code_list.append((pcode,index))
            else:
                cluster_size=len(index)//self.k
                choices, _ = kmeans_equal(centroids_data[index], cluster_size=cluster_size,num_clusters=self.k,max_iters=self.max_iters,metric=self.metric)
                for c in range(self.k):
                    queue.append((self.k * pcode + c+1, index[choices==c]))
        # print("!!", max_code)
        bucket_to_code = torch.empty((self.bucket_num,),dtype=torch.int64)
        for code, ids in id_code_list:
            new_codes = self.k * code + 1 +  torch.arange(self.k)
            bucket_to_code[ids] =  new_codes # k -> len(ids) is better
            if self.record_center:
                self.code2ct[new_codes] = centroids_data[ids]
        

        bucket_id_bucket_order = bucket_to_code - bucket_to_code.min()
        
        bucket_order = bucket_id_bucket_order[data_bucket_id] # len is embedding size, data index to bucket order
        # print(bucket_order.shape)
        self.update_index(bucket_order)
        # self.bucket_order = bucket_order
    
    def update_index_reconstruct(self, data, bucket_order):
        if self.tree_height == 1:
            self.update_index(bucket_order)
        else:
            self.reconstruct_tree(data, bucket_order)
        
    def reconstruct_tree(self, data, bucket_order):
        logger.debug("calculate_cluster_centers")
        st = time.time()
        centroids_data = calculate_cluster_centers(data, bucket_order, self.bucket_num)
        logger.debug("calculate_cluster_centers cost {}s", time.time() - st)
        st = time.time()
        self.tree_cluster(torch.from_numpy(centroids_data), bucket_order)
        logger.debug("tree_cluster cost {}s", time.time() - st)
        
    def update_index(self, bucket_order):
        self.bucket_order = bucket_order
        # not use now. we remain it
        # if counts is None:
        #     self.counts = torch.bincount(bucket_order, minlength = self.bucket_num)
        # else:
        #     self.counts = counts
        # self.counts = torch.cumsum(self.counts,dim=-1)
        # self.class_order = np.argsort(self.bucket_order)
        
    def save(self, fname):
        with open(fname, "wb") as f:
            self.bucket_order.numpy().tofile(f)