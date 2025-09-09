from math import ceil
import os
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tree_model import Tree
import numpy as np
import time
from tqdm import tqdm
import gc
import faiss
# from batl_filter import filter_and_rank
from multiprocessing import Process
from config import Config
from model import DeepModel, DeepModelSepEnc
from faiss import IndexIVFStats
from loguru import logger
import faiss.contrib.torch_utils
from collections import deque
import scipy.sparse as sp

class TreeANNs:
    def __init__(self,
                 data: np.ndarray,
                 conf: Config,
                 tree_list = None,
                 model_list = None,
                 init = True,
                 metric = "l2"
                 ):
        self.first = True
        self.conf = conf
        self.metric = metric
        self.data = data
        if tree_list is not None:
            self.tree_list = tree_list
        else:
            self.tree_list = [Tree(data, conf.k, conf.num_layers, conf.max_iter, init = init, init_method = conf.init, metric = metric, spherical=conf.spherical) for _ in range(conf.R)]
        
        if model_list is not None:
            self.model_list = model_list
        else:
            self.init_model(conf, data)
        self.model_list = [model.to(conf.device) for model in self.model_list]
        
        self.loss = nn.CrossEntropyLoss()
    
    def init_model(self, conf, data):
        net_type = conf.net_type
        if net_type == "deepnet":
            self.model_list = [DeepModel(data.shape[-1], conf.mlp_wth, self.tree_list[i].num_code, self.tree_list[i].bucket_num, conf.code_emb_dim, conf.activate, conf.used_label_num, conf.drop) for i in range(conf.R)]
        if net_type == "deepnet_sep":
            self.model_list = [DeepModelSepEnc(data.shape[-1], conf.mlp_wth, self.tree_list[i].num_code, self.tree_list[i].bucket_num, conf.code_emb_dim, conf.activate, conf.used_label_num, conf.drop) for i in range(conf.R)]
    
    def get_loss(self, r, i, batch_x, path, distri = 0, n_samples = 16384//50):
        if self.conf.k ** i > self.conf.sample_thres:
            return self.get_loss_sampled(r, i, batch_x, path, distri, n_samples)
        else:
            return self.get_loss_full(r, i, batch_x, path, distri)
    
    def get_loss_full(self, r, i, batch_x, path, distri = 0):
        tree = self.tree_list[r]
        conf = self.conf
        layer_st, layer_ed = tree.layer_st_ed[i-1]
        # batch_item_idx = torch.arange(layer_st, layer_ed, device=conf.device)
        # class_cnt = tree.card[tree.tree_height-i].item()*conf.k
        # path = path.to(conf.device)
        # # print(path)
        # if i == 1:
        #     batch_item_p = path[:,0]
        # else:
        #     batch_item_p = ((path[:,:i]+1)*tree.card[tree.tree_height-i:].to(conf.device)).sum(dim = 1) - layer_st
        batch_item_p = path[:,i-1]
        # if conf.net_type == "multienc":
        #     prob = self.model_list[r].classify_rg(batch_x.to(conf.device), layer_st, layer_ed , i-1)
        prob = self.model_list[r].classify_rg(batch_x.to(conf.device), layer_st, layer_ed, distri)

        # print(prob.shape, batch_item_p.shape)
        loss_now = self.loss(prob, batch_item_p.detach().to(conf.device)) *conf.layer_weight[i-1] # self.loss = nn.CrossEntropyLoss()
        
        return loss_now
    
    def get_loss_sampled(self, r, i, batch_x, path, distri = 0, n_samples = 16384//50):
        tree = self.tree_list[r]
        conf = self.conf
        layer_st, layer_ed = tree.layer_st_ed[i-1]
        batch_size = batch_x.shape[0]
        batch_item_p = path[:,i-1].to(self.conf.device)
        predicted_embeddings = self.model_list[r].encode(batch_x.to(conf.device), distri)
        label_emb = self.model_list[r].emb.weight[batch_item_p + layer_st, :]
        label_scores = (predicted_embeddings * label_emb).sum(-1)
        n_classes = self.model_list[r].num_classes
        samples = torch.randint(high=n_classes, size=(n_samples,), device=self.conf.device)
        noise_scores = predicted_embeddings @ self.model_list[r].emb.weight[samples + layer_st, :].T
        noise_scores += np.log(n_classes - 1)
        reject_samples = batch_item_p.unsqueeze(1) == samples.unsqueeze(0)
        noise_scores -= 1e6 * reject_samples
        noise_scores -= torch.log((n_samples - reject_samples.sum(-1, keepdims=True)).float())
        scores = torch.cat([label_scores.unsqueeze(1), noise_scores], dim=1)
        loss = torch.nn.functional.cross_entropy(scores, torch.zeros(batch_size, dtype=torch.long, device=self.conf.device))
        return loss
    
    def _save_ith(self, path, i):
        self.tree_list[i].save(f"{path}/{i}.bin")

    def save_jit(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        processes = []
        for i in range(self.conf.R):
            p = Process(target=self._save_ith, args=(path, i))
            processes.append(p)
            p.start()
        for i in range(self.conf.R):
            script_module = torch.jit.script(self.model_list[i])
            script_module.save(f"{path}/{i}.jit.pt")
        for p in processes:
            p.join()

    def save_for_retrain(self, path):
        self.save(path)
        for i in range(self.conf.R):
            np.save(f"{path}/{i}", self.tree_list[i].bucket_to_path.cpu().numpy())

    def load_for_retrain(self, path):
        for i in range(self.conf.R):
            self.tree_list[i].bucket_to_path = torch.from_numpy(np.load(f"{path}/{i}.npy")).to(self.conf.device)
        self.load(path)

    def save_trees(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        processes = []
        for i in range(self.conf.R):
            p = Process(target=self._save_ith, args=(path, i))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
            
    def save_models(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        for i in range(self.conf.R):
            torch.save(self.model_list[i], f"{path}/{i}.pth")

    def save(self, path):
        self.save_trees(path)
        self.save_models(path)
    
    def save_ckpt(self, path, epoch, optim):
        self.save(path)
        state = {"epoch":epoch,
                 "optimizer": optim.state_dict()}
        torch.save(state, f"{path}/ckpt.pth")
    
    def load_trees(self, path):
        for i in range(self.conf.R):
            bucket_order = np.fromfile(f"{path}/{i}.bin",dtype=np.int64)
            bucket_order = torch.from_numpy(bucket_order)
            # print("loaded",bucket_order.shape)
            if self.conf.reconstruct:
                logger.debug("reconstruct_tree {}", i)
                st = time.time()
                self.tree_list[i].reconstruct_tree(self.data, bucket_order)
                logger.debug("reconstruct_tree {} cost : {} s", i, time.time() - st)
            else:
                self.tree_list[i].update_index(bucket_order)
            
    def load_models(self, path):
        for i in range(self.conf.R):
            self.model_list[i] = torch.load(f"{path}/{i}.pth", map_location=self.conf.device)
    
    def load(self, path):
        self.load_trees(path)
        self.load_models(path)
    
    def load_ckpt(self, path):
        self.load(path)
        ckpt = torch.load(f"{path}/ckpt.pth", map_location=self.conf.device)
        return ckpt["optimizer"], ckpt["epoch"]
    
    def predict(self, test_queries, topm, topk, retrieve_batch_size, num_beams=100, mode = "", distri = 0):
        stats = IndexIVFStats()
        for model in self.model_list:
            model.eval()
        num_batch=math.ceil(1.0*test_queries.shape[0]/retrieve_batch_size)
        retrieve_time = time.time()
        top_buckets_list = []
        test_queries_to_model = test_queries.to(self.conf.device)
        for btN in range(num_batch):
            top_buckets=[None for _ in range(self.conf.R)]
            batch_x = test_queries_to_model[btN*retrieve_batch_size:(btN+1)*retrieve_batch_size]
            with torch.no_grad():
                for r in range(len(self.tree_list)):
                    res=self.generateBucket_fast(batch_x,r,num_beams=num_beams,topk=topm,mode = mode, distri = distri).cpu()
                    top_buckets[r] = res
            top_buckets = torch.stack(top_buckets, dim=1) #shape is [R,batch_size,topk]->[batch_sze,R,topk]
            top_buckets_list.append(top_buckets)
        retrieve_time = time.time() - retrieve_time
        logger.debug(f"retrieve: {retrieve_time} s")
        # for R = 1, exact search using faiss
        cpu_st = time.time()
        top_buckets = torch.cat(top_buckets_list, dim=0).to(self.conf.ivf_device)
        stats.reset()
        # x_v = faiss.Float32Vector()
        # faiss.copy_array_to_vector(test_queries.numpy().reshape(-1),x_v)
        self.faiss_index.nprobe = topm
        # assign_v = faiss.Int64Vector()
        # # faiss.copy_array_to_vector(top_buckets.reshape(-1),assign_v)
        # D = np.empty((test_queries.shape[0], topk), dtype=np.float32)
        # D_v = faiss.Float32Vector()
        C = torch.zeros((test_queries.shape[0], topm), dtype=torch.float32)
        # C_v = faiss.Float32Vector()
        # faiss.copy_array_to_vector(C.reshape(-1),C_v)
        # faiss.copy_array_to_vector(D.reshape(-1),D_v)
        # I = np.empty((test_queries.shape[0], topk), dtype=np.int64)
        # I_v = faiss.Int64Vector()
        # faiss.copy_array_to_vector(I.reshape(-1),I_v)
        if self.conf.ivf_device == self.conf.device:
            test_queries = test_queries_to_model
        else:
            test_queries = test_queries.to(self.conf.ivf_device)
        st = time.time()
        D, I = self.faiss_index.search_preassigned(test_queries ,topk, top_buckets.reshape((test_queries.shape[0], topm)), C, stats = stats if self.conf.ivf_device == "cpu" else None) #  
        logger.debug("search time {} s".format(time.time() - st)) #
        logger.debug("ndis: {}".format(stats.ndis/test_queries.shape[0])) # "search_time: ", stats.search_time/test_queries.shape[0]
        return I.cpu().numpy().reshape(-1,topk), retrieve_time, time.time() - cpu_st, stats.ndis/test_queries.shape[0]
    
    def generateBucket_fast(self, batch_query: torch.Tensor, r, num_beams=100, topk=1, mode = "", distri = 0):
        #batch_query [bs,1,dim]
        batch_size, vdim=batch_query.shape
        # print(batch_query.shape)
        all_emb = self.model_list[r].encode(batch_query.to(self.conf.device), distri)
        # print(all_emb.shape)
        
        # print(extend_all_emb.shape)
        # batch_query=batch_query.to(self.conf.device).\
        #     repeat(1,num_beams,1).view(batch_size*num_beams,1,emb_dim)
        
        for l, sz in enumerate(self.tree_list[r].card):
            if num_beams >= sz or sz < 256:
                break
        # print("l:", l)
        l = 0
        st_code = self.tree_list[r].card[l:].sum()
        ed_code = st_code + self.tree_list[r].card[l]*self.conf.k
        # print(st_code, ed_code)
        logit = all_emb @ self.model_list[r].emb.weight[st_code:ed_code].T
            # now_code = torch.arange(st_code, st_code+self.tree_list[r].card[fl]).unsqueeze(0).repeat((batch_size, 1)).flatten()
        # print(logit.shape)
        _,indices=logit.topk(num_beams if l !=0 else topk,largest=True,dim=1)
        now_code = indices.flatten() + st_code
        # print(now_code)
        if l > 0:
            index_base=torch.arange(batch_size,device=self.conf.device).view(-1,1)*num_beams
            extend_all_emb = all_emb.unsqueeze(1).repeat(1,num_beams,1).view(batch_size*num_beams,1,self.conf.code_emb_dim)
        for i in range(self.conf.num_layers-l,self.conf.num_layers):
            # print(i, "!!!")
            # temp_result = self.model_list[r].generate(batch_query,cur_path)
            # temp_result = torch.full((batch_size*num_beams,1,self.conf.k),-1e9, dtype=torch.float32, device=self.conf.device)
            
            # if i ==2:
    
                # now_code = cur_path # ((cur_path[:,1:]+1)*self.tree_list[r].card[self.tree_list[r].tree_height-i+1:].to(self.conf.device)).sum(dim = 1)
            # print(now_code.shape, now_code) 
            new_code = ((now_code.flatten()*self.conf.k).unsqueeze(1).repeat((1, self.conf.k)) + torch.arange(1, self.conf.k+1, device=self.conf.device))
            # print(new_code.shape)
            emb_now = self.model_list[r].emb(new_code).permute(0, 2, 1)
            # print(emb_now.shape)
            logit = torch.bmm(extend_all_emb, emb_now).squeeze(1)
            # print(logit)
            temp_result = logit.view((batch_size*num_beams,self.conf.k))#
            # temp_result = torch.exp(logit).view((batch_size,num_beams*self.conf.k))
            # result_sum = temp_result.sum(dim = 1)
            # print(temp_result.shape, temp_result.sum(dim = 1).unsqueeze(1).shape)
            # temp_result= torch.log(torch.div(temp_result, temp_result.sum(dim = 1).unsqueeze(1))).view((batch_size*num_beams,self.conf.k)).unsqueeze(1)
            # print(score.shape) 
            # print(temp_result.shape, pred_scores.shape)   
            # print(temp_result[:,-1,:][0], pred_scores[0])
            cur_log_prob = (temp_result).view(batch_size,num_beams*self.conf.k)
            # cur_log_prob = (temp_result[:,-1,:] + pred_scores.view(-1,1)).view(batch_size,num_beams*self.conf.k)
            # print("cur_log_prob", cur_log_prob.shape)
            
            # print(cur_log_prob[0][:100])
            if i < self.conf.num_layers-1:
                _,indices=cur_log_prob.topk(num_beams,largest=True,dim=1)
                # print(indices[0],"bf")
                # pred_scores[pred_scores>-1e9] = 0
                # print(pred_scores[0],"aft")
            else:
                _,indices=cur_log_prob.topk(topk,largest=True,dim=1)
                # print(indices[1],"bf", _[1])
            beam_ids=torch.div(indices,self.conf.k,rounding_mode='trunc')+index_base
            # print(beam_ids.shape, index_base.shape, beam_ids[0])
            # print(now_code[beam_ids.view(-1)]*self.conf.k, beam_ids.shape)
            now_code=(now_code[beam_ids.view(-1)]*self.conf.k+(indices.flatten()%self.conf.k))+1
            # print(now_code)
            # print("path:", cur_path[:64])
        return (now_code - self.tree_list[r].card.sum()).reshape(batch_size, topk)#[batch_size*topk]
    
    def build_faiss_index(self, r = 0):
        if hasattr(self, "faiss_index"):
            del self.faiss_index
            gc.collect()
        logger.debug("bucket max:{} min:{}", self.tree_list[r].bucket_order.max(), self.tree_list[r].bucket_order.min())
        dim = self.data.shape[-1]
        if self.metric == "ip":
            measure = faiss.METRIC_INNER_PRODUCT
        if self.metric == "linf":
            measure = faiss.METRIC_Linf
        else:
            measure = faiss.METRIC_L2
        if self.conf.ivf_device.startswith("cuda"):
            from faiss.loader import swig_ptr
            res = faiss.StandardGpuResources()
            # res.noTempMemory()
            config = faiss.GpuIndexIVFFlatConfig()
            config.device = int(self.conf.ivf_device.split(":")[1])
            nlist = self.conf.k ** self.conf.num_layers
            self.faiss_index = faiss.GpuIndexIVFFlat(res, dim, nlist, measure, config)
            precomputed_idx = self.tree_list[r].bucket_order.numpy()
            self.faiss_index.add_preassigned(self.data.shape[0], swig_ptr(self.data), swig_ptr(precomputed_idx))
            self.faiss_index.is_trained = True
            return
        nlist = self.conf.k ** self.conf.num_layers + 1
        quantizier = faiss.IndexFlat(dim, measure)
        self.faiss_index = faiss.IndexIVFFlat(quantizier, dim, nlist, measure)
        self.faiss_index.is_trained = True
        self.faiss_index.parallel_mode = 2
        logger.debug("metric type {} {}", quantizier.metric_type, self.faiss_index.metric_type)
        precomputed_idx = self.tree_list[r].bucket_order.numpy()
        x_v = faiss.Float32Vector()
        faiss.copy_array_to_vector(self.data.reshape(-1),x_v)
        item_num = self.data.shape[0]
        # del self.data
        gc.collect()
        xids = np.arange(item_num)
        xids_v = faiss.Int64Vector()
        faiss.copy_array_to_vector(xids.reshape(-1),xids_v)
        pre_v = faiss.Int64Vector()
        faiss.copy_array_to_vector(precomputed_idx.reshape(-1),pre_v)
        self.faiss_index.add_core(item_num, x_v.data(),xids_v.data(),pre_v.data())
      
    def fill_empty(self, new_bucket_order, r = 0):
        emp_pos = torch.where(new_bucket_order == -1)[0]
        logger.debug(f"empty size {emp_pos.shape[0]}")
        new_bucket_order[emp_pos] = self.tree_list[r].bucket_order[emp_pos]
        return new_bucket_order
    
    def allocate_dense(self, cm, bal_factor, num_edge):
        # logger.debug("Dense")
        if bal_factor > 0:
            val, pos = cm.topk(num_edge)
            val_srt, idx_sort = val.view(-1).sort(descending = True)
            del val_srt
            new_bucket_order = torch.full((cm.shape[0],),-1,dtype=torch.int64)
            counts = torch.zeros((cm.shape[1],),dtype=torch.int64)
            lim = cm.shape[0]/cm.shape[1] * bal_factor
            for idx in tqdm(idx_sort):
                idx = idx.item()
                # print(idx)
                # break
                qid = idx//num_edge
                pid = idx % num_edge
                bucket = pos[qid, pid].item()
                
                # print(bucket, qid)
                if new_bucket_order[qid] ==-1 and counts[bucket] < lim:
                    counts[bucket]+=1
                    new_bucket_order[qid] = bucket
        else:
            new_bucket_order = cm.argmax(dim=1)
            # row_max = cm.max(dim=1)
            # emp_pos = torch.where(row_max.values == 0)[0]
            # new_bucket_order[emp_pos] = self.tree_list[r].bucket_order[emp_pos]
        return new_bucket_order
    

    def allocate_dense_ez(self, cm, bal_factor, num_edge, bs = None):
        new_bucket_order = []
        # new_bucket_order = torch.full((cm.shape[0],),-1,dtype=torch.int64)
        cur_bucket_counts = torch.zeros(self.model_list[r].num_classes,dtype=torch.int32, device="cpu")
        lim = cm.shape[0]//cm.shape[1]*bal_factor
        if bs is None:
            bs = self.conf.bs
        bnum = (cm.shape[0]-1)//bs+1
        for i in tqdm(range(bnum)):
            res = cm[i*bs:(i+1)*bs].argsort(descending=True)
            index = (cur_bucket_counts[res] >= lim).int().argmin(dim=-1).view(-1,1)
            res = res.gather(dim=-1,index=index).view(-1)
            bucket_id, bucket_id_counts = torch.unique(res, sorted=False, return_counts=True)
            cur_bucket_counts[bucket_id] += bucket_id_counts
            new_bucket_order.append(res.cpu())
        bucket_order = torch.concat(new_bucket_order, dim= 0)
        return bucket_order
    
    def update_final(self, train_queries, train_gts, self_train_queries = None, self_train_gts = None, topt = 15, topt_self = 1, upd_bs = 20000, bal_factor = -1, num_edge = 5, r = 0):
        
        del self.faiss_index
        num_classes =self.model_list[r].num_classes
        spm = sp.csc_matrix((np.ones((train_gts.shape[0]*topt,)), train_gts.ravel(), np.arange(train_gts.shape[0]+1)*topt), 
                                    shape=(self.data.shape[0], train_gts.shape[0]))
        spm = spm.tocsr()
        st, ed = 0, 1
        lsted = self.tree_list[r].layer_st_ed
        assign_upper = None
        assign_values = []
        assign_idx = []
        
        with torch.no_grad():
            pbar = tqdm(total=spm.indptr.shape[0])
            while True:
                qid_st = spm.indptr[st]
                while ed + 1 < spm.indptr.shape[0] and spm.indptr[ed] - qid_st < upd_bs:
                    ed += 1
                
                assi = torch.zeros((ed - st, num_classes))
                assi_upper = torch.zeros((ed - st, self.conf.k))
                count = torch.zeros((ed - st,), dtype=torch.int64)
                # assi_vec = torch.empty((ed - st, num_classes))
                if self.conf.upd_on_query:
                    queries = train_queries[spm.indices[qid_st:spm.indptr[ed]]].to(self.conf.device)
                    probe = self.model_list[r].classify_all(queries)
                    # print(probe.shape)
                    # print(spm.indptr[st], spm.indptr[ed], st, ed)
                    l1_probe = probe[:, lsted[0][0] : lsted[0][1]].softmax(dim = -1)
                    
                    l2_probe = probe[:, lsted[1][0] : lsted[1][1]].softmax(dim = -1)
                    
                    l2_probe = l2_probe.cpu()
                    # emp_doc = []
                    # emp_idx = []
                    for i in range(st, ed):
                        sti = spm.indptr[i] - qid_st
                        edi = spm.indptr[i+1] - qid_st
                        # if tmp_cnt >= 1:
                        #     print(sti, edi)
                        if sti < edi:
                            assi[i-st] = l2_probe[sti:edi].cpu().sum(dim = 0)
                            assi_upper[i-st] = l1_probe[sti:edi].cpu().sum(dim = 0)
                            count[i-st] += edi - sti
                        else:
                            # empty doc proc
                            # emp_idx.append(i-st)
                            # emp_doc.append(i)
                            pass
                # print(emp_doc)
                """
                if len(emp_doc)>0:
                    emp_doc = np.array(emp_doc)
                    # print(emp_doc)
                    probes_self = self.model_list[r].classify_rg(torch.from_numpy(self.data[emp_doc]).to(self.conf.device), lsted[1][0], lsted[1][1]).softmax(dim = -1).cpu()
                    assi.scatter_(dim=0, index=torch.from_numpy(emp_doc - st).unsqueeze(1), src=probes_self)
                """
                # emp_doc = np.arange(st, ed)
                # idx = torch.from_numpy(emp_doc - st)
                probes_self = self.model_list[r].classify_all(torch.from_numpy(self.data[st:ed]).to(self.conf.device), 1)
                probes_self_l2 = probes_self[:, lsted[1][0] : lsted[1][1]].softmax(dim = -1).cpu()
                assi += probes_self_l2
                probes_self_l1 = probes_self[:, lsted[0][0] : lsted[0][1]].softmax(dim = -1).cpu()
                assi_upper += probes_self_l1
                count += 1
                # count[count==0] =1
                assi = assi/count.reshape(-1,1)
                
                assi_upper = assi_upper/count.reshape(-1,1)
                
                if assign_upper is None:
                    assign_upper = assi.T @ assi_upper
                else:
                    assign_upper += assi.T @ assi_upper
                
                top_values, top_indices = torch.topk(assi, k=num_edge, dim=1, largest=True, sorted=True)
                assign_values.append(top_values.cpu())
                assign_idx.append(top_indices.cpu())
                pbar.update(ed - st)
                if ed + 1 >= spm.indptr.shape[0]:
                    break
                
                # tmp_cnt += 1
                # if tmp_cnt > 1:
                #     break
                
                st = ed
                ed += 1
                # break
            pbar.close()
        if self.conf.upd_assign_mcmf:
            upper_bo = self.allocate_dense(assign_upper, 1.0, self.conf.k)
        else:
            logger.debug("NO MCMF")
            upper_bo = self.allocate_dense_ez(assign_upper, 1, self.conf.k, bs = 1)
        remap = torch.zeros_like(upper_bo)
        for i in range(self.conf.k):
            logger.debug(torch.where(upper_bo == i)[0].shape)
            remap[upper_bo == i] = torch.arange(self.conf.k) + i*self.conf.k
        
        
        assign_values = torch.concat(assign_values)
        assign_idx = torch.concat(assign_idx)
        if self.conf.upd_assign_mcmf:
            val_srt, idx_sort = assign_values.view(-1).sort(descending = True)
            num_base = self.data.shape[0]
            num_classes =self.model_list[r].num_classes
            new_bucket_order = torch.full((num_base,),-1,dtype=torch.int64)
            counts = torch.zeros((num_classes,),dtype=torch.int64)
            lim = num_base/num_classes * bal_factor
            for idx in tqdm(idx_sort):
                idx = idx.item()
                # print(idx)
                # break
                qid = idx//num_edge
                if new_bucket_order[qid] !=-1:
                    continue
                pid = idx % num_edge
                bucket = assign_idx[qid, pid].item()
                
                # print(bucket, qid)
                if  counts[bucket] < lim:
                    counts[bucket]+=1
                    new_bucket_order[qid] = bucket
        else:
            logger.debug("NO MCMF")
            new_bucket_order = []
            conf = self.conf
            bs = 100 # conf.bs
            bnum = (assign_values.shape[0]-1)//bs+1
            cur_bucket_counts = torch.zeros(self.model_list[r].num_classes,dtype=torch.int32, device="cpu")
            lim = assign_values.shape[0]/self.model_list[r].num_classes*bal_factor
            for i in tqdm(range(bnum)):
                res = assign_idx[i*bs:(i+1)*bs]
                index = (cur_bucket_counts[res] >= lim).int().argmin(dim=-1).view(-1,1)
                res = res.gather(dim=-1,index=index).view(-1)
                bucket_id, bucket_id_counts = torch.unique(res, sorted=False, return_counts=True)
                cur_bucket_counts[bucket_id] += bucket_id_counts
                new_bucket_order.append(res.cpu())
            new_bucket_order = torch.concat(new_bucket_order, dim= 0)
            
        new_bucket_order = self.fill_empty(new_bucket_order)
        new_bucket_order = remap[new_bucket_order]
        self.tree_list[r].update_index(new_bucket_order)
        self.build_faiss_index()
        