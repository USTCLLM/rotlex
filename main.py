import torch
import os
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import json
import time
from torch.utils.tensorboard import SummaryWriter
from utils import set_seed
import argparse
from config import Config
from dataset import *
from dataloader import intrain_dataset_factory
from tree_anns import TreeANNs
import math
from tqdm import tqdm
import numpy as np
from metric import precision, recall
from evaluate import val_test
from loguru import logger
import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default="", help='config json file')
    parser.add_argument('-e', '--extra', type=str, default="", help='extra suffix of log dir')
    # parser.add_argument("-l", "--savelast", action="store_true")
    parser.add_argument("-u", "--upd_first", action="store_true")
    parser.add_argument("-s","--seed", type=int, default=1)
    parser.add_argument("-d","--device", type=str, default="")
    parser.add_argument("-i","--ivfdevice", type=str, default="")
    parser.add_argument("-v","--eval", type=str, default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=150)
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument("-x","--extra_points", type=int, default=[], nargs="+")
    args = parser.parse_args()
    
    for k, v in args.__dict__.items():
        print(f"self.{k} = '{v}'")
    
    if args.eval != "" and args.file == "":
        args.file = os.path.join(os.path.dirname(args.eval), "config.json")
    
    if args.file != "":
        with open(args.file, "r") as f:
            conf_dict = json.load(f)
        conf = Config(**conf_dict)
    else:
        conf = Config()
    if args.device != "":
        conf.device = args.device
    if args.ivfdevice != "":
        conf.ivf_device = args.ivfdevice
    if args.extra != "":
        conf.extra = args.extra
    logger.debug(conf.getname())
    set_seed(args.seed)
    
    r = 0
    if args.eval != "":
        conf.train_mode = "only_test"
    logger.debug(conf.train_mode)
    ds = dataset_factory(conf.dataset_name, conf.train_set_len, conf.self_train_set_len, conf.train_mode)
    
    item_embedding = torch.from_numpy(ds.data)
    init_flag = (conf.load_tree == "" and args.eval == "" and conf.load_all == "")
    init_st = time.time()
    trainer = TreeANNs(ds.data, conf, tree_list= None, model_list = None, init= init_flag, metric=ds.metric)
    
    if args.eval != "":
        print("loading...")
        trainer.load(args.eval)
        trainer.build_faiss_index()
        test_queries = torch.from_numpy(ds.test_queries)
        # test_queries = torch.from_numpy(ds.self_test_queries) # TMP MOD
        if conf.norm_query:
            test_queries = F.normalize(test_queries, dim=1)
            
        if isinstance(ds, AnnDatasetSelfTrain):
            id_test_queries = torch.from_numpy(ds.self_test_queries)
            if conf.norm_query:
                id_test_queries = F.normalize(id_test_queries, dim=1)
        else:
            id_test_queries = None
        # print("warm up")
        # val_test(test_queries, item_embedding) # warmup
        print("start evalution")
        res, time_cost, retrieve, ivf_time, ndis, res2, time_cost2, retrieve_cost2, ivf_cost2, ndis2, resm, time_costm, retrieve_costm, ivf_costm, ndism = val_test(trainer, test_queries, ds.test_gts, id_test_queries, ds.self_test_gts if isinstance(ds, AnnDatasetSelfTrain) else None, args.start, args.stop, args.interval, args.extra_points)
        with open("{}/time_recall_{}_{}.json".format(args.eval, datetime.datetime.now().strftime("%y%m%d%H%M%S"), args.extra), "w") as f:
            json.dump({"time":time_cost, "recall": res, "retrieve_time": retrieve, "search_time": ivf_time, "ndis": ndis,
                       "time_id":time_cost2, "recall_id": res2, "retrieve_time_id": retrieve_cost2, "search_time_id": ivf_cost2, "ndis_id": ndis2,
                       "time_tot":time_costm, "recall_tot": resm, "retrieve_time_tot": retrieve_costm, "search_time_tot": ivf_costm, "ndis_tot": ndism}, f)
        print(res, time_cost)
        print(res2, time_cost2)
        print(resm, time_costm)
        exit()
        
    # reload and eval ends here
    my_log_dir =  "logs/"+ conf.dataset_name + "/" + conf.getname()
    writer = SummaryWriter(log_dir = my_log_dir)
    with open(my_log_dir+"/config.json", "w") as f:
        json.dump(conf.__dict__, f)
    logger.add(my_log_dir+"/{time}.log")
    logger.debug(f"SAVE TO: {my_log_dir}")
    if init_flag:
        try:
            logger.debug("Init time : %f s" % (time.time()- init_st))
            trainer.save_trees(f"{my_log_dir}/inited_tree")
        except Exception as e:
            print(e)
    
    epoch = 0
    if conf.load_all != "":
        if os.path.exists(os.path.join(conf.load_all, "ckpt.pth")):
            optimizer_dict, load_epoch = trainer.load_ckpt(conf.load_all)
            print(load_epoch, "load_epoch")
            epoch = load_epoch
        else:
            trainer.load(conf.load_all)
            
    if conf.load_tree != "":
        trainer.load_trees(conf.load_tree)
    trainer.build_faiss_index()
    
    train_query = (conf.train_mode == "all" or conf.train_mode == "query")
    train_self = (conf.train_mode == "all" or conf.train_mode == "self")
    
    intrain_ds = intrain_dataset_factory(ds, -1, 10000, conf.norm_query, conf.eval_topk) #  pin_dev=conf.device
    print(train_query, train_self)
    
    def get_train_loader():
        if train_query:
            train_loader = intrain_ds.train_dataloader(conf.bs, trainer.tree_list[r].bucket_order, trainer.tree_list[r].bucket_to_path, conf.used_label_num)
        
        if train_self:
            self_train_loader = intrain_ds.self_train_dataloader(conf.bs, trainer.tree_list[r].bucket_order, trainer.tree_list[r].bucket_to_path, conf.used_label_num)
        
        if train_query and train_self:
            main_train_loader = train_loader
            second_train_loader = self_train_loader
        elif train_query:
            main_train_loader = train_loader
            second_train_loader = None
        elif train_self:
            main_train_loader = self_train_loader
            second_train_loader = None
        else:
            raise ValueError("No train data")
        return main_train_loader, second_train_loader
    
    main_train_loader, second_train_loader = get_train_loader()
    writer.add_text("conf", str(json.dumps(conf.__dict__)), 0)
    
    
    optimizer = torch.optim.Adam(trainer.model_list[r].parameters(), lr=conf.lr, amsgrad=True)
    if epoch > 0:
        optimizer.load_state_dict(optimizer_dict)
    
    min_val_loss = 1e9
    valid_target_idx = epoch
    update_idx = epoch
    max_val_recall = 0.0
    layer_weight = conf.layer_weight
    upd_flag = args.upd_first
    tree = trainer.tree_list[r]
    while epoch < conf.epoch:
        if epoch - update_idx > conf.upd_patient:
            update_idx = epoch
            upd_flag = True
            
        if (epoch > 0 and epoch % conf.upd_interval == 0) or upd_flag: # max_val_recall < 0.98 and 
            trainer.save_ckpt(f"{my_log_dir}/ckpt_{epoch}", epoch, optimizer)
            logger.debug(f"updating...(upd_flag :{upd_flag}, balance_factor: {conf.balance_factor})")
            upd_flag = False
            if True: # conf.upd_method =="rotlex":
                upd_bs = 20000
                topt = 15
                train_queries = intrain_ds.train_queries
                train_gts = intrain_ds.train_gts[:, :topt]
                # self_train_queries = torch.from_numpy(intrain_ds.ds.data[:intrain_ds.ds.self_train_set_len])
                # self_train_gts = torch.concat((intrain_ds.self_train_gts, intrain_ds.self_val_gts))
                trainer.update_final(train_queries, train_gts, None, None, topt, 1, upd_bs, conf.balance_factor, 5)


            trainer.save_ckpt(f"{my_log_dir}/just_after_upd_ep{epoch}", epoch, optimizer)
            main_train_loader, second_train_loader = get_train_loader()
            if conf.reinit_after_upd:
                logger.debug("do reinit emb")
                torch.nn.init.kaiming_uniform_(trainer.model_list[0].emb.weight, a=math.sqrt(5))
        
        for m in trainer.model_list:
            m.train()
        
        train_st = time.time()
        
        sum_train_loss = 0
        count_train_loss = 0
        layer_loss = [0 for _ in range(trainer.tree_list[r].tree_height)]
        self_layer_loss = [0 for _ in range(trainer.tree_list[r].tree_height)]
        
        if second_train_loader:
            it = iter(second_train_loader)            
            for (batch_x, path) in tqdm(main_train_loader):
                try:
                    (batch_x2, path2) = next(it)
                except StopIteration:
                    # self_train_loader = intrain_ds.self_train_dataloader(conf.bs, trainer.tree_list[r].bucket_order, trainer.tree_list[r].bucket_to_path, norm = conf.norm_query)
                    it = iter(second_train_loader)
                    (batch_x2, path2) = next(it)
                
                sum_loss = 0
                for i in range(1,trainer.tree_list[r].tree_height+1):#
                    
                    loss_now = trainer.get_loss(r, i, batch_x, path, 0)
                    layer_loss[tree.tree_height-i] += loss_now.detach().cpu().item()
                    sum_loss += loss_now
                    
                    loss_now2 = trainer.get_loss(r, i, batch_x2, path2, 1)
                    self_layer_loss[tree.tree_height-i] += loss_now2.detach().cpu().item()
                    sum_loss += loss_now2
                
                sum_train_loss += (sum_loss.item())
                sum_loss.backward()
                # print(sum_loss.item())
                count_train_loss += 1
                optimizer.step()# update the parameters
                optimizer.zero_grad()# clean the gradient
        else:
            for (batch_x, path) in tqdm(main_train_loader):
                
                sum_loss = 0
                for i in range(1,trainer.tree_list[r].tree_height+1):#
                    
                    loss_now = trainer.get_loss(r, i, batch_x, path, 0)
                    layer_loss[tree.tree_height-i] += loss_now.detach().cpu().item()
                    sum_loss += loss_now
        
                sum_train_loss += (sum_loss.item())
                sum_loss.backward()
                # print(sum_loss.item())
                count_train_loss += 1
                optimizer.step()# update the parameters
                optimizer.zero_grad()# clean the gradient
        
        for i in range(trainer.tree_list[r].tree_height):
            writer.add_scalar(f"loss/train_loss_l_{i}", layer_loss[i]/count_train_loss, epoch)
        for i in range(trainer.tree_list[r].tree_height):
            writer.add_scalar(f"loss/self_train_loss_l_{i}", self_layer_loss[i]/count_train_loss, epoch)
        logger.debug(f"epoch {epoch} train loss: {sum_train_loss/count_train_loss}") 
        writer.add_scalar("loss/train_loss", sum_train_loss/count_train_loss, epoch)
       
        logger.debug("train cost time {} s", time.time() - train_st)
        if epoch % conf.val_interval == 0:
            trainer.save_ckpt(f"{my_log_dir}/ep{epoch}", epoch + 1, optimizer)
            val_st = time.time()
            for m in trainer.model_list:
                m.eval()
            samples= conf.eval_num
            eval_beam = conf.eval_beam
            eval_topk = conf.eval_topk        
            if train_query:
                val_queries = intrain_ds.val_queries
                result_history, retrieve_time, ivf_time, ndis_now = trainer.predict(val_queries[:samples],topm=eval_beam,num_beams=eval_beam,topk=eval_topk,retrieve_batch_size=100,distri=0)
                # print(result_history[:10])
                # for i in range(result_history.shape[1]):
                #     print(l2(val_queries[0].numpy(), ds.data[result_history[0][i]]))
                # for j in range(20):
                #     for i in range(10):
                #         print(l2(intrain_ds.val_queries[j].numpy(), ds.data[intrain_ds.val_gts[j][i]]))
                #     print("*"*10)
                logger.debug(f"epoch {epoch}:eval ({samples}/{val_queries.shape[0]}) :")
                res_list= []
                for kk in range(10, intrain_ds.val_gts.shape[1]+1, 10):
                    res = recall(result_history[:,:kk], intrain_ds.val_gts[:,:kk].numpy())
                    res_list.append(res)
                    writer.add_scalar(f"metrics/reall@{kk}", res, epoch)
                    logger.debug(f"recall@{kk} {res}")
                logger.debug(f"(val cost time {time.time()- val_st} s")
                writer.add_scalar("metrics/ndis", ndis_now, epoch)
                writer.add_scalar("metrics/retrieve_time", retrieve_time, epoch)
                writer.add_scalar("metrics/ivf_time", ivf_time, epoch)
                res = res_list[0]
            if train_self:
                val_queries = intrain_ds.self_val_queries
                logger.debug(f"epoch {epoch}:eval self ({samples}/{val_queries.shape[0]}) :")
                result_history, retrieve_time, ivf_time, ndis_now = trainer.predict(intrain_ds.self_val_queries[:samples],topm=eval_beam,num_beams=eval_beam,topk=eval_topk,retrieve_batch_size=100,distri=1)
                res_list_self = []
                for kk in range(10, intrain_ds.self_val_gts.shape[1]+1, 10):
                    res_self = recall(result_history[:,:kk], intrain_ds.self_val_gts[:,:kk].numpy())
                    res_list_self.append(res_self)
                    writer.add_scalar(f"metrics/reall@{kk}_self", res_self, epoch)
                    logger.debug(f"recall@{kk}_self {res_self}")
                logger.debug(f"(val cost time {time.time()- val_st} s")
                if not train_query:
                    res = res_list_self[0]
            
            if res > max_val_recall:
                logger.debug(f"best epoch {epoch}")
                valid_target_idx = epoch
                update_idx = epoch
                max_val_recall = res
                trainer.save_ckpt(f"{my_log_dir}/weight", epoch, optimizer)
                
            if epoch - valid_target_idx > conf.es_patient:
                logger.debug(f"Early stopping at epoch {epoch}, best epoch {valid_target_idx}")
                break
        
        epoch += 1