import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from dataset import AnnDatasetSelfTrain, AnnDataset
import torch.nn.functional as F

class AnnTensorDataset(torch.utils.data.Dataset):
    def __init__(self, queries, paths, used_label_num) -> None:
        super().__init__()
        self.queries = queries
        self.paths = paths
        self.used_label_num = used_label_num
        
    def __len__(self):    
        return self.paths.shape[0]
    
    def __getitem__(self, index):
        return self.queries[index//self.used_label_num], self.paths[index]
    
class InTrainDatasetMem:
    """
    For AnnDataset
    """
    def __init__(self, ds:AnnDataset, used_label_num, val_num = 10000, norm_query = False, eval_topk = 10, pin_dev = "cpu") -> None:
        self.ds = ds
        self.pin_dev = pin_dev
        if used_label_num == -1:
            self.used_label_num = ds.test_gts.shape[-1]
        else:
            self.used_label_num = used_label_num
        self.val_num = val_num
        self.norm_query = norm_query
        self.eval_topk = eval_topk
        if ds.train_queries is not None:
            self.val_gts = torch.from_numpy(ds.train_gts[-val_num:,:self.eval_topk])
            self.val_queries = torch.from_numpy(ds.train_queries[-val_num:])
            self.train_queries = torch.from_numpy(ds.train_queries[:-val_num])
            self.train_gts = torch.from_numpy(ds.train_gts[:-val_num,:self.used_label_num])
            print(f"train_queries: {self.train_queries.shape} | train_gts: {self.train_gts.shape}")
            print(f"val_queries: {self.val_queries.shape} | val_gts: {self.val_queries.shape}")
            if norm_query:
                self.train_queries = F.normalize(self.train_queries, dim=1)
                self.val_queries = F.normalize(self.val_queries, dim=1)
    
    def _get_dataloader(self, queries, gts, batch_size, bucket_order, bucket_to_path, shuffle, num_nns, num_works = 8):
        train_path = bucket_to_path[bucket_order[gts[:,:num_nns].reshape(-1).long()]]
        assert queries.shape[0] * num_nns == train_path.shape[0] 
        tensor_train_instances = AnnTensorDataset(queries, train_path, num_nns)
        if self.pin_dev.startswith("cuda"):
            return DataLoader(dataset=tensor_train_instances, batch_size=batch_size, shuffle=shuffle, num_workers=num_works, persistent_workers = False, pin_memory=True, pin_memory_device=self.pin_dev)
        else:
            return DataLoader(dataset=tensor_train_instances, batch_size=batch_size, shuffle=shuffle, num_workers=num_works, persistent_workers = False)
    
    def train_dataloader(self, batch_size, bucket_order, bucket_to_path, num_nns, num_works = 8):
        return self._get_dataloader(self.train_queries, self.train_gts ,batch_size, bucket_order, bucket_to_path, True, num_nns, num_works)
    
    def val_dataloader(self, batch_size, bucket_order, bucket_to_path, num_nns, num_works = 8):
        return self._get_dataloader(self.val_queries, self.val_gts ,batch_size, bucket_order, bucket_to_path, False, num_nns, num_works)
    
class InTrainDatasetMem2Distr(InTrainDatasetMem):
    """
    For AnnDatasetSelfTrain
    """
    def __init__(self, ds:AnnDatasetSelfTrain, used_label_num, val_num = 10000, norm_query = False, eval_topk = 10,  pin_dev = "cpu") -> None:
        super().__init__(ds, used_label_num, val_num, norm_query, eval_topk, pin_dev)
        
        if ds.self_train_gts is not None:
            self.self_val_gts = torch.from_numpy(ds.self_train_gts[-val_num:,:self.eval_topk])
            self.self_train_gts = torch.from_numpy(ds.self_train_gts[:-val_num,:self.used_label_num])
            self.self_val_queries = torch.from_numpy(ds.data[ds.self_train_set_len-val_num:ds.self_train_set_len])
            print(f"| self_train_gts: {self.self_train_gts.shape}")
            print(f"self_val_queries: {self.self_val_queries.shape} | self_val_gts: {self.self_val_gts.shape}")
            if norm_query:
                self.self_val_queries = F.normalize(self.self_val_queries, dim=1)
        
    def self_train_dataloader(self, batch_size, bucket_order, bucket_to_path, num_nns, num_works = 8):
        self_train_queries = torch.from_numpy(self.ds.data[:self.ds.self_train_set_len-self.val_num])
        if self.norm_query:
            self_train_queries = F.normalize(self_train_queries, dim=1)
        return self._get_dataloader(self_train_queries, self.self_train_gts ,batch_size, bucket_order, bucket_to_path, True, num_nns, num_works)
    
    def self_val_dataloader(self, batch_size, bucket_order, bucket_to_path, num_nns, num_works = 8):
        return self._get_dataloader(self.self_val_queries, self.self_val_gts ,batch_size, bucket_order, bucket_to_path, False, num_nns, num_works)


def intrain_dataset_factory(ds, used_label_num, val_num = 10000, norm_query = False, eval_topk = 10, pin_dev = "cpu"):
    if isinstance(ds, AnnDatasetSelfTrain):
        return InTrainDatasetMem2Distr(ds, used_label_num, val_num, norm_query, eval_topk, pin_dev = pin_dev)
    if isinstance(ds, AnnDataset):
        return InTrainDatasetMem(ds, used_label_num, val_num, norm_query, eval_topk, pin_dev = pin_dev)