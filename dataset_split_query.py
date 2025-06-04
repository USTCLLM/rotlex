import numpy as np
from dataset import *
from tqdm import tqdm
import os
from rwutils import *

def filter_test_query(target_gt, cutoff_index, out_dim = 10):
    out_gt = []
    for line_gt in tqdm(target_gt):
        # print(line_gt.shape)
        valid_indices_mask = line_gt  < cutoff_index
        res = line_gt[valid_indices_mask]
        # print(res.shape)
        if res.shape[0] >= out_dim:
            out_gt.append(res[:out_dim])
        else:
            raise ValueError
    return np.stack(out_gt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset", type=str, default="Webvid")
    args = parser.parse_args()
    
    num_self_test = 10000
    out_dim = 50
    
    ds = dataset_factory(args.dataset, read_mode = "all")
    print(ds.info())
    print(ds.train_gts)
    out_dir = ds.path + "_split"
    
    os.makedirs(out_dir, exist_ok=True)
    
    base_vectors = ds.data
    cutoff_index = base_vectors.shape[0] - num_self_test
    
    self_test_query = base_vectors[cutoff_index:]  # Last 10k vectors
    updated_base_vectors = base_vectors[:cutoff_index]
    
    write_fbin(os.path.join(out_dir, ds.base_fn), updated_base_vectors)
    write_fbin(os.path.join(out_dir, "self_"+ds.test_query_fn), self_test_query)
    
    del updated_base_vectors
    del self_test_query
    
    write_ibin(os.path.join(out_dir, ds.train_gt_fn), filter_test_query(ds.train_gts, cutoff_index, out_dim))
    write_ibin(os.path.join(out_dir, ds.test_gt_fn), filter_test_query(ds.test_gts, cutoff_index, out_dim))
    
    write_ibin(os.path.join(out_dir, ds.self_train_gt_fn), filter_test_query(ds.self_train_gts[:cutoff_index], cutoff_index, out_dim))
    write_ibin(os.path.join(out_dir, "self_"+ds.self_train_gt_fn), filter_test_query(ds.self_train_gts[cutoff_index:], cutoff_index, out_dim))
    
    
    
    