from base_dataset import AnnDataset, AnnDatasetSelfTrain

class Sift1M(AnnDatasetSelfTrain):
    # http://corpus-texmex.irisa.fr/
    # {'nb': 1000000, 'dim': 128, 'train_queries': (100000, 128), 'train_gts': (100000, 100), 'test_queries': (10000, 128), 'test_gts': (10000, 100), 'self_train_gts': (1000000, 100)}
    def __init__(self, train_set_len = -1, self_train_set_len = -1, path = "../sift1M") -> None:
        super().__init__("sift-1M", "l2", path,  train_set_len, self_train_set_len)
        
        self.base_fn = 'sift_base.fvecs'
        
        self.train_query_fn = 'sift_learn.fvecs'
        self.train_gt_fn = 'sift-1M.learn_l2.100K.ivecs'
        
        self.test_query_fn = 'sift_query.fvecs'
        self.test_gt_fn = 'sift_groundtruth.ivecs'
    
        self.self_train_gt_fn = 'sift-1M.self_learn_l2.1M.ivecs'

class Text2image10M(AnnDatasetSelfTrain):
    # image unnormed
    # https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search
    def __init__(self, train_set_len = -1, self_train_set_len = -1, path = "../text2image") -> None:
        super().__init__("Text2Image-10M", "ip", path,  train_set_len, self_train_set_len)
        
        if train_set_len < 0:
            self.train_set_len = 10000000
        self.base_fn = 'base.1B.fbin.crop_nb_10000000.fbin'
        
        self.train_query_fn = 'query.learn.50M.fbin'
        self.train_gt_fn = 'text2image-10M.learn_ip.10M.ivecs'
        
        self.test_query_fn = 'query.public.10K.fbin'
        self.test_gt_fn = 'text2image-10M.10K.ibin'

        self.self_test_query_fn = 'self_query.public.10K.fbin'
        self.self_test_gt_fn = 'Text2Image-10M.self_test_ip.ivecs'
        
        self.self_train_gt_fn = 'text2image-10M.self_ip.ivecs'
     
class Deep100M(AnnDataset):
    # https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search
    # info = {'nb': 100000000, 'dim': 96, 'train_queries': (10000000, 96), 'train_gts': (10000000, 100), 'test_queries': (10000, 96), 'test_gts': (10000, 100), 'self_train_gts': (10000000, 100)}
    def __init__(self, train_set_len = -1, self_train_set_len = -1, path = "../deep") -> None:
        super().__init__("deep-100M","l2", path, train_set_len)
        
        self.base_fn = 'base.100M.fbin'
        
        self.train_query_fn = 'learn.10M.fbin'
        self.train_gt_fn = 'deep100M_gt.learn.10M.ivecs'
        
        self.test_query_fn = 'query.public.10K.fbin'
        self.test_gt_fn = 'deep100M_groundtruth.ivecs'
    
        self.self_train_gt_fn = 'deep100M_gt.self_learn.10M.ivecs'

class Webvid(AnnDatasetSelfTrain):
    # https://zenodo.org/records/11090378
    # {'nb': 2495000, 'dim': 512, 'train_queries': (1000000, 512), 'train_gts': (1000000, 50), 'test_queries': (10000, 512), 'test_gts': (10000, 50), 'self_train_gts': (2495000, 50), 'self_test_queries': (10000, 512), 'self_test_gts': (10000, 50)}
    # OOD
    # l2-normalized, l2 = ip
    def __init__(self, train_set_len = -1, self_train_set_len = -1, path = "/media/nd/webvid_split") -> None:
        super().__init__("webvid-2.5M", "ip", path, train_set_len, self_train_set_len)
        
        self.base_fn = 'clip.webvid.base.2.5M.fbin' # Video
        
        self.train_query_fn = 'webvid.query.train.2.5M.fbin' # Text
        self.train_gt_fn = 'webvid-2.5M.learn_ip.2.5M.ibin'
        
        self.test_query_fn = 'webvid.query.10k.fbin' # Text
        self.test_gt_fn = 'webvid-2.5M.10k.ibin'
        
        self.self_test_query_fn = 'self_webvid.query.10k.fbin'
        self.self_test_gt_fn = 'self_webvid-2.5M.self_learn_ip.2.5M.ibin'
         
        self.self_train_gt_fn = 'webvid-2.5M.self_learn_ip.2.5M.ibin'

class Laion(AnnDatasetSelfTrain):
    # https://zenodo.org/records/11090378
    # OOD
    # l2-normalized
    # {'nb': 10004480, 'dim': 512, 'train_queries': (1000000, 512), 'train_gts': (1000000, 100), 'test_queries': (10000, 512), 'test_gts': (10000, 100), 'self_train_gts': (10004480, 100)}
    def __init__(self, train_set_len = -1, self_train_set_len = -1, path = "../laion-10M_split") -> None:
        super().__init__("laion-10M", "ip", path, train_set_len, self_train_set_len)
        
        self.base_fn = 'base.10M.fbin' # Img
        
        self.train_query_fn = 'query.train.10M.fbin' # Text
        self.train_gt_fn = 'laion-10M.learn_ip.10M.ibin'
        
        self.test_query_fn = 'laion.query.10k.fbin' # Text
        # self.test_gt_fn = 'laion.gt.10k.ibin'
        self.test_gt_fn = 'laion.gt_ip.10k.ibin'
        
        self.self_test_query_fn = 'self_laion.query.10k.fbin'
        self.self_test_gt_fn = 'self_laion-10M.self_learn_ip.10M.ibin'
        
        self.self_train_gt_fn = 'laion-10M.self_learn_ip.10M.ibin'


dataset_dict = { 'Deep100M':Deep100M,'Laion':Laion,'Sift1M':Sift1M,'Text2image10M':Text2image10M,'Webvid':Webvid }

def dataset_factory(dataset_name, train_set_len = -1, self_train_set_len = -1, read_mode = "only_test"):
    if dataset_name in dataset_dict:
        ds_class = dataset_dict[dataset_name]
        if issubclass(ds_class, AnnDatasetSelfTrain):
            ds:AnnDatasetSelfTrain = ds_class(train_set_len=train_set_len, self_train_set_len = self_train_set_len)
        else:
            ds:AnnDataset = ds_class(train_set_len=train_set_len)
    else:
        raise NotImplementedError(dataset_name)
    if read_mode == "no_read":
        return ds
    if read_mode == "only_vecs":
        ds.read_vecs()
    elif read_mode == "only_test":
        if issubclass(ds_class, AnnDatasetSelfTrain):
            ds.read(load_train = False, load_self_train = False)
        else:
            ds.read(False)
    elif read_mode == "all":
        ds.read()
    elif read_mode == "query":
        if issubclass(ds_class, AnnDatasetSelfTrain):
            ds.read(load_train = True, load_self_train = False)
        else:
            ds.read()
    elif read_mode == "self":
        assert issubclass(ds_class, AnnDatasetSelfTrain)
        ds.read(load_train = False, load_self_train = True)
    else:
        raise NotImplementedError
    return ds
    

if __name__=="__main__":
    
    import inspect
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--getdict", "-g", action="store_true")
    parser.add_argument("-d","--dataset", type=str, default="")
    parser.add_argument("-i","--info", action="store_true")
    args = parser.parse_args()
    
    if args.getdict:
        classes = []
        for name, member in inspect.getmembers(__import__(__name__)):
            if inspect.isclass(member):
                classes.append(name)
                dataset_dict[name] = member
        print('{', ",".join([f"'{x}':{x}" for x in classes]) ,"}")

    if args.dataset != "":
        
        ds = dataset_dict[args.dataset](1000000)
        
        print(" ".join(ds.files()))
        
        if args.info:
            ds.read()
            print(ds.info())