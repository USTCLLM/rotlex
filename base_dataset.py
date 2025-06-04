from rwutils import read_fbin, read_ibin, fvecs_read, ivecs_read
import os


class AnnDataset:

    def __init__(self, dataset_name, metric, path, train_set_len = -1) -> None:
        self.dataset_name = dataset_name
        self.metric = metric
        self.path = path
        self.train_set_len = train_set_len

        self.base_fn = None
        self.data = None
        
        self.train_query_fn = None
        self.train_gt_fn = None
        
        self.train_queries = None
        self.train_gts = None

        self.test_query_fn = None
        self.test_gt_fn = None
        
        self.test_queries = None
        self.test_gts = None

    @staticmethod
    def read_float(*args):
        if args[0].endswith(".fvecs"):
            return fvecs_read(*args)
        elif args[0].endswith(".fbin") or args[0].endswith(".bin"):
            return read_fbin(*args)
        else:
            raise NotImplementedError(args[0])
        
    @staticmethod
    def read_int32(*args):
        if args[0].endswith(".ivecs"):
            return ivecs_read(*args)
        elif args[0].endswith(".ibin") or args[0].endswith(".bin"):
            return read_ibin(*args)
        else:
            raise NotImplementedError(args[0])
    
    def read_vecs(self, load_train = True):
        self.test_queries = self.read_float(os.path.join(self.path, self.test_query_fn))
        if load_train:
            if self.train_set_len > 0:
                self.train_queries = self.read_float(os.path.join(self.path, self.train_query_fn), 0, self.train_set_len)
            else:
                self.train_queries = self.read_float(os.path.join(self.path, self.train_query_fn))
        self.data = self.read_float(os.path.join(self.path, self.base_fn))
    
    def read(self, load_train = True):
        self.test_gts = self.read_int32(os.path.join(self.path, self.test_gt_fn))
        if load_train:
            if self.train_set_len > 0:
                self.train_gts = self.read_int32(os.path.join(self.path, self.train_gt_fn), 0, self.train_set_len)
            else:
                self.train_gts = self.read_int32(os.path.join(self.path, self.train_gt_fn))
                self.train_set_len = self.train_gts.shape[0]
        self.read_vecs(load_train)

    def info(self):
        return {
            "nb": self.data.shape[0],
            "dim": self.data.shape[-1],
            "train_queries": self.train_queries.shape,
            "train_gts": self.train_gts.shape,
            "test_queries": self.test_queries.shape,
            "test_gts": self.test_gts.shape
        }
    
    def files(self):
        fs = []
        for k, v in self.__dict__.items():
            if k.endswith("_fn"):
                fs.append(os.path.join(self.path, v))
        return fs


class AnnDatasetSelfTrain(AnnDataset):
    
    def __init__(self, dataset_name, path, metric, train_set_len = -1, self_train_set_len = -1) -> None:
        super().__init__(dataset_name, path, metric, train_set_len)
        self.self_train_set_len = self_train_set_len
        
        self.self_test_query_fn = None
        self.self_test_queries = None
        
        self.self_test_gt_fn = None
        self.self_test_gts = None
        
        self.self_train_gt_fn = None
        self.self_train_gts = None

    def read(self, load_train = True, load_self_train = True):
        self.self_test_queries = self.read_float(os.path.join(self.path, self.self_test_query_fn))
        self.self_test_gts = self.read_int32(os.path.join(self.path, self.self_test_gt_fn))
        if load_self_train:
            if self.self_train_set_len > 0:
                self.self_train_gts = self.read_int32(os.path.join(self.path, self.self_train_gt_fn), 0, self.self_train_set_len)
            else:
                self.self_train_gts = self.read_int32(os.path.join(self.path, self.self_train_gt_fn))
                self.self_train_set_len = self.self_train_gts.shape[0]
        super().read(load_train)

    def info(self):
        res = super().info()
        res["self_train_gts"] = self.self_train_gts.shape
        res["self_test_queries"] = self.self_test_queries.shape
        res["self_test_gts"] = self.self_test_gts.shape
        return res