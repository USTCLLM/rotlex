import copy
import torch

class Config:
    def __init__(self, **kwargs) -> None:
        # dataset
        self.dataset_name = "Webvid"
        self.used_label_num = 10
        self.train_set_len = -1
        self.self_train_set_len = -1
        
        # net
        self.net_type = "deepnet_sep" # 2 encoder: deepnet_sep, 1 encoder: deepnet
        self.mlp_wth = [512]
        self.code_emb_dim = 512
        self.device = "cuda:0"
        self.bn = True
        self.drop = 1e9
        self.activate = "swish"
        
        # tree
        self.k = 128
        self.num_layers = 2
        self.R = 1
        self.ivf_device = "cuda:0"
        
        # train config
        self.epoch = 60
        self.lr = 1e-3
        self.bs = 5000
        self.val_bs = 100
        self.es_patient = 40
        self.layer_weight = [1, 1]
        self.sample_thres = 20000
        self.sample_num = 16384//50
        self.val_interval = 5
        self.upd_interval = 20
        self.upd_patient = 1000000
        self.load_best_upd = False
        self.balance_factor = 1.5
        self.norm_query = False
        # init
        self.spherical = False
        self.init = "kmeans"
        self.max_iter = 1
        self.reinit_after_upd = False
        self.train_mode = "all" # all, query or self
        # update
        self.upd_log_softmax = True
        self.upd_norm = False
        self.upd_method = "rotlex"
        self.upd_assign_mcmf = True
        self.upd_on_query = True
        # load config
        self.reconstruct = False
        self.load_tree = ""
        self.load_all = ""
        self.extra = ""
        
        # eval
        self.eval_topk = 100
        self.eval_num = 10000
        self.eval_beam = 100
        self.set(**kwargs)

    def set(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.__dict__ and (
                self.__dict__[k] is None or type(v) == type(self.__dict__[k])
            ):
                self.__setattr__(k, v)
            else:
                print(k, v)
                raise NotImplementedError(k)

    def getname(self):
        return "k{}_l{}_{}_{}-{}_{}_{}".format(
            self.k,
            self.num_layers,
            self.used_label_num,
            "-".join(map(str, self.mlp_wth)),
            self.code_emb_dim,
            self.activate,
            self.extra,
        )

    def get_hparams(self):
        d = copy.deepcopy(self.__dict__)
        res = {}
        for k, v in d.items():
            if isinstance(v, list):
                v = torch.LongTensor(v)
            res[k] = v
        return res


if __name__ == "__main__":
    import json

    conf = Config(epoch=400)
    with open("conf.json", "w") as f:
        json.dump(conf.__dict__, f)
    print(json.dumps(conf.__dict__))
    # conf.set(epoch = 30)
    # print(conf.__dict__)
    print(conf.get_hparams())
