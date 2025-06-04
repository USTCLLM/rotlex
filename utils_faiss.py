import faiss

def to_faiss_metric(metric):
    if metric == "ip":
        return faiss.METRIC_INNER_PRODUCT
    if metric == "l2":
        return faiss.METRIC_L2
    raise NotImplementedError(metric)