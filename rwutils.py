import struct
import numpy as np
import os

"""
                  IO Utils
"""
def ivecs_read(fname, start_idx=0, chunk_size=None):
    
    with open(fname, "rb") as f:
        f.seek(0, os.SEEK_END)
        sz = f.tell()
        f.seek(0, 0)
        d = int(np.fromfile(f, count=1, dtype=np.int32))
        # print(d, sz)
        assert sz % (4 * (d + 1)) == 0
        nvecs = sz // (4 * (d + 1))
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        f.seek(0, 0)
        a = np.fromfile(f, count = nvecs * (d + 1), dtype=np.int32, offset = start_idx * 4 * (d + 1))
        return a.reshape(-1, d + 1)[:, 1:].copy()
    # a = np.fromfile(fname, dtype='int32')
    # d = a[0]
    # return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname, start_idx=0, chunk_size=None):
    return ivecs_read(fname, start_idx, chunk_size).view('float32')

def ivecs_write(fname, m):
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    m1.tofile(fname)

def fvecs_write(fname, m):
    m = m.astype('float32')
    ivecs_write(fname, m.view('int32'))

def read_fbin(filename, start_idx=0, chunk_size=None):
    """ Read *.fbin file that contains float32 vectors
    Args:
        :param filename (str): path to *.fbin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read. 
                                 If None, read all vectors
    Returns:
        Array of float32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=int(nvecs) * int(dim), dtype=np.float32, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)


def read_ibin(filename, start_idx=0, chunk_size=None):
    """ Read *.ibin file that contains int32 vectors
    Args:
        :param filename (str): path to *.ibin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of int32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=int(nvecs) * int(dim), dtype=np.int32, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)


def write_fbin(filename, vecs):
    """ Write an array of float32 vectors to *.fbin file
    Args:s
        :param filename (str): path to *.fbin file
        :param vecs (numpy.ndarray): array of float32 vectors to write
    """
    assert len(vecs.shape) == 2, "Input array must have 2 dimensions"
    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape
        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))
        vecs.astype('float32').flatten().tofile(f)

        
def write_ibin(filename, vecs):
    """ Write an array of int32 vectors to *.ibin file
    Args:
        :param filename (str): path to *.ibin file
        :param vecs (numpy.ndarray): array of int32 vectors to write
    """
    assert len(vecs.shape) == 2, "Input array must have 2 dimensions"
    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape
        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))
        vecs.astype('int32').flatten().tofile(f)