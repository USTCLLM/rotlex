# ROLTEX

Implentmentation of *Robust Tree-based Learned Vector Index with Query-aware Repartitioning*.

# Prerequisites

Install our modified [faiss](https://github.com/weiwch/faiss_preassigned) that supports add_preassigned & search_preassigned for IVF indexes(`IndexIVF` and `IndexShardsIVF`). Set `CMAKE_CUDA_ARCHITECTURES`, `BLA_VENDOR` and `DFAISS_OPT_LEVEL` below appropriately for your system. You can refer to the faiss documentation for details. (We are working to merge this feature into the main branch; see [this issue](https://github.com/facebookresearch/faiss/issues/3908)).

```
cd faiss_preassigned

cmake -B _build \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILD_TESTING=OFF \
      -DFAISS_OPT_LEVEL=avx512 \
      -DFAISS_ENABLE_GPU=ON \
      -DFAISS_ENABLE_RAFT=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DFAISS_ENABLE_PYTHON=OFF \
      -DBLA_VENDOR=Intel10_64lp \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DCMAKE_BUILD_TYPE=Release .

make -C _build -j$(nproc) faiss faiss_avx2 faiss_avx512

cmake --install _build --prefix _libfaiss_stage/

cmake -B _build_python \
      -Dfaiss_ROOT=_libfaiss_stage/ \
      -DFAISS_OPT_LEVEL=avx512 \
      -DFAISS_ENABLE_GPU=ON \
      -DFAISS_ENABLE_RAFT=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DPython_EXECUTABLE=$PYTHON \
      faiss/python

make -C _build_python -j$(nproc) swigfaiss swigfaiss_avx2 swigfaiss_avx512

cp -v _libfaiss_stage/lib/libfaiss* _build_python
```

Install other dependencies:

```
conda create -n anns_env python=3.11 tqdm swig mkl=2023 mkl-devel=2023 numpy scipy pytest cmake loguru tensorboard pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

Alternatively, you can install the dependencies from yml.

```
conda env create -n anns_env -f environment.yml
```

Install our compiled faiss into the conda environment created.

```
conda activate anns_env
cd faiss_preassigned
pip install .
```

# Run

All datasets, including database vectors, training query vectors, test query vectors, and ground truth, are stored in fvecs or bin format. Refer to `dataset.py` for details on dataset handling. The tree and neural network structures for the algorithms, along with training hyperparameters and their descriptions, can be found in `config.py`. Alternatively, you can use a JSON file for configuration, e.g. `conf.json`, specifying it at runtime.

```
python3 main.py -f conf.json -d cuda:0 -i cuda:0
```

This will train using a single GPU. The device specified on the command line will take precedence over the configuration file. Use the `-d` flag to specify the device for the neural network and training processes, and the `-i` flag to designate the device where the inverted lists are stored. Logs and checkpoints will be saved under `./logs/<DATASET_NAME>/<NAME_OF_THIS_RUN>/`.

# Evaluation

To evaluate a specific checkpoint, run the following command:

```
python3 main.py -v ./logs/<DATASET_NAME>/<NAME_OF_THIS_RUN>/<CHECKPONITS> 
```

This command will evaluate the performance of the chosen checkpoint, display the results, and save them to a file named `time_recall_<CURRENT_TIMESTAMP>_.json`. You can specify an alternative configuration file using the `-f` option. Additionally, you can control the number of inverted lists to probe by using the `--start`, `--stop`, and `--interval` flags, which respectively specify the starting number, stopping number, and the interval for the range of inverted lists to evaluate, to balance of the latency and recall.


# References

Please cite ROTLEX in your publications with the following bibtex if it helps your research:

```
@inproceedings{10.1145/3711896.3737112,
author = {Wei, Wenqing and Lian, Defu and Feng, Qingshuai and Wu, Yongji},
title = {Robust Tree-based Learned Vector Index with Query-aware Repartitioning},
year = {2025},
isbn = {9798400714542},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3711896.3737112},
doi = {10.1145/3711896.3737112},
booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
pages = {3134–3143},
numpages = {10},
keywords = {approximate nearest neighbor search (anns), learning-to-index, maximum inner product search (mips), vector retrieval},
location = {Toronto ON, Canada},
series = {KDD '25}
}
```

# License

MIT