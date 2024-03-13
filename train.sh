#!/usr/bin/env bash

nvidia-smi
set -x
echo $1
NGPUS=$1
PY_ARGS=${@:2}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

#ln -s /ai/9b5646dd16f9 /ai/volumn
python setup.py develop --no_cuda_ext

#python -m torch.distributed.launch --nproc_per_node=${NGPUS}  train.py ${PY_ARGS}
python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port=4321 basicsr/train.py ${PY_ARGS}


#python -m torch.distributed.launch --nproc_per_node=2 train.py --device 0,1 --lmdb true