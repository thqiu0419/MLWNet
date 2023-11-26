#!/bin/sh
cd $(dirname $0)
set -x
# 定义变量
GPUS=""
PARAMS=""
# 解析命令⾏参数
while [ $# -gt 0 ]; do
  case "$1" in
  --GPUS)
    GPUS="$2"
    shift 2
    ;;
  *)
    PARAMS="$PARAMS $1"
    shift
    ;;
  esac
done
GPU_NUMBER=$(echo "$GPUS" | sed 's/.*\([0-9]\)$/\1/')
PARAMS="$PARAMS"
echo "GPUS: $GPU_NUMBER"
echo "PARAMS: $PARAMS"
chmod 777 ./train.sh
./train.sh ${GPU_NUMBER} ${PARAMS}
