#!/bin/bash
NAME=${1}
NUM=${2}
SCAFFOLD=${3}
TARGET=${4}
OUTPUT=${5}

python slurm-launch.py --exp-name $NAME --num-nodes $NUM --partition normal --load-env 'conda activate pocketSearch' --command "python pocketSearch.py $SCAFFOLD $OUTPUT $TARGET"
