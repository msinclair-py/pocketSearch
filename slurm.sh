#!/bin/bash
TARGET=${1}
INIT=${2}
SCAFFOLD=${3}
NCPU=${4}


ray start --head --num-cpus $NCPU
python pocketsearch.py $TARGET $INIT $SCAFFOLD -mp True
