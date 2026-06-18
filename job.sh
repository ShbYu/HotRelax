#!/bin/bash
#BSUB -q 65305090ib!
#BSUB -gpu "num=1:aff=yes"
#BSUB -n 8
#BSUB -J trainMP
#BSUB -o train.out
#BSUB -e errortrain.out
source activate hotrelax2
export OMP_NUM_THREADS=1
#python train.py
python eval.py
