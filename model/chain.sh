#!/bin/bash
cd /data/s4977696/Urban_Computing/model
source ../venv/bin/activate
while pgrep -f "run_queue.py" > /dev/null; do sleep 60; done
python train_ind.py --mode ref -d PEMS08 -g 0
python train_ind.py --mode ref -d PEMS04 -g 0
python run_queue_ind.py 0,1
