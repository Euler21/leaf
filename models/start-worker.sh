#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

echo "starting ray worker node"
ray start --address $1 --redis-password=$2
# echo "$$" | tee ~/pid_storage/worker${2}.pid

sleep infinity

# echo "Worker ${2} stopped"
