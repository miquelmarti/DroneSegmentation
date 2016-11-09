./trainNetwork.py --weights $1 $2 --device 0 2>&1 | tee $2/solve_$(date "+%F-%T").log
