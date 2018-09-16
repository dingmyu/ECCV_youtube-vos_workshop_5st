#!/bin/bash

counter=0
while [ $counter -lt $3 ]
do
    python davis_test.py test-dev configs/test_config.py --output out_test --cache cache_test --gpu_num 1 --gpu $counter &
    ((counter++))
done
