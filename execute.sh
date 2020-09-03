#!/bin/bash
for test_recurrence_num in 50 70 100 120 150
do
    for local_search_iteration in 2000 2500 3000 3500
    do
        for ((epsilon=30; epsilon<=50;epsilon+=5 ));do
            temp=$(echo "$epsilon*0.01"|bc)
#            echo $temp

            echo "python3 build/scripts-3.6/satyr.py config/Predict/PDP-np-nd-np-gcnf-10-100-pytorch.yaml datasets/test/test $test_recurrence_num -w $local_search_iteration -e $temp"
        done
    done
done