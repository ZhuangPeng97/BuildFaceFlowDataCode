#!/bin/bash
for ((i=0;i<2;i++))
do
(
    start=$((${i}*1));
    end=$(($start+1));
    echo 1 ${start}
    echo 2 ${end}
    python build_flow.py test_frames.txt test 1 ${start} ${end};
    sleep 10
) &
done
wait
echo -E "########## $SECONDS ##########"
