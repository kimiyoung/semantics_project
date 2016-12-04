set title "Training Loss"
set key right top
plot "<(sed 's/=/ /g' ". filename ." | grep TRAIN)" using 5 every 100 title "train" with lines, \
    "<(sed 's/=/ /g' ". filename ." | grep VAL)" using 5 title "val" with lines
pause -1
