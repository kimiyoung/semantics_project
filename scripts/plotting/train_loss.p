set title "Training Loss"
plot "<(sed 's/=/ /g' ". filename ." | grep TRAIN)" using 5 with lines
pause -1
