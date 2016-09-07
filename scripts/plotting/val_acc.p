set title "Validation Acc"
plot "<(sed 's/=/ /g' ". filename ." | grep VAL)" using 7 with lines
pause -1
