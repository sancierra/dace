#!/bin/zsh
read metrics < metrics.txt
for ((i=1;i<=9;i++)); 
do  
    ncu --metrics $metrics python3 isolated.py $i > output$i.txt
done 

