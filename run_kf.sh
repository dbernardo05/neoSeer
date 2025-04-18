#!/bin/bash

# regen=false

# if [ "$regen" = false ]; then
#   echo "#######################################"
#   echo "# Regenerating data"
#   for i in 0 
#     do
#     	echo "                                  "
#     	echo "##################################"
#       echo "Fold $i"
#       # python run_tsai_v19v2_DL_JohnsPaper_AFE.py -k $i -r
#       python run_tsai_v19f_DL_JohnsPaper.py -k $i -r

#   done

for i in 0 1 2 3 4 5 6 7 8 9
  do
    echo "                                  "
    echo "##################################"
    echo "Fold $i"
    # using hydra
    python run_tsai.py kfold=$i max_epochs=5

    # python run_tsai_v19f_DL_JohnsPaper.py -k $i 
    # python run_tsai_v19_DL_JohnsPaper_MLcomps.py -k $i 
    # python run_tsai_v23mL_skl_KNN.py -k $i 
    # python run_tsai_v25mL_xgb.py -k $i 
    # python run_tsai_v19v2_DL_JohnsPaper_AFE.py -k $i 

  # mv "history.csv" "logs/history_kf${i}.csv"

done
