#!/bin/csh
 
i=1

for ((j=1; j<=2; j+=1 ))
do

# run python
cd /Volumes/Tae_backup-8TB/MadGraph/MG5_aMC_v2_7_0/Delphes

now=$(date +"%T")
echo "Current time : $now"

for ((k=0; k<=4; k+=1 ))
do

echo "dim6 pT200-300"
python examples/pp2wz-pT200_Jet_clustering_TRR.py /Volumes/Tae_backup-8TB/Research/6_ML_convolutional-NN_jet-classif/Events/dim6_pp2wz/c3w/3_10-3/pT200-300/ "$j" "$k" >& /Volumes/Tae_backup-8TB/Research/6_ML_convolutional-NN_jet-classif/Events/dim6_pp2wz/c3w/3_10-3/pT200-300/TRR_clustering/clustering_output/clustering_run_"$j"-"$k".out

echo "dim6 pT400-500"
python examples/pp2wz-pT400_Jet_clustering_TRR.py /Volumes/Tae_backup-8TB/Research/6_ML_convolutional-NN_jet-classif/Events/dim6_pp2wz/c3w/3_10-3/pT400-500/ "$j" "$k" >& /Volumes/Tae_backup-8TB/Research/6_ML_convolutional-NN_jet-classif/Events/dim6_pp2wz/c3w/3_10-3/pT400-500/TRR_clustering/clustering_output/clustering_run_"$j"-"$k".out


done
((i+=1))
done
