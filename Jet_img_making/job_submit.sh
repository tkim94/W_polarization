#!/bin/csh
 
i=1
# rm result13TeVlo.dat

# change the imputs of the MadEvent file
for (( j=1; j<=3; j+=1  ))
do

# go and run MadGraph
cd /afs/crc.nd.edu/user/t/tkim12/Work/MadGraph/TRR_clustering/pTwindow200-300/pp2wz/img_code

cat jet_img.sub | sed     -e "s/ number/ $j/" > jet_img_t_$j.sub

qsub jet_img_t_$j.sub

rm jet_img_t_$j.sub

((i+=1))
done
