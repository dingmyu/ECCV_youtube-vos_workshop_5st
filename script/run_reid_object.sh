#nohup srun -p Segmentation1080 -n1 --gres=gpu:1 python -u object.py $1 $2 2>&1 > log/$1.log &

#export PATH="/mnt/lustre/share/openmpi-1.8.5/bin:$PATH"
#export LD_LIBRARY_PATH="/mnt/lustre/share/openmpi-1.8.5/lib:$LD_LIBRARY_PATH"
value1=$1                                                                                                 
mkdir ${value1}Annotations
#mkdir ${value1}${value2}log                                                                                          
#for((i=0;i<48;i++));                                                                                                 
#do                                                                                                                   
#mkdir ${value1}${value2}$(expr $i \* 10)                                                                             
#done

#nohup srun -p Segmentation1080 -n1 --gres=gpu:1 python -u reid_object.py 56 60 ${value1} 2>&1 > log/${value1}56.log & 
for((i=0;i<51;i++));  
do   
nohup srun -p Segmentation1080 -n1 --gres=gpu:1 python -u reid_object.py $(expr $i \* 10) $(expr $i \* 10 + 10) ${value1} 2>&1 > log/${value1}$(expr $i \* 10).log & 
done
