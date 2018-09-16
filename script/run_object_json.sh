#nohup srun -p Segmentation1080 -n1 --gres=gpu:1 python -u object.py $1 $2 2>&1 > log/$1.log &

#export PATH="/mnt/lustre/share/openmpi-1.8.5/bin:$PATH"
#export LD_LIBRARY_PATH="/mnt/lustre/share/openmpi-1.8.5/lib:$LD_LIBRARY_PATH"

value1=$1
value2=$2

mkdir ${value1}${value2}object_json
mkdir ${value1}${value2}log
for((i=0;i<51;i++));
do
mkdir ${value1}${value2}$(expr $i \* 10)
done

for((i=0;i<51;i++));  
do   
nohup srun -p Segmentation1080 --mpi=pmi2 -n1 --gres=gpu:1 python -u get_object_json.py $(expr $i \* 10) $(expr $i \* 10 + 10) ${value1} ${value2} 2>&1 > ${value1}${value2}log/$(expr $i \* 10).log & 
done
