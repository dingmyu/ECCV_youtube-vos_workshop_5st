#nohup srun -p Test -n1 --gres=gpu:1 python -u object.py $1 $2 2>&1 > log/$1.log &

export PATH="/mnt/lustre/share/openmpi-1.8.5/bin:$PATH"
export LD_LIBRARY_PATH="/mnt/lustre/share/openmpi-1.8.5/lib:$LD_LIBRARY_PATH"

mkdir 0.5person_json

for((i=0;i<51;i++));  
do   
nohup srun -p Segmentation1080 -n1 --gres=gpu:1 python -u get_person_json.py $(expr $i \* 10) $(expr $i \* 10 + 10) 0.5 2>&1 > log/$(expr $i \* 10).log & 
done
