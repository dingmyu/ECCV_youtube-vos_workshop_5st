#nohup srun -p Segmentation1080 -n1 --gres=gpu:1 python -u object.py $1 $2 2>&1 > log/$1.log &

#export PATH="/mnt/lustre/share/openmpi-1.8.5/bin:$PATH"
#export LD_LIBRARY_PATH="/mnt/lustre/share/openmpi-1.8.5/lib:$LD_LIBRARY_PATH"

mkdir 0.5Annotations
for((i=0;i<51;i++));  
do   
nohup srun -p Segmentation1080 -n1 --gres=gpu:1 python -u reid_person.py $(expr $i \* 10) $(expr $i \* 10 + 10) 0.5 2>&1 > log/$(expr $i \* 10).log & 
done
#for((i=0;i<24;i++));  
#do   
#nohup srun -p Test -n1 --gres=gpu:1 python -u reid_person.py $(expr $i \* 20) $(expr $i \* 20 + 20) 0.7 2>&1 > log/$(expr $i \* 20).log & 
#done
#for((i=0;i<24;i++));  
#do   
#nohup srun -p Test -n1 --gres=gpu:1 python -u reid_person.py $(expr $i \* 20) $(expr $i \* 20 + 20) 0.75 2>&1 > log/$(expr $i \* 20).log & 
#done
#for((i=0;i<24;i++));  
#do   
#nohup srun -p Test -n1 --gres=gpu:1 python -u reid_person.py $(expr $i \* 20) $(expr $i \* 20 + 20) 0.8 2>&1 > log/$(expr $i \* 20).log & 
#done
#for((i=0;i<24;i++));  
#do   
#nohup srun -p Test -n1 --gres=gpu:1 python -u reid_person.py $(expr $i \* 20) $(expr $i \* 20 + 20) 0.85 2>&1 > log/$(expr $i \* 20).log & 
#done
#CUDA_VISIBLE_DEVICES=0 python -u get_person_json.py 0 50 2>&1 > log/0.log &
#CUDA_VISIBLE_DEVICES=1 python -u get_person_json.py 50 100 2>&1 > log/1.log &
#CUDA_VISIBLE_DEVICES=2 python -u get_person_json.py 100 150 2>&1 > log/2.log &
#CUDA_VISIBLE_DEVICES=3 python -u get_person_json.py 150 200 2>&1 > log/3.log &
#CUDA_VISIBLE_DEVICES=0 python -u get_person_json.py 250 300 2>&1 > log/4.log &
#CUDA_VISIBLE_DEVICES=1 python -u get_person_json.py 200 250 2>&1 > log/5.log &
#CUDA_VISIBLE_DEVICES=2 python -u get_person_json.py 300 350 2>&1 > log/6.log &
#CUDA_VISIBLE_DEVICES=3 python -u get_person_json.py 350 400 2>&1 > log/7.log &
#CUDA_VISIBLE_DEVICES=0 python -u get_person_json.py 400 450 2>&1 > log/8.log &
####CUDA_VISIBLE_DEVICES=1 python -u get_person_json.py 450 480 2>&1 > log/9.log &
