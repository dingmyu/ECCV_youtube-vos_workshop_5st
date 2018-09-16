#source /mnt/lustre/share/miniconda3/envsetup9.0.sh
#source activate r0.1.2

export LD_LIBRARY_PATH=/mnt/lustre/share/miniconda3/lib:$LD_LIBRARY_PATH                                                                                               
export LD_LIBRARY_PATH=/mnt/lustre/share/cuda-9.0/lib64:$LD_LIBRARY_PATH                                                                                               
export LD_LIBRARY_PATH=/mnt/lustre/share/nccl_2.1.15-1+cuda9.0_x86_64/lib:$LD_LIBRARY_PATH                                                                             
export LD_LIBRARY_PATH=/mnt/lustre/share/intel64/lib/:$LD_LIBRARY_PATH                                                                                                 
export PATH=/mnt/lustre/share/cuda-9.0/bin:$PATH
source /mnt/lustre/sunpeng/anaconda3/bin/activate
#source activate r0.1.2
