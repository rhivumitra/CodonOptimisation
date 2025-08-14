#!/bin/bash
#SBATCH --job-name=install_flash_attn
#SBATCH --output=test_flash_attn_%j.out
#SBATCH --error=test_flash_attn_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH -A l0003106
#SBATCH --partition=acc_short
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
##SBATCH -p acc_short
#S#BATCH --time=00:30:00  
# (Optional) Load modules if your HPC uses them
# module load singularity cuda

# Absolute path to your container:
CONTAINER="/home/rm15weti/pytorchContainer/pytorch_25.06.sif"

# Path to scratch directory
export TMPDIR="work/scratch/$USER/tmp_flash_attn_build"
mkdir -p "$TMPDIR"

echo "Using TMPDIR: $TMPDIR"
df -h "$TMPDIR"

# Confirm CUDA visible devices
nvidia-smi

#find and symlink libcuda.so.1
singularity exec --nv "$CONTAINER" ls -l /usr/local/cuda-12.9/compat/lib.real/libcuda.so
singularity exec --nv "$CONTAINER" bash -c 'export TRITON_LIBCUDA_PATH=/usr/local/cuda-12.9/compat/lib.real/libcuda.so'


# Print PyTorch version
singularity exec --nv "$CONTAINER" python -c "import torch; print('PyTorch version:', torch.__version__)"

# Install flash-attn in user mode inside the container
#singularity exec --nv --writable-tmpfs "$CONTAINER" MAX_JOBS=4 pip install flash-attn==2.8.0.post2 --no-build-isolation

# Install from requirements.txt
singularity exec --nv --writable-tmpfs "$CONTAINER" pip install --user -r /home/rm15weti/CodonStuff/CodonStuff/CodonStuff/StripedHyena-Hessian-7B/requirements.txt

# Verify installation
singularity exec --nv "$CONTAINER" python -c "import flash_attn; print('flash-attn installed successfully.')"

#Add symlink libcuda.so.1
singularity exec --nv "$CONTAINER" bash -c 'export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat/lib.real:$LD_LIBRARY_PATH'
singularity exec --nv "$CONTAINER" bash -c 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
# export MASTER_ADDR=MASTER_ADDR=127.0.0.1
# export MASTER_PORT=29500
# export WORLD_SIZE=$SLURM_NTASKS
# export RANK=$SLURM_PROCID
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Run your test script
#singularity exec --nv "$CONTAINER" python -m torch.distributed.launch --nproc_per_node=2 train_2.py
#singularity exec --nv "$CONTAINER" torchrun --nproc_per_node=2 train_2.py
#singularity exec --nv "$CONTAINER" torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29500 train_2.py

singularity exec --nv "$CONTAINER" python train.py

