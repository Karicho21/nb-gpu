#!/bin/bash
#SBATCH --job-name=nbody_gpu
#SBATCH --partition=GPU
#SBATCH --time=00:10:00     
#SBATCH --nodes=1             
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1         
#SBATCH --output=result_%j.out  
#SBATCH --error=timelog_%j.err   

module purge
module load cuda/12.4

echo "Compiling program..."
nvcc -arch=sm_61 -o nbody_gpu nbc.cu
if [ $? -ne 0 ]; then
    echo "Compilation failed, you suck"
    exit 1
fi

echo "Running solar system simulation"
./nbody_gpu planet 86400 100 10 128

echo "Running 1,000 particle simulation"
./nbody_gpu 1000 0.1 100 10 128

echo "Running 10,000 particle simulation"
./nbody_gpu 10000 0.1 100 10 128

echo "Running 100,000 particle simulation"
./nbody_gpu 100000 0.1 100 10 256

echo "All simulations done! yay"
