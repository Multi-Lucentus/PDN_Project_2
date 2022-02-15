#!/bin/bash
# Choose the queue that you want to use
#SBATCH --partition=oucspdn_cpu
#SBATCH --exclusive
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --chdir=/home/oucspdn007/Project_2/Problem_1
#SBATCH --output=/home/oucspdn007/Project_2/Problem_1/Problem1_%J_stdout.txt
#SBATCH --error=/home/oucspdn007/Project_2/Problem_1/Problem1_%J_stderr.txt
# Adjust time according to your needs and queue limit 
#SBATCH --time=00:15:00
#SBATCH --job-name=Project2_Problem1
#SBATCH --mail-user=parker.a.brandt-1@ou.edu@ou.edu
#SBATCH --mail-type=ALL

#################################################

echo "**Test 1 - 1 threads**"
./parallel_mult_mat_mat test1_input_mat_a.csv 1000 1000 test1_input_mat_b.csv 1000 1000 1 output1.csv test1_time_1p.csv

echo "**Test 1 - 2 threads**"
./parallel_mult_mat_mat test1_input_mat_a.csv 1000 1000 test1_input_mat_b.csv 1000 1000 2 output1.csv test1_time_2p.csv

echo "**Test 1 - 4 threads**"
./parallel_mult_mat_mat test1_input_mat_a.csv 1000 1000 test1_input_mat_b.csv 1000 1000 4 output1.csv test1_time_4p.csv

echo "**Test 1 - 8 threads**"
./parallel_mult_mat_mat test1_input_mat_a.csv 1000 1000 test1_input_mat_b.csv 1000 1000 8 output1.csv test1_time_8p.csv

echo "**Test 2 - 1 threads**"
./parallel_mult_mat_mat test2_input_mat_a.csv 1000 1000 test2_input_mat_b.csv 1000 2000 1 output2.csv test2_time_1p.csv

echo "**Test 2 - 2 threads**"
./parallel_mult_mat_mat test2_input_mat_a.csv 1000 1000 test2_input_mat_b.csv 1000 2000 2 output2.csv test2_time_2p.csv

echo "**Test 2 - 4 threads**"
./parallel_mult_mat_mat test2_input_mat_a.csv 1000 1000 test2_input_mat_b.csv 1000 2000 4 output2.csv test2_time_4p.csv

echo "**Test 2 - 8 threads**"
./parallel_mult_mat_mat test2_input_mat_a.csv 1000 1000 test2_input_mat_b.csv 1000 2000 8 output2.csv test2_time_8p.csv

echo "**Test 3 - 1 threads**"
./parallel_mult_mat_mat test2_input_mat_a.csv 2000 1000 test2_input_mat_b.csv 1000 2000 1 output2.csv test3_time_1p.csv

echo "**Test 3 - 2 threads**"
./parallel_mult_mat_mat test2_input_mat_a.csv 2000 1000 test2_input_mat_b.csv 1000 2000 2 output2.csv test3_time_2p.csv

echo "**Test 3 - 4 threads**"
./parallel_mult_mat_mat test2_input_mat_a.csv 2000 1000 test2_input_mat_b.csv 1000 2000 4 output2.csv test3_time_4p.csv

echo "**Test 3 - 8 threads**"
./parallel_mult_mat_mat test2_input_mat_a.csv 2000 1000 test2_input_mat_b.csv 1000 2000 8 output2.csv test3_time_8p.csv

echo "**Test 4 - 1 threads**"
./parallel_mult_mat_mat test2_input_mat_a.csv 2000 2000 test2_input_mat_b.csv 2000 2000 1 output2.csv test4_time_1p.csv

echo "**Test 4 - 2 threads**"
./parallel_mult_mat_mat test2_input_mat_a.csv 2000 2000 test2_input_mat_b.csv 2000 2000 2 output2.csv test4_time_2p.csv

echo "**Test 4 - 4 threads**"
./parallel_mult_mat_mat test2_input_mat_a.csv 2000 2000 test2_input_mat_b.csv 2000 2000 4 output2.csv test4_time_4p.csv

echo "**Test 4 - 8 threads**"
./parallel_mult_mat_mat test2_input_mat_a.csv 2000 2000 test2_input_mat_b.csv 2000 2000 8 output2.csv test4_time_8p.csv

# ...
