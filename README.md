This  is  for ITCS 4145 Nbody CUDA Assignmnnet

But before you begin, make sure you have access to

- A Linux-based system with SLURM workload management
- gcc compiler (this is critical to execute the program)
- Access to hpc lab computer Centaurus
- UNC Charlotte VPN if you are outside of campus or not using "eduroam" network

Steps to compile and experiment:

1. Connecting hpc lab computer by "ssh hpc-student.charlotte.edu -l your-username"
2. Authenticate with Duo
3. Type "sbatch nb.sh" It will create two files. One is "timelog_?????.err", this contains execution times. "result_?????.out" file contains the data calculated in each command.
4. Wait a bit for command to finish running and record the time it takes.
5. "cat" command lets you see the file you desire.  Just type "cat 'filename' " to view.
6. If you would like a csv file recording the time, type "sbatch nb.sh > timelog.csv". It will schedule the job and record the time onto csv file called timelog.csv. You can name the file whatever you desire, but this timelog is what I named.


Sample output:
- Execution time: 116 ms
- Execution time: 241 ms
- Execution time: 1868 ms
- Execution time: 93088 ms

Time varies each time, but should not be far from sample record.
