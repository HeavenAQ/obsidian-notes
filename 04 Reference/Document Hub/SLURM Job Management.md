---
base: "[[Document Hub.base]]"
Created by: Heaven Chen
Created time: 2026-06-05T12:34:00
Status: In-Progress
Last updated time: 2026-06-05T12:38:00
Last edited by: Heaven Chen
Category:
  - GPU
  - Linux
---
## Architecture


Slurm is basically the **operating system layer for a shared compute cluster**.

When many people share the same machines, especially expensive GPU servers, you need something that answers questions like:

Who gets which GPU?

For how long?

On which node?

With how many CPUs and how much memory?

What happens when two users request the same resource?

How do we record usage, enforce limits, and prevent people from accidentally occupying the whole cluster?

That is what Slurm does.

Slurm describes itself as an open-source, fault-tolerant, highly scalable cluster management and job scheduling system for Linux clusters. Its three core jobs are: allocate resources to users, start and monitor work on those resources, and manage the queue of pending work. ([Slurm](https://slurm.schedmd.com/overview.html?utm_source=chatgpt.com))

---

## 1. The basic idea

Imagine a cluster like this:

```plain text
Login node
   |
   | users submit jobs
   v
Slurm controller
   |
   | assigns work
   v
Compute nodes
   - node001: 4 GPUs, 64 CPUs, 512 GB RAM
   - node002: 8 GPUs, 128 CPUs, 1 TB RAM
   - node003: CPU-only
   - ...
```

As a user, you usually do **not** directly SSH into a GPU node and start training manually. Instead, you submit a job request:

```plain text
I need:
- 1 node
- 2 GPUs
- 16 CPU cores
- 64 GB RAM
- 24 hours
- run this training script
```

Slurm decides when and where that job should run.

If resources are available, it starts the job. If not, it puts the job in a queue. This is why you see jobs as `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, or `CANCELLED`.

---

## 2. Why Slurm exists

Without Slurm, a shared GPU server becomes chaotic.

For example, suppose there are 4 A100 GPUs and 10 users. If everyone SSHs into the machine and runs Python manually, several problems happen:

One user may accidentally use all GPUs.

Two users may start jobs on the same GPU and crash each other’s training.

Nobody knows who is using what.

Long jobs can block urgent jobs.

The administrator cannot enforce fair usage.

There is no clean accounting of GPU hours, CPU hours, memory, failures, or job history.

Slurm solves this by making resource use **explicit**. You do not just “run a script.” You request resources, Slurm grants an allocation, and your script runs inside that allocation.

That is why, in your earlier situation, freeing CUDA memory is different from releasing Slurm resources. CUDA only knows what your process is using. Slurm knows what your job was allocated.

---

## 3. The main design concept: allocation first, execution second

A very important Slurm idea is the distinction between:

```plain text
Resource allocation
```

and

```plain text
Program execution
```

When you submit a job, Slurm first creates an **allocation**. That allocation may include CPUs, memory, GPUs, nodes, time limit, and sometimes licenses or other special resources.

Then, inside that allocation, Slurm launches one or more **job steps**.

Conceptually:

```plain text
Job allocation:
    "This job owns GPU 0 and GPU 1 on node004 for 24 hours."

Job step:
    "Run python train.py inside that allocation."
```

This is why a job can reserve multiple GPUs even if your Python process only uses one of them. Slurm’s reservation is based on what the job requested, not only on what `nvidia-smi` currently shows.

---

## 4. Slurm architecture

A typical Slurm cluster has several important components.

### `slurmctld`: the central controller

`slurmctld` is the brain of the cluster.

It keeps track of:

```plain text
- all submitted jobs
- job states
- node states
- available CPUs, memory, GPUs
- scheduling policies
- partitions
- priorities
- limits
```

When you submit a job, the request goes to `slurmctld`. It decides where the job can run.

You can think of it as the scheduler and resource manager.

---

### `slurmd`: the daemon on each compute node

Each compute node runs a daemon called `slurmd`.

Its job is to:

```plain text
- receive instructions from slurmctld
- start job processes on that node
- monitor those processes
- report status back to the controller
- clean up after jobs finish
```

So if `slurmctld` is the brain, `slurmd` is the worker agent installed on every compute server.

Example:

```plain text
slurmctld: "node007, start Heaven's job using 2 GPUs."
slurmd on node007: "Okay, launching the job and reporting status."
```

---

### `slurmdbd`: accounting database daemon

Many clusters also use `slurmdbd`.

This component records job history and accounting information:

```plain text
- who ran what
- how long it ran
- how many CPUs/GPUs were allocated
- memory usage
- exit status
- job failures
- usage by user, group, project, or account
```

This is useful for billing, fair-share scheduling, lab usage reports, and debugging completed jobs.

---

### Database

Behind `slurmdbd`, there is usually a database such as MariaDB or MySQL. Slurm uses it to store historical job accounting data.

This is why you can later inspect finished jobs even after they disappear from the active queue.

---

### `slurm.conf`

Slurm’s behavior is mostly controlled through configuration files, especially `slurm.conf`.

That file defines things like:

```plain text
- what nodes exist
- how many CPUs/memory/GPUs each node has
- what partitions exist
- scheduling parameters
- time limits
- accounting settings
```

Official Slurm documentation describes `slurm.conf` as the file that defines general Slurm configuration, managed nodes, partitions, and scheduling parameters. ([Slurm](https://slurm.schedmd.com/slurm.conf.html?utm_source=chatgpt.com))

---

## 5. Partitions: Slurm’s version of queues

A Slurm **partition** is like a queue or resource pool.

For example, a cluster might have:

```plain text
cpu
gpu
a100
debug
long
short
highmem
```

Each partition can have different rules:

```plain text
gpu partition:
    has GPU nodes
    max time: 48 hours

debug partition:
    short jobs only
    max time: 30 minutes
    faster scheduling

long partition:
    long-running CPU jobs
    max time: 7 days

a100 partition:
    only A100 GPU nodes
    stricter access policy
```

When you submit a job, you often choose a partition. If you choose the wrong one, the job may never run, may be rejected, or may run on the wrong type of node.

---

## 6. Nodes, tasks, CPUs, memory, and GPUs

Slurm was originally designed heavily around HPC-style workloads, so it thinks in terms of nodes and tasks.

A **node** is a physical or virtual machine.

A **task** is usually a process launched by Slurm. In distributed computing, one task might correspond to one MPI rank or one process.

A **CPU** usually means a CPU core or hardware thread, depending on cluster configuration.

Memory can be requested per node or per CPU.

GPUs are handled as special schedulable resources. Slurm uses the concept of **GRES**, or Generic Resources, for resources like GPUs. The official documentation says Slurm can define and schedule arbitrary Generic RESources, with built-in features for GPUs, CUDA MPS, and related GPU mechanisms. ([Slurm](https://slurm.schedmd.com/gres.html?utm_source=chatgpt.com))

So in a GPU job, Slurm is not just saying:

```plain text
Run this somewhere.
```

It is saying:

```plain text
Run this on a node where the requested GPU, CPU, memory, and time constraints can be satisfied.
```

---

## 7. How GPU scheduling works conceptually

On a GPU cluster, Slurm needs to know:

```plain text
node001 has 4 A100 GPUs
node002 has 8 H100 GPUs
node003 has 4 RTX 6000 GPUs
```

Then, when a job requests GPUs, Slurm chooses a node that satisfies that request.

For example:

```plain text
Job A requests 1 GPU.
Job B requests 4 GPUs.
Job C requests 8 GPUs.
```

Slurm may place them like this:

```plain text
node001: Job A uses 1 GPU, Job B cannot fit anymore if it needs 4 full GPUs
node002: Job C uses all 8 GPUs
```

Depending on configuration, Slurm may also isolate visible GPUs through environment variables like `CUDA_VISIBLE_DEVICES`, and possibly through Linux cgroups. But the degree of isolation depends on how the cluster administrator configured Slurm, GRES, and cgroup settings.

This is why sometimes you may see only one GPU inside a job, while the physical machine has several GPUs.

---

## 8. Cgroups and enforcement

Slurm can use Linux **cgroups** to enforce resource boundaries.

Cgroups can limit or isolate things like:

```plain text
- CPU usage
- memory usage
- device access
- GPU device visibility
```

Without cgroup enforcement, Slurm might set `CUDA_VISIBLE_DEVICES`, but a user could theoretically bypass it in some configurations. With stricter cgroup device enforcement, the job can only access the devices assigned to it.

This matters a lot on shared GPU clusters.

Conceptually:

```plain text
Slurm allocation:
    You get GPU 2.

Environment:
    CUDA_VISIBLE_DEVICES=0
    because inside your job, assigned GPU 2 may appear as local GPU 0.

Cgroup:
    blocks access to physical GPUs you were not assigned.
```

That remapping is a common source of confusion. Inside your job, `CUDA_VISIBLE_DEVICES=0` does not always mean physical GPU 0. It may mean “the first GPU assigned to this job.”

---

## 9. Scheduling: how Slurm decides what runs

Slurm scheduling is not simply first-come, first-served.

Clusters often use policies such as:

```plain text
priority
fair share
partition limits
QoS limits
account limits
job size
requested time
resource availability
reservation policies
```

For example, your job may be pending because:

```plain text
- no GPU is currently free
- your group has reached its GPU limit
- your requested time is too long
- you requested a specific GPU type that is unavailable
- another user has higher priority
- your job needs 4 GPUs on the same node, but only scattered GPUs are free
```

This is why a smaller job can sometimes start before a larger job. Slurm may backfill smaller jobs if they can run without delaying higher-priority reservations.

---

## 10. Backfilling

Backfilling is one of the important scheduling ideas in Slurm.

Suppose a big job is waiting for 8 GPUs, and Slurm predicts those GPUs will become available in 3 hours. In the meantime, there may be 1 free GPU.

Instead of leaving that GPU idle, Slurm can run a short job that only needs 1 GPU and will finish before the big job starts.

Conceptually:

```plain text
Large job:
    needs 8 GPUs
    expected start: 3 hours later

Small job:
    needs 1 GPU
    time limit: 30 minutes

Slurm:
    runs the small job now because it will not delay the large job
```

This improves cluster utilization.

This is also why requesting a more accurate time limit can help your job start sooner. If you request 7 days for a job that actually takes 20 minutes, Slurm may be less willing to backfill it.

---

## 11. Slurm’s mental model vs normal Linux process management

This is the part that directly relates to your earlier question.

Normal Linux process management asks:

```plain text
What processes are running?
What memory are they using?
What GPU memory are they using?
Can I kill this PID?
```

Slurm asks:

```plain text
What job owns this allocation?
What resources were reserved?
Which user/account/QoS does it belong to?
What is its time limit?
Which node and partition is it using?
```

So there are two layers:

```plain text
Slurm layer:
    job allocation owns resources

Linux/CUDA layer:
    actual processes consume CPU/GPU memory
```

Example:

```plain text
Slurm job requested 4 GPUs.
Python currently uses only GPU 0.
GPUs 1, 2, 3 show no memory usage in nvidia-smi.
```

From CUDA’s perspective:

```plain text
Only GPU 0 is being used.
```

From Slurm’s perspective:

```plain text
The job still owns 4 GPUs.
```

That is why you usually cannot “release just part of the job” cleanly while keeping the running script alive. Slurm allocated the resources to the job as a unit.

---

## 12. Common Slurm use cases

### Machine learning training

This is probably your main case.

Typical workloads:

```plain text
- train a detection model
- run distributed PyTorch training
- fine-tune a vision-language model
- run multiple experiments with different configs
- evaluate checkpoints
```

Slurm is useful because GPUs are scarce and expensive. It prevents everyone from fighting over the same GPU manually.

---

### HPC simulation

Slurm is very common in traditional high-performance computing.

Examples:

```plain text
- climate simulation
- molecular dynamics
- physics simulation
- fluid dynamics
- weather modeling
- computational chemistry
```

These workloads often use many nodes and many CPU cores, sometimes with MPI.

---

### Batch data processing

Slurm can also run non-GPU batch jobs:

```plain text
- preprocess large datasets
- extract video frames
- run feature extraction
- convert annotations
- generate embeddings
- run large evaluation jobs
```

Even if a job does not need GPUs, Slurm helps schedule CPU and memory fairly.

---

### Hyperparameter sweeps

Slurm job arrays are often used for:

```plain text
learning rate = 1e-3, 1e-4, 1e-5
batch size = 8, 16, 32
model = small, medium, large
seed = 1, 2, 3
```

Instead of manually launching 100 scripts, users submit an array. Each task runs one configuration.

---

### Shared research lab infrastructure

In a university lab, Slurm is useful because many students share the same machines.

It gives the lab:

```plain text
- fair GPU sharing
- job history
- user accountability
- less accidental interference
- easier debugging
- centralized resource policy
```

For example, your lab may say:

```plain text
Each user can use at most 2 GPUs.
Long jobs must use the long partition.
Debug jobs must be under 30 minutes.
A100 nodes require special access.
```

Slurm enforces those rules.

---

### Production-like internal compute

Some companies also use Slurm for internal model training or research infrastructure, especially when workloads are batch-oriented rather than always-on services.

It is less like Kubernetes and more like:

```plain text
Run this compute job using these resources, then finish.
```

That makes it good for training, evaluation, simulation, and offline pipelines.

---

## 13. Slurm vs Kubernetes

Since you work with cloud/data systems too, this comparison may help.

Slurm and Kubernetes both schedule workloads, but they were designed for different worlds.

```plain text
Slurm:
    designed for HPC and batch jobs
    strong for finite jobs
    strong for GPU/CPU cluster scheduling
    common in universities and supercomputing centers
    job starts, runs, finishes

Kubernetes:
    designed for services and containers
    strong for long-running microservices
    strong for autoscaling web/backend systems
    common in cloud-native infrastructure
    pod should stay alive and be restarted if it dies
```

For example:

```plain text
Train RF-DETR for 20 hours:
    Slurm is natural.

Run a web API for users:
    Kubernetes is natural.

Run many batch experiments on shared GPUs:
    Slurm is natural.

Deploy a scalable inference service:
    Kubernetes may be more natural.
```

There is overlap, but their design centers are different.

---

## 14. Slurm’s strengths

Slurm is popular because it is:

```plain text
- scalable to very large clusters
- good at batch scheduling
- mature in HPC environments
- flexible with partitions, QoS, priorities, and accounting
- good for GPU and CPU resource allocation
- script-friendly
- widely supported in research computing
```

For ML researchers, its biggest practical strength is that it gives a shared GPU cluster a clear resource-management system.

---

## 15. Slurm’s weaknesses or pain points

Slurm can also feel annoying because:

```plain text
- jobs may wait in queue
- resource requests must be precise
- cluster-specific configuration varies a lot
- GPU visibility can be confusing
- error messages are sometimes indirect
- interactive debugging is less convenient than local running
- modifying running jobs is limited
```

The most important limitation you already ran into:

```plain text
Slurm resources are allocated at the job level.
```

So if you request too many GPUs, Slurm may reserve them even if your code does not use them.

---

## 16. A simple end-to-end example

Conceptually, a Slurm workflow looks like this:

```plain text
1. User writes a training script.

2. User writes a job request:
       I need 1 GPU, 16 CPUs, 64 GB memory, 24 hours.

3. User submits the job.

4. slurmctld receives the request.

5. Slurm checks:
       Which partition?
       Which nodes have enough resources?
       What is the user’s priority?
       Are there account/QoS limits?
       Can the job start now?

6. If resources are unavailable:
       job becomes PENDING.

7. If resources are available:
       Slurm chooses a node.

8. slurmd on that node starts the job.

9. The job runs inside its allocation.

10. Slurm tracks state, time, resource usage, and exit code.

11. When the job ends:
       resources are released,
       accounting records are saved.
```

That is the core lifecycle.

---

## 17. The key mental model

For your GPU training work, the most useful mental model is:

```plain text
Slurm does not manage “Python scripts” first.
Slurm manages resource allocations first.
Your Python script is something that runs inside an allocation.
```

So when you request:

```plain text
4 GPUs
```

Slurm thinks:

```plain text
This job owns 4 GPUs until the job ends.
```

Even if your script only uses one GPU, the Slurm-level reservation remains.

That is the main reason Slurm is powerful, but also why it can feel rigid. It protects shared resources by treating them as scheduled allocations, not just whatever the current process happens to consume.

## Commands

Slurm’s core user commands include `sbatch`, `srun`, `salloc`, `squeue`, `scontrol`, `scancel`, `sacct`, `sstat`, and `sinfo`. Officially, `sbatch` submits batch scripts, `srun` launches applications or job steps, `scancel` cancels/signals jobs, `squeue` shows queued/running jobs, `sacct` shows accounting data, and `scontrol` views or modifies Slurm state. ([slurm.schedmd.com](https://slurm.schedmd.com/man_index.html?utm_source=chatgpt.com))

---

## 1. Check cluster / GPU availability

```bash
sinfo
```

Show partitions:

```bash
sinfo -s
```

Show nodes and their state:

```bash
sinfo -N -l
```

Show GPU-related node info, if your cluster exposes GRES:

```bash
sinfo -o "%P %N %t %G"
```

Useful columns:

```bash
sinfo -o "%20P %10a %10l %10D %20G %N"
```

Meaning:

```plain text
%P  partition
%a  availability
%l  time limit
%D  number of nodes
%G  generic resources, often GPUs
%N  node list
```

---

## 2. Submit a batch job

Basic:

```bash
sbatch train.sh
```

With job name:

```bash
sbatch --job-name=my_train train.sh
```

Request 1 GPU:

```bash
sbatch --gres=gpu:1 train.sh
```

Request specific GPU type, if supported:

```bash
sbatch --gres=gpu:a100:1 train.sh
```

Request CPUs, memory, time:

```bash
sbatch \
  --job-name=rf_detr_train \
  --gres=gpu:1 \
  --cpus-per-task=16 \
  --mem=64G \
  --time=24:00:00 \
  train.sh
```

Specify partition:

```bash
sbatch -p gpu --gres=gpu:1 train.sh
```

Common `sbatch` flags:

```bash
-J, --job-name          job name
-p, --partition         partition/queue
--gres=gpu:N           request N GPUs
--gres=gpu:a100:N      request N A100 GPUs, if configured
--nodes=N              number of nodes
--ntasks=N             number of tasks
--ntasks-per-node=N    tasks per node
--cpus-per-task=N      CPU cores per task
--mem=64G              memory per node
--mem-per-cpu=4G       memory per CPU
--time=HH:MM:SS        wall time
--output=file.out      stdout log
--error=file.err       stderr log
--array=0-9            job array
--dependency=afterok:<jobid> run after another job succeeds
```

Example script:

```bash
#!/bin/bash
#SBATCH --job-name=rf_detr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

nvidia-smi

uv run python train.py
```

`%x` means job name. `%j` means job ID.

---

## 3. Check your jobs

Show your jobs:

```bash
squeue -u $USER
```

Show all jobs:

```bash
squeue
```

Show a specific job:

```bash
squeue -j <job_id>
```

Better custom format:

```bash
squeue -u $USER -o "%.18i %.9P %.30j %.8u %.2t %.10M %.10l %.6D %R"
```

Meaning:

```plain text
i  job ID
P  partition
j  job name
u  user
t  state
M  used time
l  time limit
D  number of nodes
R  reason / node list
```

Watch your jobs live:

```bash
watch -n 2 squeue -u $USER
```

Check why a job is pending:

```bash
squeue -j <job_id> -o "%.18i %.9P %.30j %.8u %.2t %.10M %.10l %R"
```

Common pending reasons:

```plain text
Resources       waiting for CPUs/GPUs/memory
Priority        waiting behind higher-priority jobs
ReqNodeNotAvail requested node unavailable
PartitionTimeLimit requested time exceeds partition limit
QOSMaxGRES      requested too many GPUs for your QoS
AssocGrpGRES    your group/user GPU limit reached
```

---

## 4. Inspect a running or pending job

Detailed job info:

```bash
scontrol show job <job_id>
```

More readable:

```bash
scontrol show job <job_id> | tr ' ' '\n'
```

Important fields to look for:

```plain text
JobState=
Reason=
RunTime=
TimeLimit=
NumNodes=
NumCPUs=
TRES=
GRES=
NodeList=
Command=
WorkDir=
StdOut=
StdErr=
```

For example:

```bash
scontrol show job 1482582 | tr ' ' '\n' | grep -E "JobState|Reason|RunTime|TimeLimit|TRES|GRES|NodeList|Command|WorkDir|StdOut|StdErr"
```

---

## 5. Cancel jobs safely

Cancel one job:

```bash
scancel <job_id>
```

Cancel all your jobs:

```bash
scancel -u $USER
```

Cancel only pending jobs:

```bash
scancel -u $USER --state=PENDING
```

Cancel only running jobs:

```bash
scancel -u $USER --state=RUNNING
```

Cancel by job name:

```bash
scancel --name=my_train
```

Cancel a job array element:

```bash
scancel <array_job_id>_<task_id>
```

Example:

```bash
scancel 123456_7
```

Send a signal instead of immediately killing:

```bash
scancel --signal=TERM <job_id>
```

or:

```bash
scancel --signal=USR1 <job_id>
```

This is useful if your training script catches the signal and saves a checkpoint.

---

## 6. Check finished jobs

Show recent jobs:

```bash
sacct
```

Show your jobs with useful fields:

```bash
sacct -u $USER --format=JobID,JobName,Partition,State,ExitCode,Elapsed,Timelimit,AllocCPUS,ReqMem,MaxRSS
```

Show one job:

```bash
sacct -j <job_id>
```

Better detailed view:

```bash
sacct -j <job_id> --format=JobID,JobName%30,State,ExitCode,Elapsed,Timelimit,AllocCPUS,ReqMem,MaxRSS,NodeList
```

For GPU jobs, your cluster may support TRES fields:

```bash
sacct -j <job_id> --format=JobID,JobName,State,Elapsed,AllocTRES,ReqTRES
```

If supported:

```bash
sacct -j <job_id> --format=JobID,JobName,State,Elapsed,TRESUsageInAve,TRESUsageInMax
```

`sacct` displays accounting data for jobs and job steps, and its output can be customized with `--format`. ([slurm.schedmd.com](https://slurm.schedmd.com/sacct.html?utm_source=chatgpt.com))

---

## 7. Monitor resource usage while running

For a running job:

```bash
sstat -j <job_id>
```

For job steps:

```bash
sstat -j <job_id>.batch
```

Useful format:

```bash
sstat -j <job_id>.batch --format=JobID,AveCPU,AveRSS,MaxRSS,MaxVMSize
```

But for GPU memory/utilization, usually use:

```bash
nvidia-smi
```

Watch it:

```bash
watch -n 1 nvidia-smi
```

Show only processes:

```bash
nvidia-smi pmon -c 1
```

Show GPU utilization repeatedly:

```bash
nvidia-smi dmon
```

Show process list:

```bash
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv
```

Show GPU status:

```bash
nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total --format=csv
```

---

## 8. Start an interactive GPU session

Interactive shell with 1 GPU:

```bash
salloc --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00
```

Then inside the allocation:

```bash
srun --pty bash
```

Or directly:

```bash
srun --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash
```

Then check:

```bash
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
```

Use this for debugging before submitting a long `sbatch` job.

---

## 9. Run commands inside an existing allocation

Inside an `sbatch` or `salloc` job:

```bash
srun python train.py
```

Run with specific task settings:

```bash
srun --ntasks=1 --cpus-per-task=16 python train.py
```

For distributed training on one node with multiple GPUs:

```bash
srun --ntasks=4 --gres=gpu:4 python train.py
```

But for PyTorch, you often use:

```bash
torchrun --nproc_per_node=4 train.py
```

inside a job requested with:

```bash
#SBATCH --gres=gpu:4
```

or:

```bash
srun torchrun --nproc_per_node=4 train.py
```

depending on cluster convention.

---

## 10. Attach to a running job step

If a job step was launched with `srun`, you may be able to attach:

```bash
sattach <job_id>.<step_id>
```

For example:

```bash
sattach 123456.0
```

`sattach` attaches to a running Slurm job step and exposes its I/O streams, but it cannot attach directly to some batch/extern steps. ([slurm.schedmd.com](https://slurm.schedmd.com/sattach.html?utm_source=chatgpt.com))

In practice, many training jobs are easier to monitor through log files:

```bash
tail -f logs/job.out
```

or:

```bash
tail -f slurm-<job_id>.out
```

---

## 11. Modify pending jobs

You can often modify a **pending** job:

```bash
scontrol update JobId=<job_id> TimeLimit=12:00:00
```

Change job name:

```bash
scontrol update JobId=<job_id> Name=new_name
```

Change dependency:

```bash
scontrol update JobId=<job_id> Dependency=afterok:<other_job_id>
```

Hold a pending job:

```bash
scontrol hold <job_id>
```

Release it:

```bash
scontrol release <job_id>
```

Important: modifying a **running** job is much more limited. Many changes, especially resource changes like GPUs, may fail or require admin permission. `scontrol` can view or modify Slurm state, but many operations require elevated permissions. ([slurm.schedmd.com](https://slurm.schedmd.com/scontrol.html?utm_source=chatgpt.com))

---

## 12. Job arrays

Submit many similar jobs:

```bash
sbatch --array=0-9 train_array.sh
```

Inside the script:

```bash
echo $SLURM_ARRAY_TASK_ID
python train.py --config configs/config_${SLURM_ARRAY_TASK_ID}.yaml
```

Limit concurrency:

```bash
sbatch --array=0-99%4 train_array.sh
```

This runs at most 4 array tasks at the same time.

Cancel one array task:

```bash
scancel <job_id>_<task_id>
```

Cancel array tasks 1 to 3:

```bash
scancel <job_id>_[1-3]
```

Slurm supports cancelling either a whole job array or specific array elements. ([slurm.schedmd.com](https://slurm.schedmd.com/job_array.html?utm_source=chatgpt.com))

---

## 13. Dependencies

Run job B only after job A succeeds:

```bash
jid=$(sbatch train.sh | awk '{print $4}')
sbatch --dependency=afterok:$jid eval.sh
```

Common dependency types:

```bash
after:<jobid>       after job starts
afterok:<jobid>     after job finishes successfully
afternotok:<jobid>  after job fails
afterany:<jobid>    after job finishes, regardless of result
```

Example:

```bash
sbatch --dependency=afterok:123456 eval.sh
```

---

## 14. Useful environment variables inside a job

Inside your script:

```bash
echo $SLURM_JOB_ID
echo $SLURM_JOB_NAME
echo $SLURM_SUBMIT_DIR
echo $SLURM_NODELIST
echo $SLURM_JOB_NODELIST
echo $SLURM_CPUS_PER_TASK
echo $SLURM_NTASKS
echo $SLURM_PROCID
echo $SLURM_LOCALID
echo $SLURM_ARRAY_TASK_ID
echo $CUDA_VISIBLE_DEVICES
```

Very useful debug block:

```bash
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Submit dir: $SLURM_SUBMIT_DIR"
echo "Node list: $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
hostname
nvidia-smi
```

---

## 15. Commands for your current situation

Find your jobs:

```bash
squeue -u $USER
```

Inspect the job:

```bash
scontrol show job <job_id> | tr ' ' '\n'
```

Check what GPUs Slurm allocated:

```bash
scontrol show job <job_id> | grep -E "TRES|GRES|NodeList"
```

Check actual GPU usage:

```bash
nvidia-smi
```

Find processes on GPUs:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Kill a specific unwanted process only:

```bash
kill <pid>
```

Force kill:

```bash
kill -9 <pid>
```

Cancel the whole Slurm job:

```bash
scancel <job_id>
```

Cancel all your pending jobs only:

```bash
scancel -u $USER --state=PENDING
```

---

## 16. My personal “minimum Slurm toolkit”

These are the ones I would memorize first:

```bash
# submit
sbatch train.sh

# submit GPU job
sbatch --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=24:00:00 train.sh

# see my jobs
squeue -u $USER

# inspect one job
scontrol show job <job_id> | tr ' ' '\n'

# cancel one job
scancel <job_id>

# see completed job info
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES,ReqTRES,MaxRSS

# interactive GPU shell
srun --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash

# monitor GPU
watch -n 1 nvidia-smi
```

For your GPU training work, the most important mental model is:

```plain text
sbatch / salloc = request resources
srun = run something inside resources
squeue = see queue
scontrol = inspect job deeply
sacct = inspect after running
scancel = stop job
nvidia-smi = inspect actual GPU usage
```