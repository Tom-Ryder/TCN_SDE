This code simulates the solution to three independent mean-reverting SDEs conditional on two observations per dimension.

Run the code by executing the following:
```
python3 inference.py GPU_ID EXPERIMENT_NAME_FOR_TENSORBOARD
```
If no GPU is present, use any integer value to default to the CPU (which will be much slower!).

Visualise losses during training with:
```
tensorboard --logdir=runs
```
`n` samples from the posterior will additionally plot and save to the local file `tmp_out.png` periodically (currently every 250 iterations). 
