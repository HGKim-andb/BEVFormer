# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test
export PYTHONPATH="${PWD}:${PYTHONPATH}"

Train BEVFormer with 8 GPUs 
```
./tools/dist_train.sh ./projects/configs/bevformer/bevformer_base.py 8
```
서버
tools/dist_train.sh  projects/configs/bevformer/bevformer_tiny.py  3  --work-dir ./work_dirs/bevformer_tiny_mini
tools/dist_train.sh       projects/configs/bevformer/bevformer_small_20percent.py  3  --work-dir ./work_dirs/bevformer_small_20percent
z490
python tools/train.py     projects/configs/bevformer/bevformer_tiny.py     --work-dir ./work_dirs/bevformer_tiny_mini


Eval BEVFormer with 8 GPUs
```
./tools/dist_test.sh ./projects/configs/bevformer/bevformer_base.py ./path/to/ckpts.pth 8
```
Note: using 1 GPU to eval can obtain slightly higher performance because continuous video may be truncated with multiple GPUs. By default we report the score evaled with 8 GPUs.



# Using FP16 to train the model.
The above training script can not support FP16 training, 
and we provide another script to train BEVFormer with FP16.

```
./tools/fp16/dist_train.sh ./projects/configs/bevformer_fp16/bevformer_tiny_fp16.py 8
```


# Visualization 

see [visual.py](../tools/analysis_tools/visual.py)