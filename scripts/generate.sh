#! /bin/bash

NFE=8
checkpoint='02_03_18_07_39'
out_dir="CIN256_NFE_8_ts_8_linear_4800"
ITER_NUM=4800


torchrun --standalone --nproc_per_node=1 generate.py \
	--config="configs/ldm/cin256-v2.yaml" \
	--outdir=${out_dir} \
	--seeds=00000-15 \
	--batch=1 \
	--steps=${NFE} \
	--checkpoint_path=checkpoints/${checkpoint}/${ITER_NUM}.pt

# torchrun --standalone --nproc_per_node=1 fid.py calc \
# 	--images=${out_dir} \
# 	--ref=fid-refs/edm/cifar10-32x32.npz \
# 	--batch=1024 \
# 	--num=50000 >> FID_${out_dir}.txt

