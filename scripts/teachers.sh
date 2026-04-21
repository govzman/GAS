#! /bin/bash

# # EDM
# edm_dataset_list=("cifar10" "ffhq" "afhqv2")
# for dataset in "${edm_dataset_list[@]}"
# do
# echo "Generate ${dataset}"
# torchrun --standalone --nproc_per_node=1 generate.py \
# 	--config=configs/edm/${dataset}.yaml \
# 	--outdir=data/teachers/edm/${dataset} \
# 	--seeds=00000-49999 \
# 	--batch=1024 \
# 	--create_dataset=True

# echo "Collate ${dataset}"
# python collate.py \
#     --synt_dir=data/teachers/edm/${dataset}/dataset \
#     --out_pkl=data/teachers/edm/${dataset}/dataset.pkl
# done

# LDM
# ldm_dataset_list=("cin256-v2")
# for dataset in "${ldm_dataset_list[@]}"
# do
# echo "Generate ${dataset}"
# torchrun --standalone --nproc_per_node=1 generate.py \
# 	--config=configs/ldm/${dataset}.yaml \
# 	--outdir=data/teachers/ldm/${dataset} \
# 	--seeds=00000-15 \
# 	--batch=1 \
# 	--create_dataset=True

# echo "Collate ${dataset}"
# python collate.py \
#     --synt_dir=data/teachers/ldm/${dataset}/dataset \
#     --out_pkl=data/teachers/ldm/${dataset}/dataset.pkl \
#     --num_samples=16
# done

# SD
# NFEs=(5 6 7 8)
NFEs=(5)
for NFE in "${NFEs[@]}"
do
echo "Generate SD nfe=${NFE}"
torchrun --standalone --nproc_per_node=1 generate.py \
    outdir="'data/teachers/sd-v1/nfe=${NFE}'" \
    seeds="00000-01449" \
    max_batch_size=4 \
    create_dataset=True

echo "Collate SD nfe=${NFE}"
python collate.py \
    --synt_dir=data/teachers/sd-v1/nfe=${NFE}/dataset \
    --out_pkl=data/teachers/sd-v1/nfe=${NFE}/dataset.pkl \
	--num_samples=1450
done
