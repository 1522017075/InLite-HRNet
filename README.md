# single-gpu testing
python tools/test.py 
${CONFIG_FILE} 
${CHECKPOINT_FILE} 
[--out ${RESULT_FILE}] 
[--eval ${EVAL_METRIC}] 
[--proc_per_gpu ${NUM_PROC_PER_GPU}] 
[--gpu_collect] 
[--tmpdir ${TMPDIR}] 
[--average_clips ${AVG_TYPE}]
[--launcher ${JOB_LAUNCHER}]
[--local_rank ${LOCAL_RANK}]
# lite-hrnet
python test.py configs/top_down/lite_hrnet/coco/litehrnet_18_coco_256x192.py weights/litehrnet_18_coco_256x192.pth 3 --out test_litehrnet_18_coco_256x192.txt --eval mAP
./tools/dist_test.sh configs/top_down/lite_hrnet/coco/litehrnet_18_coco_256x192.py weights/litehrnet_18_coco_256x192.pth 3 --eval mAP

python -m torch.distributed.launch --nproc_per_node=3 --master_port=29500 test.py configs/top_down/lite_hrnet/coco/litehrnet_18_coco_256x192.py \
weights/litehrnet_18_coco_256x192.pth --launcher pytorch --eval mAP

# dite-hrnet
python train.py configs/top_down/dite_hrnet/coco/ditehrnet_18_coco_256x192.py

navie_litehrnet_18_coco_256x192_e210_v3.pth

# navie-lite-hrnet
python -m torch.distributed.launch --nproc_per_node=3 --master_port=29500 train.py configs/top_down/naive_litehrnet/coco/wider_naive_litehrnet_18_coco_256x192.py --launcher pytorch

# my navie-lite-hrnet train
python -m torch.distributed.launch --nproc_per_node=3 --master_port=29500 train.py configs/top_down/dite_hrnet/coco/wider_naive_litehrnetv3_18_coco_256x192.py --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=3 --master_port=29500 train.py configs/top_down/dite_hrnet/coco/inlitehrnet_18_coco_256x192.py --launcher pytorch
# test
python -m torch.distributed.launch --nproc_per_node=3 --master_port=29500 test.py configs/top_down/dite_hrnet/coco/wider_naive_litehrnetv3_18_coco_256x192.py weights/navie_litehrnet_18_coco_256x192_e210_v3.pth --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=3 --master_port=29500 test.py configs/top_down/dite_hrnet/coco/inlitehrnet_18_coco_256x192.py weights/inlitehrnet_18_coco_256x192.pth --launcher pytorch
# test 18 - 384x288
python -m torch.distributed.launch --nproc_per_node=3 --master_port=29500 test.py configs/top_down/dite_hrnet/coco/inlitehrnet_18_coco_384x288.py weights/inlitehrnet_18_coco_384x288_e190.pth --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=3 --master_port=29500 test.py configs/top_down/dite_hrnet/coco/inlitehrnet_18_coco_256x192.py weights/navie_litehrnet_18_coco_256x192_e210_v3.pth --launcher pytorch


# train mpii
python -m torch.distributed.launch --nproc_per_node=3 --master_port=29500 train.py configs/top_down/dite_hrnet/mpii/inlitehrnet_18_mpii_256x256.py --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=3 --master_port=29500 test.py configs/top_down/dite_hrnet/mpii/inlitehrnet_18_mpii_256x256.py work_dirs/inlitehrnet_18_mpii_256x256/epoch_220.pth --launcher pytorch



# InLite 30 384*288
python -m torch.distributed.launch --nproc_per_node=3 --master_port=29500 train.py configs/top_down/dite_hrnet/coco/inlitehrnet_30_coco_384x288.py --launcher pytorch
CUDA_VISIBLE_DEVICES='2' python train.py configs/top_down/dite_hrnet/coco/wider_naive_litehrnetv3_18_coco_256x192.py
CUDA_VISIBLE_DEVICES='1' python train.py configs/top_down/dite_hrnet/coco/inlitehrnet_30_coco_384x288.py
CUDA_VISIBLE_DEVICES='0' python train.py configs/top_down/dite_hrnet/coco/inlitehrnet_18_coco_384x288.py



# InLiteV1（大论文 + LMA

python -m torch.distributed.launch --nproc_per_node=3 --master_port=29500 train.py configs/top_down/dite_hrnet/coco/inlitehrnetv1_18_coco_256x192.py --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=3 --master_port=29500 train.py configs/top_down/dite_hrnet/coco/inlitehrnetv1_18_coco_384x288.py --launcher pytorch

# InLiteV1(CInlite-HRNet COCO test值)
python -m torch.distributed.launch --nproc_per_node=3 --master_port=29500 test.py configs/top_down/dite_hrnet/coco/inlitehrnetv1_18_coco_256x192.py weights/inlitehrnetv1_18_coco_256x192.pth --launcher pytorch
python -m torch.distributed.launch --nproc_per_node=3 --master_port=29500 test.py configs/top_down/dite_hrnet/coco/inlitehrnetv1_18_coco_384x288.py weights/inlitehrnetv1_18_coco_384x288.pth --launcher pytorch

# multiple-gpu testing
./tools/dist_test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRIC}] \
    [--proc_per_gpu ${NUM_PROC_PER_GPU}] [--gpu_collect] [--tmpdir ${TMPDIR}] [--average_clips ${AVG_TYPE}] \
    [--launcher ${JOB_LAUNCHER}] [--local_rank ${LOCAL_RANK}]


conda create -n ziyang_hrnet --clone ziyang_ditehrnet

# 计算网络复杂度
python summary_network.py configs/top_down/dite_hrnet/coco/inlitehrnet_18_coco_384x288.py
python summary_network.py configs/top_down/dite_hrnet/coco/inlitehrnetv1_18_coco_256x192.py
python test_speed.py configs/top_down/dite_hrnet/coco/inlitehrnetv1_18_coco_256x192.py
python summary_network.py configs/top_down/dite_hrnet/coco/wider_naive_litehrnetv3_18_coco_256x192.py

# 可视化log
tensorboard --logdir ${WORK_DIR}/${TIMESTAMP}/vis_data
conda activate xinru-pet
tensorboard --logdir E:\zzy\LearnWorkspace\LearnDeepLearning\Dite-HRNet-main\work_dirs
# DSC 创新点，dropout特征图，算出所有通道的注意力，把最小的一半扔掉，然后剩下的进行GhostNet增值
