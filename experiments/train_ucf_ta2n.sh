CUDA_VISIBLE_DEVICES=0,1 python run/run_ta2n.py \
--dataset 'ucf' \
--shot 1 \
--backbone 'resnet50' \
--scratch '/home/sjtu/data' \
--metric 'cos' \
--timewise \
--num_gpus 2 \
--num_workers 8 \
--seq_len 8 \
--query_per_class 5 \
--img_size 224 \
--checkpoint_dir 'checkpoint/ucf/1shot/ta2n/baseline_ta2n' \
--test_iter 200 \
