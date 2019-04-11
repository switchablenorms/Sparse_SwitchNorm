now=$(date +"%Y%m%d_%H%M%S")
python -u -m torch.distributed.launch --nproc_per_node=8 train_imagenet.py  \
  --config configs/config_resnetv1ssn50_step_moving_average.yaml \
  --workers 2 \
  --print_freq 100 \
  2>&1|tee train_${now}.log &\
