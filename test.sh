now=$(date +"%Y%m%d_%H%M%S")
python eval_imagenet.py  \
  --config configs/config_resnetv1ssn50_step_moving_average.yaml \
  --checkpoint_path model_zoo/ssn_8x2_75.848.pth \
  2>&1|tee test_${now}.log &\
