export CUDA_VISIBLE_DEVICES=0

python train.py -a  dct_resnet18 --resume checkpoints/dct_resnet18/model_best.pth.tar /datasets/imagenet --eval --batch-size 1024 --workers 32 
