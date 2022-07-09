imagenet_dir=/datasets/imagenet

python train.py -a dct_resnet18 $imagenet_dir --pretrained --batch-size 2048 --workers 32 --epochs 160

# pretrained pytorch model:
# *   Acc@1 69.644 Acc@5 88.982


