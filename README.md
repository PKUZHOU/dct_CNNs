
# An example of training and quantize resnet-18 in the frequency domain.

## Dependence:

    Pytorch==1.8.1

    MQBench: https://github.com/ModelTC/MQBench

## Train:

> We train the dct-input networks based on the official pretrained pytorch model

    Resnet-18: Acc@1 69.644 Acc@5 88.982
    
>  Load the dct-input Resnet-18 model using the pretrained weights 

    bash dct_train.sh

## Eval:
> After training, you can  test the accuracy or directly evalute our trained model without training:

    bash dct_eval.sh

> The pretrained model will show:

```
=> creating model 'dct_resnet18'
=> loading checkpoint 'checkpoints/dct_resnet18/model_best.pth.tar'
=> loaded checkpoint 'checkpoints/dct_resnet18/model_best.pth.tar' (epoch 104)
Test: [ 0/49]   Time 33.642 (33.642)    Loss 6.5714e-01 (6.5714e-01)    Acc@1  83.50 ( 83.50)   Acc@5  95.02 ( 95.02)
Test: [10/49]   Time  0.252 ( 3.312)    Loss 9.2553e-01 (9.5762e-01)    Acc@1  75.98 ( 75.51)   Acc@5  93.55 ( 92.44)
Test: [20/49]   Time  0.251 ( 1.864)    Loss 1.5670e+00 (9.7937e-01)    Acc@1  62.40 ( 74.72)   Acc@5  85.64 ( 92.55)
Test: [30/49]   Time  0.252 ( 1.355)    Loss 1.8532e+00 (1.1585e+00)    Acc@1  57.91 ( 71.16)   Acc@5  80.47 ( 90.11)
Test: [40/49]   Time  0.252 ( 1.086)    Loss 1.7424e+00 (1.2532e+00)    Acc@1  58.59 ( 69.28)   Acc@5  81.05 ( 88.72)
*   Acc@1 68.872 Acc@5 88.532
```

## Quantization:

> Run the following command to quantize the dct model:

    bash dct_quant.sh

> After quantization, you will see the accuracy after INT4 quantization:
```
Test: [  0/782] Time  0.893 ( 0.893)    Acc@1  84.38 ( 84.38)   Acc@5  95.31 ( 95.31)
Test: [100/782] Time  0.395 ( 0.150)    Acc@1  79.69 ( 74.81)   Acc@5  95.31 ( 91.43)
Test: [200/782] Time  0.377 ( 0.142)    Acc@1  71.88 ( 74.13)   Acc@5  90.62 ( 92.50)
Test: [300/782] Time  0.541 ( 0.146)    Acc@1  78.12 ( 74.70)   Acc@5  93.75 ( 92.92)
Test: [400/782] Time  0.179 ( 0.143)    Acc@1  64.06 ( 71.97)   Acc@5  92.19 ( 91.11)
Test: [500/782] Time  0.209 ( 0.142)    Acc@1  82.81 ( 70.48)   Acc@5  93.75 ( 89.89)
Test: [600/782] Time  0.041 ( 0.141)    Acc@1  78.12 ( 69.39)   Acc@5  89.06 ( 89.02)
Test: [700/782] Time  0.047 ( 0.140)    Acc@1  70.31 ( 68.40)   Acc@5  90.62 ( 88.28)
 * Acc@1 68.386 Acc@5 88.300
```