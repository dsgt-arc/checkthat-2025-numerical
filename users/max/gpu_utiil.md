## Specifics

#### Model
- ModernBERT-base: 149 billion parameters
- Classifier:
  - 1x Hidden Layer with 768x256 = 768x256 + 256 (bias) parameters
  - 1x Output Layer with 256x3 = 256x3 + 3 parameters
  - Total Classifier-Head = 197,635 parameters
- Total Encoder-Classifer: 149,000,197,635 parameters

#### Data & Train
- Dataset size:
  - Train: 9,935 examples
  - Val: 3,084 examples
- Batch size: 40
- Epochs: 5
- Batches/steps per epoch: examples/batch size ->  249

#### GPU Performance (TFLOPS/s)
- V100: 7 TFLOPs double-precision

## Calculation (w/o val)

- Total model parameters: 149,000,197,635
- Total training steps: 5 epochs X steps per epoch = 1,245

#### FLOPS estimate
- FLOPS per step = Number of parameters * 2 (forward and backward pass) * batch size  
                 = 149,000,197,635 * 2 * 40  
                 = 11.9 e^+12 (tera)  
- Total FLOPS for training = FLOPS per step * Total training steps  
                           = 1.19 e^+13 * 1,245  
                           = 14.8 e^+15 (peta)  

#### Time estimate
- Time = Total FLOPS for training / GPU performance (TFLOPS/s)  
         14,800 e^+12 / 7 e^+12  
       = 2,114 s  
       = 35h  