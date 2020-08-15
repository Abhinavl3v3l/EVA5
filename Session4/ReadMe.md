# Assignment 4 



**ASSIGNMENT - Create an architecture using below conditions**

- 99.4% validation accuracy
- Less than 20k Parameters
- Less than 20 Epochs
- No fully connected layer



----

#### Concepts Used

- Convolution - (3x3 and Pointwise Convolution)
- Batch Normalization 
- Activation - Relu
- DropOuts (0.1 - 0.3)

#### Architecture

Block 1 - Conv2d(3x3) > Relu > BN > DO >Conv2D(3x3) > Relu > BN > DO >Conv2d(1x1) > MaxPool (as far away from last layer)

Block2 -  Conv2D(3x3) > Relu > BN >DO >Conv2D(3x3) > RELU > BN >DO 

Block3 - Conv2D(3x3) > BN > DO > Conv(3x3) > RELU > BN > DO

Block 4 - Conv(3x3) >RELU > BN > DO > Conv2D

```python
x = self.pool1(self.one1(self.do2(self.bn2(F.relu(self.conv2(self.do1(self.bn1(F.relu(self.conv1(x))))))))))
x = self.do4(self.bn4(F.relu(self.conv4(self.do3(self.bn3(F.relu(self.conv3(x))))))))
x = self.do6(self.bn6(F.relu(self.conv6(self.do5(self.bn5(F.relu(self.conv5(x))))))))
x = self.conv8(self.do7(self.bn7(F.relu(self.conv7(x)))))
```

### Model Summary

```txt
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 12, 26, 26]             120
       BatchNorm2d-2           [-1, 12, 26, 26]              24
           Dropout-3           [-1, 12, 26, 26]               0
            Conv2d-4           [-1, 20, 24, 24]           2,180
       BatchNorm2d-5           [-1, 20, 24, 24]              40
           Dropout-6           [-1, 20, 24, 24]               0
            Conv2d-7           [-1, 10, 24, 24]             210
         MaxPool2d-8           [-1, 10, 12, 12]               0
            Conv2d-9           [-1, 16, 10, 10]           1,456
      BatchNorm2d-10           [-1, 16, 10, 10]              32
          Dropout-11           [-1, 16, 10, 10]               0
           Conv2d-12             [-1, 16, 8, 8]           2,320
      BatchNorm2d-13             [-1, 16, 8, 8]              32
          Dropout-14             [-1, 16, 8, 8]               0
           Conv2d-15             [-1, 16, 6, 6]           2,320
      BatchNorm2d-16             [-1, 16, 6, 6]              32
          Dropout-17             [-1, 16, 6, 6]               0
           Conv2d-18             [-1, 16, 4, 4]           2,320
      BatchNorm2d-19             [-1, 16, 4, 4]              32
          Dropout-20             [-1, 16, 4, 4]               0
           Conv2d-21             [-1, 16, 2, 2]           2,320
      BatchNorm2d-22             [-1, 16, 2, 2]              32
          Dropout-23             [-1, 16, 2, 2]               0
           Conv2d-24             [-1, 10, 1, 1]             650
================================================================
```



#### Total Parameters -  14,120

#### Model Run

```
 0%|          | 0/469 [00:00<?, ?it/s]
loss=0.07585278898477554 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 34.77it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0503, Accuracy: 9838/10000 (98.38%)

loss=0.05997771397233009 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.89it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0367, Accuracy: 9886/10000 (98.86%)

loss=0.09265804290771484 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.45it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0333, Accuracy: 9895/10000 (98.95%)

loss=0.10768947005271912 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 36.22it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0363, Accuracy: 9889/10000 (98.89%)

loss=0.04319295287132263 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 36.18it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0346, Accuracy: 9888/10000 (98.88%)

loss=0.03285282105207443 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 36.08it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0289, Accuracy: 9916/10000 (99.16%)

loss=0.0400102362036705 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.97it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0258, Accuracy: 9922/10000 (99.22%)

loss=0.05393126979470253 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 36.30it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0272, Accuracy: 9906/10000 (99.06%)

loss=0.08353656530380249 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 36.85it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0251, Accuracy: 9922/10000 (99.22%)

loss=0.012827788479626179 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 36.65it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0271, Accuracy: 9917/10000 (99.17%)

loss=0.022555382922291756 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 36.76it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0227, Accuracy: 9927/10000 (99.27%)

loss=0.010289781726896763 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 36.06it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0232, Accuracy: 9930/10000 (99.30%)

loss=0.016766713932156563 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.85it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0205, Accuracy: 9930/10000 (99.30%)

loss=0.020957129076123238 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 36.31it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0194, Accuracy: 9941/10000 (99.41%)

loss=0.0024465073365718126 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 34.80it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0205, Accuracy: 9931/10000 (99.31%)

loss=0.00916708167642355 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 36.53it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0215, Accuracy: 9942/10000 (99.42%)

loss=0.061258912086486816 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 36.09it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0184, Accuracy: 9942/10000 (99.42%)

loss=0.007747257128357887 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.97it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0210, Accuracy: 9937/10000 (99.37%)

loss=0.02396484650671482 batch_id=468: 100%|██████████| 469/469 [00:13<00:00, 35.85it/s]

Test set: Average loss: 0.0184, Accuracy: 9951/10000 (99.51%)
```



##### Final Accuracy  - 99.51

> **99.4 Achieved at 14th Epoch**