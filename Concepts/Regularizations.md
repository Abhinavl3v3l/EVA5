# Regularizations

1. L1 and L2 Loss
2. Dropouts
3. Data Augmentation.
4. Early Stopping
5. Batch Normalization
6. Image Normalization
7. Cutouts.













### DropOut (Later Cutouts)

Philoshopy : The object behind dropouts is the make sure network identifies an object even when a part of object is seen.

For example:  If $n$ nodes are responsible for identifying an object, we need to make  network can identify that object even when $n-1$ or $n-2$ nodes  available. This is true  for fully connected layers.



DropOut(0.2) drop 20% pixels from a channel over which a kernel will convolve.  It could be possible that those pixels are dropped which are important. 

DropOut dont really workout for  CNN, Cutouts does.

![](C:\Users\level\Documents\GitHub\EVA4\Concepts\Images\whywedontneedcutouts.png)

Here say we drop a pixel in layer number 5. We have not seen 9x9 from first layer. But all the blue 1x1 in 5th layer have seesome parts of that 9x9. So even if we loose data we are not loosing or putting network in handicap. If it would be possible to replace 9x9 in first layer, it would increase the validation accuracy higher by putting a stricter handicap to make it learn better. Hence why Cutouts are prefered in CNN.









### Batch  Normalization



Fix the data so that they are similar in nature. Normalization distributes values between.

![](C:\Users\level\Documents\GitHub\EVA4\Concepts\Images\normalized.png)

Normalization vs Equalization 

BN solves Internal covariant shift. 