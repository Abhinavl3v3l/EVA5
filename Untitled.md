# Session3



- Convolving on an image of n * n by a 3 * 3 kernel reduces the size by 2 * 2

$$
Formula 
$$

- Output channel size. 
  $$
  O_c =  \frac{I_n + 2p -k}{S} + 1
  $$
  Where 

$O_c$ - Output Channel Size

$I_n$ - Input

$p $   -  Padding

$k$   - Kernel

$S$   - Stride







Strides

Stride are bad as it ends up blurring the image

- Strides are good for low end machine where we can compromise accuracy

- Strides are bad where accuracy needed is near to perfection



Kernels - 1 x 1

3x3 - Good feature Extractor

1x1 - Good feature combiner (2014 -2015) Richer Kernels

-  `|` and `-` to  `+`