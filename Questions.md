1. Edges and Gradients are made or seen with at least a receptive field of **11 x 11** for an image of 200x 200.

   1. Does this change with change in input size ?

   **Answer** - Depends on Dataset

2. Why do we need layers, and why not just train a very big kernel that will convolve and predict the          object ? What limitations are causing us to have many many layers and  not just one ? Why do we need many-many-many-many layers? effects of many layers

3. When do we use first max pool ? After making edges and gradient i.e.11x 11 ? 

   1. Similarly do we use next max pool ? when we have enough layers and kernels to form textures and so on patterns ?

   **Answer** - Yes it depends on dataset when edges and gradients are formed that's when we apply transformations.

4. Why add max pool ? 

   Answer - to reduce the number of layers , to overcome hardware restriction also calculation reduction

5. 1x1 is used for mixing channel say 512 to say 32 rich kernels which extracts multiple features ?

   1. 1x1 is also used to filter out features not required 

6. Activation function and why we need non linearity in our network. How does it helps ?

   Activation function add non linearity to network, 