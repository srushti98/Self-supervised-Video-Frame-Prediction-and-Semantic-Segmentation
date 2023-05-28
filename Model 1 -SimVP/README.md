This folder pertains to our code for the implementation of SimVP: Simpler Yet Better Video Prediction, and was inspired from this [paper](https://arxiv.org/pdf/2206.05099.pdf) and [GitHub](https://github.com/gaozhangyang/SimVP-Simpler-yet-Better-Video-Prediction) repository. 

## Description

### Task

In this section of the project, our objective is to predict the future frames given the past frames of a video clip of moving objects, that interact with each other on the basic principles of physics. 

### Dataset

The dataset consists of 13000 video clips, each video provided in the form of 22 frames. The video essentially consists of 3D moving objects. Objects in videos have three shapes (cube, sphere, and cylinder), two materials (metal and rubber), and eight colors (gray, red, blue, green, brown, cyan, purple, and yellow). In each video, there is no identical objects, such that each combination of the three attributes uniquely identifies one object. 


## Our Approach 

Whilst initally trying RNN-based architectures such as ConvLSTM + GANs, and then more advanced architectures that utilize self-supervised learning such as masked autoencoder for this task, we pivoted and found the best performance with SimVP. This utilizes a fully CNN-based architecture that is trained on minimizing the mean squared error loss objective. 
We trained the SimVP model for 25 epochs on the 13000 clips at a learning rate of 0.001. 
We mainitained a batch size of 4 after experimenting with different batch sizes and learning rates for hyperparameter tuning. 
We used `torch.nn.DataParallel` module for speedup, but it is optional and that line can be commented out.

### Model

The SimVP model uses an encoder, a translator and decoder module, all of which are convolutional neural networks.
The encoder module extracts spatial features from the input frames, the translator learns evolution of the frames across time, and the decoder integrates
spatio-temporal information to predict future frames.


- Encoder: The Encoder stacks 4 blocks of Conv2D, Layer Normalization and LeakyReLU to obtain the hidden features, with input and output shapes to the encoder being num_past_frames × num_channels × height × width, which for our dataset corresponds to 11 × 3 × 160 × 240. 

- Translator: The translator module uses 8 Inception modules (called Mid_XNet) of different kernel sizes (3,5,7,11). The Inception module consists of a bottleneck Conv2d with 1×1 kernel followed by parallel GroupConv2d operators.

- Decoder: The decoder has 4 blocks of ConvTranspose2d, Group Normalization and LeakyReLU to reconstruct the ground truth frames, which convolutes C channels on the image dimensions 160 × 240. The design of this module is very similar to the encoder module with the diffrence of using ConvTanspose2D for up-convolution operation, required for reconstructing an image from the hidden representation given by the translator module.


SimVP is trained on standard MSE loss and we worked on experimenting with different hyperparameters to minimize it.


### Steps To Reproduce

In this directory, run the `main.py` file using
```
$ python main.py --gpu '0'
```
You may add additional flags with the above command for more GPU utilization with `--gpu='0,1'` for 2 GPUs or `--gpu='0,1,2,3'`.
Provide the directory for results with `--res_dir=<path/to/folder>` and dataset directory of the unlabeled moving objects dataset with `--data_root=<path/to/folder>`.

Additional flags for number of epochs, learning, batch size and hidden dimensions can be provided from command line as well (refer to main.py for all possible arguments). 

The model after training is used to predict frames on the 2000 videos on the hidden dataset, which contains only the first 11 frames. 
The future 11 frames are predicted, and the final frame is then fed to our trained segmentation model for mask prediction.
