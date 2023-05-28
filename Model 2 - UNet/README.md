This folder contains our code for the second part of submission which is Segmentation of Images.
For segmentation task we UNET model, where we use 4 encoding layers and 4 decoding layers.

## Description

### Task Description

In this section, our objective is to train the U-Net model on train set images provides and evaluate
the model's performance on validation set.

### Model Description
U-Net : CNN based model for semantic segmentation of objects in 22nd Frame.
- 4 encoding layers and 4 decoding layers 
- 1 bottle neck layer


- trained on 22 x 1000 images of training set
- trained for 10, 20 and 50 epochs. 
- learning rate : 1e-4
- optimiser : Adam
- loss : Cross Entropy
- evaluated on 1000 images of validation set.
- evaluation metric : Jaccard index


### Dataset description

Dataset used in this section is:

• 1,000 labeled training videos with 22 frames each,
Each video folder has masks.npy which contains masks of 22 all 22 frames in the video.

Thus, we inflated all the frames and their respective masks in the "SegmentationDataSet" class, and shuffled all the images.
Now total 22 X 1000 images, i.e. 22000 frames(images) are used to train model.

• 1,000 labeled validation videos with 22 frames each, this is used for validating the images performance.

### Steps to Reproduce

1) In `main` function of `main.py`, specify the path of train set and validation set.

`python main.py`

2) After the running the main.py, it will save trained U-Net model as `unet.py`.
3) Now to check the performance of trained unet model, on validation set, run the `Checking_Jaccard_index.ipynb` file.
4) In the 3rd cell of .ipynb file specify the path of "UNET trained" model `unet.pt` and "validation set".
5) Run all the cells of the `Checking_Jaccard_index.ipynb` file, and final jaccard score on validation set can be derived.