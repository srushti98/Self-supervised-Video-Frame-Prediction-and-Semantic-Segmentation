For more Details refer to the [Project Report.](https://drive.google.com/file/d/1Qwrr-EDlx-IGeudGE8XTHuLczqvsnfpP/view?usp=sharing) 

This is Video Frame Prediction and Semantic Segmentation project:

### Project Description

#### Problem Statement:
To predict 22nd frame from given 11 frames of a video and perform semantic segmentation on the 22nd image. 

#### Folders Description

##### Model 1 - SimVP

- This folder contains code for first part of the project which is Future frame prediction.
- Refer the `README.md` inside this folder to execute the code.

#### Model 2 - U-Net

- This folder contains code for second part of the project which is Semantic segmentation and masks generation.
- Refer the `README.md` inside this folder to execute the code.

#### Final Pipeline

- This folder contains code for third part of the project which is Pipeline which takes the saved model of part 1 and part 2 and hidden datset and generates the required final output numpy array.
- Refer the `README.md` inside this folder to execute the code.


### Sequence to Reproduce

1) Run code in `Model 1-SimVP`
2) Run code in `Model 2-UNet`
3) Use the trained models generated in `part 1` and `part 2` along with the `hidden dataset` to work with `Final Pipeline`
