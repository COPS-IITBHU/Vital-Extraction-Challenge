<h1 align="center">Report</h1>

This notebook presents the code for our Approach in tackling the problem statement given to us by CloudPhysician. Below is a figure depicting our pipleine from a higher level

<!-- Figure here -->


<h2 align="center">Pipeline Componets with Training and Inference settings </h2>

<h3> 1. MONITOR SEGMENTATION</h3>

<h4>- Approch</h4>

Our first task is to extract the monitor from a complete image, we have used segmentation based unet architecture as our backbone and initialized it with IMAGENET weights. Choice of using Segmentation is due to the fact that the monitors postions are not always rectangular so detection will not work that good instead pixel level work will do the thing. **we had 7000 unlabelled data, so we decided to combine it with 2000 labelled data and use a Semi-Supervised(SS) Semantic Segmentation Approach(Inspired from this paper [LINK](https://arxiv.org/pdf/1904.12848.pdf))**, their work were based on Image classification, we have modified their working for semantic segmentation of our case, Additionally, our novel additions like LR scheduler, augmentations variations(randAugment -  a type of novel augmentation technique - [LINK](https://medium.datadriveninvestor.com/why-randaugment-is-the-best-data-augmentation-approach-4a48f22b2152)) boost the quality of segmentation mask compared to supervised approach as it able to generalize well on cross-domain data. 

Below are some images which shows the better mask generation of Semi-Supervised Approach in most scenarios where Supervised alone method is not working well.

<!-- Figures of Comparison-->

<h4>- Training Setup</h4>
In this section, we will present the training setup and parameters for our Semi-Supervised Segmentation.

We used a loss function as,  

$L = L_{s} + \lambda L_{u}$ 

whre **$\lambda$** is a hyperparameter adjusting both losses, $L_{s}$ is binary cross entropy supervised loss and $L_{u}$ is unsupervised consistency loss, we used KL divergence as a consistency loss, it helps in matching the mask similarity. Below is an image for Training procedure of SS Method.

<!-- <Figure> -->


*HYPERPARAMETERS:-*

```
LEARNING_RATE = 3e-5
BATCH_SIZE = 6
UNSUP_BATCH_SIZE = 16
NUM_EPOCHS = 25
IMAGE_HEIGHT = 320
IMAGE_WIDTH = 640
LAMBDA = 0.3
DECAY_FACTOR = 0.9
```
It is recommended to use a bigger batch size for unsupervised image compared to supervised, we used a low learning rate to allow model to converge slowly keeping info of imagenet params, used loss factor lambda as 0.3, and learning rate decay factor as 0.3. We used image shape almost half compared to original size for segmentation. We done this and then upsampled the mask to original size because the monitor ha a nice geometrical shape which allowed easy resizing of mask.
We saved the model based on mIOU (mean Intersection over union score) on validation. Set 1800 labelled to train and 200 labelled to validation.

Plot for MIOU score is below:- 
<!-- UDA UNET miou figure -->
<p align="center">
<img height = "300" width = "500" src ="https://user-images.githubusercontent.com/60649720/216918754-f9f96b74-0535-491f-bcf3-7413d1e748fd.png" />
</p>


<h4>- Inference Setup</h4>
Inference section is provided in the notebook itself, we just have to perform preprocessing of normalization and then take the inference on cpu. It takes rougly 0.4-0.6 seconds for each inference on cpu.


<h3> 2. PRESPECTIVE TRANSFORMATION</h3>

<h4>- Approch</h4>

After getting the mask we have to extract the monitor using that mask from input image and then aligning the monitor in birds eye view, we used convex hull and contours methods for finding mask corners and then used opencv perspective transform for wrapped perception, and we output the wrapped image in two resolution, one used for ocr and other used for classification. Code for it is present in the notebook itself. It's inference time is between 0.02-0.08 second.

Below present an output from perspective transform

<h4> Novel thing  - Choice of Size Selection</h4>

One thing before moving forward is to see how we decided the optimum size for ocr and classification.

 - We generated masks for all unlabelled data and then found their corners and then 
 we stored the two dimension values of each masks in an array
 - then we draw a histogram of those dimension to see the volume covered by monitor on an average in unlabelled data
 - Below present the histogram of both data, based on this we decided to take the mean value of that histograms i.e. (360, 640) as (h, w) for classification
 - For ocr we switchhed between above dimension and (180, 320) based on the layout which was provided by classification.

<!-- HISTOGRAM -->



<h3> 3. Monitor Layout Classification</h3>

<h4>- Approch</h4>

Identifying the layout for further pipeline is very essential because no matter how many types of monitors you have if you fix the number of classes then the model tries to assign the input monitor to the best possible class from the available class, we used 4 given layout classes for classification, any input image which resembles closely to any class will be assigned that class and assumed it's layout to be the same as that class.

We used resnet18 model for classification(it is light and gave around 99% validation accuracy),fine tuned it on our data.We used 80:20 ratio for train-val split(Stratified)

<h4>- Training Setup</h4>
In this section, we will present the fine-tuning training setup and parameters for our Resnet-18.

We used cross-entropy loss function with stepLR scheduler of pytorch.
Below are the training results.

<!-- <Figure> -->


*HYPERPARAMETERS:-*

```
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 50
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 640
```


<h4>- Inference Setup</h4>
Inference section is provided in the notebook itself, we just have to perform preprocessing of normalization and then take the inference on cpu. It takes rougly 0.1-0.2 seconds for each inference on cpu.


<h3> 4. OCR</h3>

<h4>- Approch</h4>

We used paddleocr, which is a fast, lightweight and open source detector+ocr model based on CRNN (Convolutional Recurrent Neural Network), we used it's fastest and recent version of PPOCR-v3, which runs considerably faster on cpu which achieving a good recognition accuracy.


We used the input resolution dependent on the layout classification, FOr example layout which seems too crowded, we provided higher resolution of (360,640) and layout which have values apart we set their input ocr resolution to be (180, 320), our this trick helped in utilising layout information for achieving higher accuracy in extracting vitals

Inference code for this model is one liner simple and provided in the notebook along with it's pip installation. It takes rougly 0.5-1 seconds for each inference on cpu. Higher time like 1 second usually accounts when we use layout of size (360, 640) otherwise till is generally less, and it also depends on the number of boxes detected by paddleocr, sometimes it detects more boxes so time goes on higher side like 1 seconds but generally it takes around 0.6-0.7 seconds in our pipeline.

<!-- <May be figure> -->


<h3> 4. Rules Checking</h3>

<h4>- Approch</h4>

We used the input resolution dependent on the layout classification, For example layout which seems too crowded, we provided higher resolution of (360,640) and layout which have values apart we set their input ocr resolution to be (180, 320), our this trick helped in utilising layout information for achieving higher accuracy in extracting vitals

Inference code for this model is one liner simple and provided in the notebook along with it's pip installation. It takes rougly 0.5-1 seconds for each inference on cpu. Higher time like 1 second usually accounts when we use layout of size (360, 640) otherwise till is generally less, and it also depends on the number of boxes detected by paddleocr, sometimes it detects more boxes so time goes on higher side like 1 seconds but generally it takes around 0.6-0.7 seconds in our pipeline.

For extracting vitals, we have used rule based method to rule out the boxes which are not required. We decided to make a class based approach where the vitals either have a specific location or value or color to a particular class. For example: `HR value` is obtained if size of bounding box is above a threshold value and is of green color as most of the classes have HR in green color. Similarly `SPO2` and `RR` is obtained if the text color is either cyan or yellow. Text color is determined by placing a mask of selected range of colors and then the number of pixels per area is calculated if the calculated ratio is above a threshold, we call the text color is of that color. `DBP` and `SBP` values are obtained in a different fashion. In every layout there is a `/` between the two values and we check if any bounding box contains that character. Usually two digit value to the right of `/` is `BBP` and two or three digit value to the left of `/` is `DBP`. 

<!-- <May be figure> -->


<h3> Vitals Detection</h3>

<h4>- Approch</h4>

Now, the next step is to detect the locations of the required vitals (HR, SBP, DBP, MAP, RR, HR_W) from the segmented and perspective corrected monitor. The bounding boxes for the all these vitals were given in the training dataset. Our first baseline approach was to take the union of all the bounding boxes for a particular class, then apply some rules. That was fast but the performance was not very good. So, we decided to use a deep learning model which gives bounding fast for multiple classes, is fast and can be trained on the given dataset. We tried a few transformer based models like CRAFT, FAST and DeepText-DETR. They gave good bounding but were too slow on GPU.
So, finally to find the bounding boxes, we used **YOLOv5nano**. It is lightweight, has faster inference time and can be trained on our training dataset.

<!-- YOLO models comparison figures -->
Comparison of different YOLOv5 models:
<p align="center">
<img height = "300" width = "500" src ="yolo_model_plot.png" />
</p>


<!-- Figures of Comparison-->

<h4>- Training Setup</h4>
In this section, we will present the training setup and parameters for our YOLOv5nano.

We used a loss function as,  

```
Loss = w_o * Loss_o + w_c * Loss_c + w_l * Loss_l + w_s * Loss_s
```

where:

- Loss_o is the objectness loss,
- Loss_c is the classification loss,
- Loss_l is the localization loss,
- Loss_s is the size loss,
- 'w_o', 'w_c', 'w_l', 'w_s' are the weight coefficients that determine the relative importance of each loss.


<!-- <Figure> -->


*HYPERPARAMETERS:-*

```
LEARNING_RATES:
    - 'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
    - 'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0*lrf)
BATCH_SIZE = 16
MOMENTUM : (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
WEIGHT_DECAY : (1, 0.0, 0.001),  # optimizer weight decay
NUM_EPOCHS = 540
IMAGE_HEIGHT = 320
IMAGE_WIDTH = 640
'box': (1, 0.02, 0.2),  # box loss gain
'cls': (1, 0.2, 4.0),  # cls loss gain
'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
'iou_t': (0, 0.1, 0.7),  # IoU training threshold
'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
```


<h4>- Inference Setup</h4>
Inference section is provided in the notebook itself. Preprocessing and normalization are done by the model itself. It takes rougly 50ms-100ms for each inference on cpu.


