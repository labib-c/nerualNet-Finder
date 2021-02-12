# neuralNet Finder (or DeepNetFinder)
Detect outdoor basketball nets using satellite imagery fed through a Convolutional Neural Network

## Introduction

Basketball is an increasingly popular sport in Canada and around the world; casual games can be played at indoor courts but they are often expensive. Outdoor courts are usually discovered through word of mouth or online repositories which are far from comprehensive.

Our goal is to solve this problem by making a machine learning model that will detect the location of basketball courts using satellite imagery, with a false negative rate of less than 20% and a false positive rate of less than 5%. This is because in our case false negatives are less harmful than false positives.

![image](https://drive.google.com/uc?export=view&id=1isC35GoQNzo3JBeZzSSiPnlGfO7bFajq)

This project is interesting because the model can be applied to satellite imagery from anywhere in the world. This model and code could also be translated to use to other classification problems based on satellite images, such as detecting other sports venues or public spaces. 

Machine learning is sensible to use due to its ability to solve classification problems, with the ideal model being a You Only Live Once (YOLO) convolutional neural network (CNN) because of its proficiency with image analysis and object detection. 

## Background Related Work

There are existing online repositories of online basketball courts including on toronto.ca [1], Google Maps, and other commercial sites, however we know from personal experience these are not nearly complete.

DOTA (A Large-scale Dataset for Object DeTection in Aerial Images) was developed for object detection in aerial images, with continuing growth in size and scope. The images are annotated in 15 categories by experts in aerial images.[2]

There has been extensive research in the field of object detection, and one prominent result is the YOLO (You Only Look Once) architecture. YOLO is one of the most powerful and fastest object detection algorithms, which outperforms other detection algorithms such as R-CNN, due to requiring only one pass for an image through the network. YOLO learns very general representations of objects, is less likely to predict false positives in the background but results in a greater amount of localization errors [4]. 
We will be using YOLOv2 architecture, which provides various improvements on YOLOv1 which can also run at varying image sizes as it is a fully convolutional network [7].

## Architecture

![image](https://drive.google.com/uc?export=view&id=1X8l7_9PFiHu820F1GFApHVlgqJ0LnLtK)

We intend to train a model by passing custom classified images of basketball courts through the YOLOv2 architecture [7]. The YOLO CNN will perform object detection and draw bounding boxes around the regions of the image it judges to have a probability over some threshold of being a basketball court, which we can then apply to satellite imagery.

![image](https://drive.google.com/uc?export=view&id=1wPMJsJjwejix-7LmEEbaRw4NcW6M31Ak)

## Data Processing

We will be using two datasets consisting of satellite imagery for training object detection models. One is the DOTA dataset [1], which has 2806 images consisting of 15 classes of object labels, including basketball courts. The second dataset we will use is the VHR-10 dataset [5][6], which consists of 715 satellite images consisting of 10 classes including basketball courts. The VHR-10 dataset contains 159 instances of basketball courts.

We will only use the images with basketball courts in them from both of these datasets to train the model. We plan on using the YOLOv2 architecture, which is fully convolutional, so it is not required to resize the images to a unique size. However, the DOTA images range in size from 800x800 to 4000x4000, and training on these large images may take a while. If we see that the training takes an unreasonable amount of time, we will proportionally resize the images and corresponding bounding boxes to make them smaller.

The amount of training data available is somewhat limited. Data augmentation will thus be important for this project. We will augment the data through resizing, adding some noise (such as Gaussian noise), and random crops to the images to produce new data we can use for training. 

We can also normalize the data by subtracting the mean across each color channel, with the hope of improving performance.

## Baseline Model

We can use baseline results posted on the DOTA GitHub page [3]. These model results reflect the performance of several different convolutional neural network-based architectures for object detection, including Faster R-CNN and RetinaNet. Note that there are no results for YOLOv2 in these benchmarks. We want to see how YOLOv2 performs in comparison to the other models in terms of speed and mean average precision, which is a metric used to compare the accuracy of the predicted bounding boxes relative to the ground truth ones.

## Ethical Considerations

There are privacy concerns because our model would very likely find and expose courts that are private and it would be an unreasonable amount of work to hand filter those out. We also must keep in mind that our training data is biased towards courts that have crisp, clear features and our model will likely follow suit. This may cause our model to find more courts in higher class neighborhoods, a phenomenon we will stay cognizant of and do our best to remedy.

## References
[1] City of Toronto, “Parks Map,” City of Toronto, 06-Mar-2017. [Online]. Available: https://www.toronto.ca/data/parks/maps/index.html#BASKETBALLCOURT. [Accessed: 9-Feb-2021].


[2] G.-S. Xia, X. Bai, J. Ding, Z. Zhu, S. Belongie, J. Luo, M. Datcu, M. Pelillo, and L. Zhang, “DOTA: A Large-scale Dataset for Object Detection in Aerial Images,” arXiv.org, 19-May-2019. [Online]. Available: https://arxiv.org/abs/1711.10398. [Accessed: 10-Feb-2021].


[3] “Benchmarks for Object Detection in Aerial Images,” https://github.com/dingjiansw101/AerialDetection, 08-Dec-2020. [Accessed: 11-Feb-2021].

[4]  J. Redmon, S. Divvala, R. Girshick , and A. Farhad, “You Only Look Once: Unified, Real-Time Object Detection,” arXiv.org, 09-May-2016. Available: https://arxiv.org/pdf/1506.02640v5.pdf [Accessed: 10-Feb-2021]
[5] Su H, Wei S, Yan M, et al. Object Detection and Instance Segmentation in Remote Sensing Imagery Based on Precise Mask R-CNN[C]. IGARSS 2019-2019 IEEE International Geoscience and Remote Sensing Symposium. IEEE, 2019: 1454-1457. [Accessed: 10-Feb-2021].
[6] Su, H.; Wei, S.; Liu, S.; Liang, J.; Wang, C.; Shi, J.; Zhang, X. HQ-ISNet: High-Quality Instance Segmentation for Remote Sensing Imagery. Remote Sens. 2020, 12, 989. [Accessed: 11-Feb-2021].
[7] J. Redmon and A. Farhadi, “YOLO9000: Better, Faster, Stronger,” 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

[8] Derakhshani, Mohammad & Masoudnia, Saeed & Shaker, Amir & Mersa, Omid & Sadeghi, Mohammad & Rastegari, Mohammad & Araabi, Babak. (2019). “Assisted Excitation of Activations: A Learning Technique to Improve Object Detectors”. 10.1109/CVPR.2019.00942. Available: https://www.researchgate.net/publication/333609719_Assisted_Excitation_of_Activations_A_Learning_Technique_to_Improve_Object_Detectors#pf5 [Accessed: 11-Feb-2021]
