Crater detection using Hypothesis Generation and CNN



 
Electrical Department
SRIP -2019


Rajnish Kumar Ranjan
Computer Science and Engineering, IIT (ISM) Dhanbad




Advisors: -
Prof. Vikrant Jain
Dr. Nitin Khanna






Acknowledgement
I would like to express the deepest appreciation to my project mentor Dr. Nitin Khanna and Dr. Vikrant Jain. They continually and convincingly conveyed a spirit of adventure in regard to the project, and an excitement in regard to the teaching. I would also like to thank Athira Haridas (MTech Student) and Atal Tewari (PhD Student) for helping me during this project. Without their persistent help this project would not have been possible.





Date: 14/07/2019								Rajnish Kumar Ranjan
IIT Gandhinagar











Abstract

Craters are surface characteristics, which are mostly generated after the impact of falling an asteroid over the planetary surface. Crater density on the lunar surface provide important information about the chronology of the lunar surface. Further, recent studies on crater morphology provided new insight about geomorphic processes in the absence of water and air. Crater detection is very crucial for the morphological study of the planetary system. The main challenges, this area has faced are:-  
(i) Generating a large number of training samples, which is essential for greater accuracy in the crater detection. 
(ii) Detecting a large number of craters in different sizes from large sized high quality surface imagery is highly complex in nature. 
Here, Highlight shadow algorithm used to generate all possible subsamples from the high quality imagery, which are to be verified using supervised algorithm such as CNN.
























CONTENT

	Introduction
	Related Works
	Methods
	Results and Discussions
	Conclusions
	References












	INTRODUCTION

Craters are the prime characteristics on the surface of the moon or the mars. Detecting a crater using imagery is very similar to detecting any other objects in computer vision. Crater can be either detected using supervised algorithm and/or unsupervised algorithm.
High quality satellite imagery are taken from high resolution cameras installed in planetary satellites, which are used for crater detection. Supervised algorithm takes a number of training samples to train itself. Convolutional Neural Network has been the main attraction for object detection in computer vision. Detecting multiple craters of different sizes from an image will require a very complicated CNN network [1]. Again for a complicated CNN, it will require a very large number of labelled datasets for training. Other way is two-step process. First step is creating samples using brute force technique (Sliding Window approach) and classifying them as crater or non-crater using Convolution Neural Network [2] in second step. We must find some method to detect crater in simpler way. Unsupervised algorithm might be useful to make the job of CNN easier.
Unsupervised algorithms are to detect objects without utilizing any training samples. It may find the patterns which is similar to craters to detect and locate the craters. Ortho images contains highlight and shadow regions at the places of craters due to illumination effects. A supervised algorithm ‘Highlight and shadow’ uses this pattern to detect craters from a satellite imagery.
Unsupervised algorithm and supervised algorithm both can be utilized together to enhance the performance of crater detection as explained in [3]. The Highlight and shadow algorithm creates hypothesis, which can be verified later using Convolution Neural Network[2]. 

	RELATED WORKS

Crater detection has seen many methods and algorithms. There has been many other unsupervised algorithm for crater detection like convex grouping, hough transform, interest points. The performance of different unsupervised algorithm is discussed in [3]. Convex grouping [4] is another unsupervised algorithm, which can be used to create samples for classifying them further. Haar-like features [5] can also be employed for crater detection, which has to be trained well. Yiran Wang and Bo Wu used Active machine learning method [6] to automate the training process by automating generation of large number of training samples through profiling using SVM. Four profiles (horizontal, vertical and diagonals) for each sample were checked and classified using Support Vector Machine. The positively and negatively classified samples were further used in training the Haar-cascade algorithm. Cohen and Joseph has analyzed the performance of Convolutional Neural Network [2] for the purpose of crater detection. 

	METHODS
The method consists of two phases:
(i) Image Sample Creation or Hypothesis Generation 

Hypothesis generation features highlight and shadow algorithm. Due to illumination, the patterns of highlight and shadow are formed near craters in the planetary images. This algorithm identify such patterns and generates a number of image samples, which are probable craters.
The algorithms is mentioned below:– 
	The input image I is negated to generate negative image N
N=O-I
Where, O is the image where each element is 1.
	Median filtering is applied on both I and N with large sized median filter, to generate MI and MN consecutively. It reduces the large sized features inside the image, which might be generated due to angle and location of the camera.
	To remove these features, subtract MI and MN from I and N respectively,
I=I-MI
N=N-MN
	The images I and N are binarized using thresholding.
	Connected components C_I and C_N are extracted from binarized I and N respectively.
	The enclosing bounding box 𝑏𝑖 for each c_i ∈ C_I and b_jfor each c_j∈ C_N are found, calling them  B_I and  B_N respectively.
	The pairs of highlight (b_i ∈  B_I ) and shadow (b_j ∈ B_N) regions are matched when:
distance (b_i,b_j) < 2×〖(max(area(b_i),area(b_j  )))〗^0.5
	The enclosing bounding box B_C are found, which enclose each paired regions.
	Overlapping detections are removed to improve the performance by the steps:
IOU among each of the detections are calculated and the two detections are merged if the IOU between them is more than a threshold.
Overlapping detection are removed by calculating IOU (Intersection over Union) between two detections and merging those two detections whose IOU is greater than a threshold.

(ii) Hypothesis Verification –
The generated samples from highlight shadow algorithm contains large number of non-crater samples that must be verified using a supervised algorithm. However, supervised algorithms like CNN and Haar-Cascade can be used for hypothesis verification. We are using a trained Convolutional Neural Network, which is same as the network discussed in [2].
 
Fig. 1 Convolutional Neural Network

Image samples of size 32x32 generated using highlight and shadow algorithm are input to a CNN having two convolutional layers and one fully connected layer.

	RESULTS AND DISCUSSIONS
Our goal is to improve the performance of Convolutional Neural Network by creating hypothesis using highlight and shadow algorithm.
 
Fig. 2 (a) Highlight and (b) shadow produced while running phase-I algorithm

The results are the labelled imagery to identify the craters in it. The crater detection technique is evaluated in the testing phase. The labelling of the images has been done in this paper by defining the bounding boxes and its’ boundary axes.
 
Fig. 3 Ground truth of LROC image, where detected craters are in boundary boxes

 
     Fig. 4 Detected craters from LROC image using highlight and shadow are in boundary boxes
The result is evaluated using measures like IOU, precision and recall
Recall and Precision:–

Recall=  TP/GP x100

Precision=  TP/(TP+FP) x100
Where, TP, FP, and GP are the number of true positives, false positives, and ground truth craters, respectively.
Since, the output generated from highlight and shadow are cross-checked by Convolutional Neural Network. The detected samples in first phase if found non-crater have a chance to be re-classified as non-crater later in verification phase. But, if a crater is not detected in first phase, it will have no chance to be classified as crater. Thus, the recall value for highlight and shadow algorithm must be very high.
Intersection over union (IOU): – 
"IOU(" B_i " ," B_j " )"=("Area(" B_i " ∩" B_j " )" )/("Area(" B_i " ∪ " B_j " )" )
Where B_i and B_J are two sample rectangular regions. B_i and B_J are clustered together if, "IOU(" B_i " ," B_j " )"  is greater than a predefined threshold.
After applying highlight and shadow algorithm, the result had given a recall of 85 % and precision of 48 %. The overall IOU between the ground truth and the output result is found to be 0.47.
The median filter size, we have found here giving best performance is 81x81. The threshold value TH has been chosen here in such a way that 90% of pixels belongs to grey values less than TH.
Using 2000 samples of each crater and non-crater image of size 32x32, CNN has been trained in 3 epochs. The recall and precision obtained by CNN has been 97% and 89% respectively.
The overall recall for the model generated using these two phase is 83%.

	CONCLUSION

Craters are important characteristics of the surface of Moon, Mars and other celestial bodies. 
	The two phase algorithm increases the performance of crater detection in comparison to one phase supervised algorithm.
	This approach has been implemented for the size of craters in range 10-100m diameter. The reason behind this limit is the lack of ground truth availability. It can be improved for the other size of craters.
 The code for highlight and shadow algorithm and CNN is located at https://github.com/Rajnish-Ranjan/Crater-Detection-using-Highligh-and-Shadow-algorithm. Some of the datasets and/or their links are also there.

	REFERENCES

[1]	H. Wang, J. Jiang, and G. Zhang, “CraterIDNet: An End-to-End Fully Convolutional Neural Network for Crater Detection and Identification in Remotely Sensed Planetary Images,” Remote Sens., vol. 10, no. 7, p. 1067, Jul. 2018.
[2]	J. P. Cohen, H. Z. Lo, T. Lu, and W. Ding, “Crater Detection via Convolutional Neural Networks,” ArXiv160100978 Cs, Jan. 2016.
[3]	E. Emami, T. Ahmad, G. Bebis, A. Nefian, and T. Fong, “Crater Detection Using Unsupervised Algorithms and Convolutional Neural Networks,” IEEE Trans. Geosci. Remote Sens., pp. 1–11, 2019.
[4]	D. W. Jacobs, “Robust and efficient detection of salient convex groups,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 18, no. 1, pp. 23–37, Jan. 1996.
[5]	R. Lienhart and J. Maydt, “An extended set of Haar-like features for rapid object detection,” in Proceedings. International Conference on Image Processing, 2002, vol. 1, pp. I–I.
[6]	Y. Wang and B. Wu, “Active Machine Learning Approach for Crater Detection From Planetary Imagery and Digital Elevation Models,” IEEE Trans. Geosci. Remote Sens., pp. 1–13, 2019.

