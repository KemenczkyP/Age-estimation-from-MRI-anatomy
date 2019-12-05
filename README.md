# Age-estimation-from-MRI-anatomy

Python scripts for MSc Thesis [\*].

[\*] P. Kemenczky, Brain age prediction based on convolutional neural networks trained on T1 weighted MRI volumes, 2019

main_preprocess.py generates TFRecords files for different 3D brain MRI databases. It uses scripts in directory "preprocessing".

main_ml.py trains a convolutional neural network (structures in networks directory) for brain age prediction. It also uses the training_class.py script in "machine_learning" directory.

Requirements:
	Windows 10.
	Python 3.7
	Tensorflow 1.13

In directory “RAM_tf2” the scripts realize Regression Activation Mapping (RAM) with Tensorflow 2 package. 
main_train.py trains the neural network (“RAM_tf2//networks”) for brain age prediction. main_RAM_retrain.py uses the changed models to realize RAM by training a new output layer. It also saves the generated activation heatmaps.
heatmap_visualization.py creates an animation from the heatmaps.
Requirements:
	Windows 10.
	Python 3.7
	Tensorflow 2.0

