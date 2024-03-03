# Model Card - CNN Image Classification - Using Intel Image Classification Dataset (Using FastAI/ResNet)

## Model Description

**Input:**
 This input data contains around 25k images of size 150x150 distributed under 6 categories: 'buildings', 'forest', 'glacier', 'mountain', 'sea' and 'street'. There are approximately 14k images in Train, 3k in Test and 7k in Prediction. This data was initially published on https://datahack.analyticsvidhya.com by Intel to host a Image classification Challenge. 
The model accepts images of natural scenes, including categories such as buildings, forests, glaciers, mountains, seas, and streets. Images are resized from 150x150 pixels to 224x224 pixels to conform with the standard used for the ResNet Architecture.

**Output:** 
Categorical predictions across six classes corresponding to the scene types: buildings, forests, glaciers, mountains, seas, and streets.

**Model Architecture:** 
Utilises the ResNet-18 architecture, a convolutional neural network pre-trained on the ImageNet dataset, chosen for its balance between performance and efficiency. ResNet, like many other Convolutional Neural Networks (CNNs), was originally designed and trained on the ImageNet dataset. ImageNet is a large visual database designed for use in visual object recognition software research. More than 14 million images have been hand-annotated to indicate what objects are pictured and in at least one million of the images, bounding boxes are also provided. The images used to train ResNet and similar models in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) are typically resized to 224x224 pixels.

The model is centred around and makes use of the FastAI libraries. FastAI is a deep learning library which provides practitioners with high-level components that can quickly and easily provide state-of-the-art results in standard deep learning domains, and provides researchers with low-level components that can be mixed and matched to build new approaches. It aims to do both things without substantial compromises in ease of use, flexibility, or performance. This is possible thanks to a carefully layered architecture, which expresses common underlying patterns of many deep learning and data processing techniques in terms of decoupled abstractions. These abstractions can be expressed concisely and clearly by leveraging the dynamism of the underlying Python language and the flexibility of the PyTorch library. 

## Performance

The model demonstrates high accuracy in classifying the Intel Image Classification dataset. Key performance metrics include accuracy, precision, and recall scores, indicating the model's effectiveness in identifying correct scene categories.

Validation Loss (0.2200719118118286): This is a measure of how well the model is performing on a separate dataset that was not used during training, known as the validation set. The loss typically represents the model's error - lower values are better.

Accuracy (0.9365645051002502): the ratio of the correctly predicted observations to the total observations. An accuracy of 0.9365 means that approximately 93.65% of the model's predictions are correct.

Precision Score (0.9371265308790853): The ratio of correctly predicted positive observations to the total predicted positive observations. It is a measure of a classifier's exactness. High precision indicates a low rate of false-positive predictions. A precision score of 0.9371 means that when the model predicts a positive class, it is correct about 93.71% of the time.

Recall Score (0.9383282712981598): Recall (also known as sensitivity or true positive rate) is the ratio of correctly predicted positive observations to all observations that are actually positive. It is a measure of a classifier's completeness. High recall indicates that the class is correctly recognized (a low number of false negatives). A recall score of 0.9383 means that the model correctly identified about 93.83% of the actual positive cases.

Overall the model can achieve close to 94% accuracy for this dataset. For comparison purposes, an excellent result for SOTA (State of the Art) models on the ImageNet dataset (subtantially larger dataset) is anything above 90%. 

## Limitations

The model's performance may vary with images not represented in the training dataset, potentially leading to reduced accuracy on different or more complex natural scene datasets. The Intel classification dataset contains a small percentage of noise - images that are not as described by the label (e.g., a band playing music on a stage). The likely purpose is noise injection: Adding noise to inputs or weights during training can improve robustness and generalization by preventing the model from memorizing the training data too closely. In addition, there are images that could be fairly labelled sea or mountain; this may be a case of label Smoothing which softens the target labels, preventing the model from becoming too confident about its predictions, which can lead to improved generalisation.

## Trade-offs

Choosing ResNet-18 provides a balance between computational efficiency and predictive performance. However, this comes at the cost of potentially overlooking finer details that more complex models might capture, affecting performance on more challenging or diverse datasets.

## Appendix A - ResNet Model Description (ResNet18)
The provided CNN architecture summary describes a Convolutional Neural Network (CNN) structured in a sequential manner, typical of image recognition tasks. This network processes input images of shape 64 x 3 x 224 x 224, where 64 is the batch size, 3 is the number of colour channels (RGB), and 224 x 224 is the resolution of each image. Here's a summary of how this CNN works, layer by layer:

1.	Initial Convolution and Activation:
•	A convolutional layer (Conv2d) with ReLU activation processes the input, reducing the spatial dimensions to 112 x 112 while increasing the depth to 64 feature maps. Batch normalization (BatchNorm2d) is applied, making training more stable and faster.

2.	Pooling and Intermediate Convolution Blocks:
•	A MaxPool2d layer reduces the spatial dimensions by half (to 56 x 56), decreasing the amount of computation required for subsequent layers.
•	Multiple Conv2d layers and ReLU activations continue processing the data, with batch normalization applied after each convolution. These layers progressively reduce the spatial dimensions (to 28 x 28, then 14 x 14, and finally 7 x 7) while increasing the depth (number of feature maps) through the network (64, 128, 256, then 512). This process extracts increasingly complex and abstract features from the input images.

3.	Adaptive Pooling:
•	AdaptiveAvgPool2d and AdaptiveMaxPool2d layers reduce each 512-channel feature map to a 1 x 1 area, effectively summarizing the features extracted from each part of the image into a single number per channel. This step prepares the network for classification.

4.	Flattening and Fully Connected Layers:
•	The output from the pooling layers is flattened into a vector and passed through a series of fully connected (Linear) layers and ReLU activations, with dropout applied to prevent overfitting. Batch normalization is used after flattening and each fully connected layer to further stabilize and speed up training.

5.	Output Layer:
•	The final fully connected layer reduces the dimensionality to match the number of classes in the dataset (in this case, 6), producing the logits for each class.

6.	Total Parameters:
•	The model has a total of 11,706,944 parameters, of which 540,032 are trainable. The rest are frozen, indicating that their weights will not be updated during training. This is a common practice when using transfer learning, where pre-trained weights are used as a starting point for feature extraction.

7.	Optimizer and Loss Function:
•	The model uses the Adam optimizer and CrossEntropyLoss for training, a common choice for multi-class classification problems.

8.	Model Freezing:
•	The model is frozen up to parameter group #2, meaning that only the parameters in the final layers (after group #2) are trainable. This is typical in transfer learning scenarios to fine-tune the model on a new dataset while preserving the feature extraction capabilities learned on a large, comprehensive dataset (like ImageNet).

9.	Callbacks:
•	Various callbacks are used for monitoring training progress, handling tensor casting, recording metrics, and more.
This architecture exemplifies a deep CNN tailored for image classification, leveraging pre-trained layers for feature extraction and additional trainable layers for task-specific fine-tuning.
