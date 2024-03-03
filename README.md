# CNN Image Classification - Using Intel Image Classification Dataset


## NON-TECHNICAL EXPLANATION OF YOUR PROJECT
This project aims to demonstrate the process of training an image classification model utilising the Intel Image Classification dataset. Leveraging the power of FastAI libraries and the pre-trained ResNet18 architecture, the project simplifies the complex task of image classification, making it accessible for educational purposes and further adaptation.

## DATA
The project uses the Intel Image Classification dataset, known for its collection of natural scene images across multiple categories such as buildings, forests, and glaciers. The dataset is publicly available on Kaggle: https://www.kaggle.com/puneet6060/intel-image-classification.

## MODEL 
The choice of model for this project is ResNet18, a variant of the ResNet Convolutional Neural Network (CNN) architecture, known for its efficiency and depth. ResNet18 is utilised here for its balance between performance and computational efficiency, making it suitable for the scope of this project.

## HYPERPARAMETER OPTIMSATION
Hyperparameters were optimised through an iterative process, leveraging the FastAI Learning Rate Finder to identify optimal learning ratespre and post model unfreezing. This enabled fine-tuning of the model with learning rates that encourage stable and effective learning.
For the project's hyperparameter optimization, I adopted the FastAI lr_find method, a strategic approach to determine the optimal learning rate range. This innovative tool plots the loss against varying learning rates to identify where the loss most significantly decreases, offering a visual cue for the ideal learning rate "sweet spot." I subsequently, applied 'one-cycle' training, an advanced method that dynamically adjusts the learning rate within the discovered range throughout different training phases. Initially starting with a lower learning rate, it escalates to the peak value suggested by lr_find before tapering off again. This nuanced approach helps accelerate convergence but also avoids overfitting by ensuring the learning rate is neither too aggressive at the outset nor too conservative towards the end of training. This methodological blend of identifying a pivotal learning rate followed by its optimised application via one-cycle training encapsulates our comprehensive strategy for hyperparameter tuning, designed to enhance model performance efficiently.

## RESULTS
The model achieved commendable accuracy, with precision and recall metrics indicating strong performance across the various classes of the dataset. This underscores the model's ability to generalise well, making correct predictions on a majority of the test data.

## (OPTIONAL: CONTACT DETAILS)
For further inquiries or collaboration, feel free to reach out via email: giantatlas@outlook.com



