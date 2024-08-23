## Iris Flower CV Classifier

### Summary
This is a small project where I created the class for a classifier object intended to classify species of iris flowers (setosa, versicolour or virginica).

**init** is a constructor that initializes key model components for training. There are also methods for training and classification tasks.
**preprocess_train** uses a PyTorch NN to finetune model, then saves the model weights to a directory to be used after. Training loss is reported per epoch. Method is void.
**file_classify** reads and formats an image using OpenCV and Torchvision. Returns the predicted species as a string.

### Extra Information
Highly recommend downloading the Iris CV dataset @ https://www.kaggle.com/datasets/jeffheaton/iris-computer-vision for training
