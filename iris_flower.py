from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import cv2

class IrisImageClassifierModel:

    '''
    I created a new classifier object that contains a classifier object for the sake of the new classifier already being finetuned on a custom dataset so that it can be used
    in your main/driver file. Hypothetically, you could still recreate this without creating a new classifier constructor.
    '''

    def __init__ (self):

        model_name = 'google/vit-base-patch16-224-in21k'
        self.model = ViTForImageClassification.from_pretrained(model_name)
        self.config = ViTConfig.from_pretrained(model_name, num_labels = 3)
        self.extractor = ViTImageProcessor.from_pretrained(model_name)
        self.model.classifier = torch.nn.Linear(self.model.config.hidden_size, 3)

        self.batch_size = 64
        self.epochs = 10

        self.model.eval()

    def preprocess_train(self):

        '''
        For the dataset path, make sure you download the Kaggle Iris CV at https://www.kaggle.com/datasets/jeffheaton/iris-computer-vision or another adequate dataset. 
        Use the file location path and replace it for "enter path here".
        '''

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        path = 'enter path here'
        training_dataset = datasets.ImageFolder(root=path, transform=transform)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        train_loader = DataLoader(training_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self.model(images).logits
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # PROGRESS BARS

            #print(f"Epoch {epoch+1}/{self.epochs} with a Loss: {running_loss/len(train_loader)}")
            print("Epoch: " + str(epoch+1) + "/" + str(self.epochs))
            print("Loss: " + str(running_loss/len(train_loader)))

        torch.save(self.model.state_dict(), 'enter dir u wish to save ur model at')

    def file_classify(self, path):

        # Returns an int 0-2
        # 0 - Setosa
        # 1 - Versicolour
        # 2 - Virginica
        
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        assert img is not None, "Path is inaccessible. Try backslashing or os.path.exists()"
        img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img_tensor = img_transform(img)
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            _, predicted = torch.max(self.model(img_tensor).logits, 1)

        classes = ['iris-setosa', 'iris-versicolour', 'iris-virginica']

        return classes[predicted.item()]
