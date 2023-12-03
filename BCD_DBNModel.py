import pydicom
import os
import numpy as np
import pandas as pd
from skimage.transform import resize

dicom_dir = 'E:\\INbreast Release 1.0\\AllDICOMs'
csv_path = 'E:\\INbreast Release 1.0\\INbreast.csv'

#DEBUGGING
if not os.path.exists(dicom_dir):
    print("DICOM directory not found.")
if not os.path.exists(csv_path):
    print("CSV file not found.")
    
    
print("Reading CSV file...")
df = pd.read_csv(csv_path, sep=';')  

labels_dict = pd.Series(df['Bi-Rads'].values, index=df['File Name']).to_dict()


image_data = []
labels = []

print("Processing DICOM files...")
for filename in os.listdir(dicom_dir):
    if filename.endswith(".dcm"):
        print(f"Processing {filename}...")
        
       
        file_id = filename.split('_')[0]

        filepath = os.path.join(dicom_dir, filename)
        dicom = pydicom.dcmread(filepath)

  
        img = dicom.pixel_array


        img_resized = resize(img, (256, 256), anti_aliasing=True)

       
        image_data.append(img_resized)

    
        labels.append(labels_dict.get(int(file_id), None))  # Use 'None' or a default value if the file_id is not found


print("Converting to numpy arrays...")

image_data = np.array(image_data)
labels = np.array(labels)

print("Dataset shape:", image_data.shape)
print("Labels shape:", labels.shape)
X = np.array(image_data)
y = np.array(labels)

from sklearn.model_selection import train_test_split

print("Splitting the dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
   
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        
        self.dropout = nn.Dropout(0.5)
    
        self.fc1 = nn.Linear(64 * 53 * 53, 128)  # Adjust the input features size
        self.fc2 = nn.Linear(128, 2)  # Two output classes: high or low risk

    def forward(self, x):
        # Convolutional layers with ReLU and Max pooling
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 53 * 53) 
        x = F.dropout(self.fc1(x), training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = CNN()


def train(model, train_dataloader, optimizer, print_freq=10):
    model.train()

    train_loss = 0
    
    for batch_index, (data, target) in enumerate(train_dataloader):
        #print("Batch_index: ", batch_index, "; Data: ", data, "; Target:", target)
        optimizer.zero_grad()
        output = model(data) # pass the data through the model to get the model output

        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()

        optimizer.step()
        
        train_loss += loss.item() * data.shape[0]
        
        if not (batch_index % print_freq):
            # print the current training loss of the model.
            print("Training Loss:", train_loss)

    return train_loss / len(train_dataloader.dataset)



def test(model, test_dataloader):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        # for every batch of data and labels in the test dataloader
        for batch_index, (data, target) in enumerate(test_dataloader):
            output = model(data)# pass the data through the model to get the model output
            
            test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    test_accuracy = correct / len(test_dataloader.dataset)
    
    return test_loss, test_accuracy



#Create the training loop
import matplotlib.pyplot as plt
results = []
def train_model(model, train_dataloader, test_dataloader, optimizer, num_epochs):

    for i in range(num_epochs):
        train_loss = train(model, train_dataloader, optimizer)
        test_loss, test_accuracy = test(model, test_dataloader)
        results.append((test_loss, test_accuracy))
        print(
            f'Epoch: {i + 1} | Train loss: {train_loss:.5f} |',
            f'Test loss: {test_loss:.5f} | Test accuracy: {test_accuracy:.5f}'
        )
        

        
#Train the model

X_train_tensor = torch.tensor(X_train).float().unsqueeze(1)  # Add channel dimension
X_test_tensor = torch.tensor(X_test).float().unsqueeze(1)
y_train_tensor = torch.tensor(y_train).long()
y_test_tensor = torch.tensor(y_test).long()

from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
train_model(model, train_loader, test_loader, optimizer, num_epochs)

#Plot
epochs = list(range(1, num_epochs + 1))
test_losses = [result[0] for result in results]
test_accuracies = [result[1] for result in results]
            
plt.figure(figsize=(12, 5))

        # Plot test loss
plt.subplot(1, 2, 1)
plt.plot(epochs, test_losses, marker='o', label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Test Loss over Epochs')
plt.legend()

            # Plot test accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, test_accuracies, marker='o', color='r', label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()