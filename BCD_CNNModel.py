import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import train_test_split



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(128 * 32 * 32, 512)  
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 8)  
        
    def forward(self, x):
     
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
     
        x = x.view(-1, 128 * 32 * 32)  
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        

        x = self.fc3(x)
        
        return x

# Load preprocessed data
preprocessed_images = np.load('preprocessed_images.npy')
preprocessed_labels = np.load('preprocessed_labels.npy')

# Create a train-test split
train_images, test_images, train_labels, test_labels = train_test_split(
    preprocessed_images, preprocessed_labels, test_size=0.2, random_state=42
)


class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

batch_size = 5  # Adjust as needed
train_dataset = CustomDataset(train_images, train_labels)
test_dataset = CustomDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def train(model, train_loader, optimizer, print_freq=10):
    model.train()
    train_loss = 0
    criterion = torch.nn.CrossEntropyLoss()  # Use CrossEntropyLoss

    for batch_index, (data, bi_rads_label) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)


        bi_rads_label = torch.tensor(bi_rads_label).long()
        
        loss = criterion(output, bi_rads_label)  # Use CrossEntropyLoss
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        
        if batch_index % print_freq == 0:
            print("Training Loss:", train_loss / (batch_index + 1))

    return train_loss / len(train_loader.dataset)

def test(model, test_loader):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_index, (data, bi_rads_label) in enumerate(test_loader):
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(output, bi_rads_label, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(bi_rads_label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    
    return test_loss, test_accuracy

# Create the training loop
results = []

def train_model(model, train_loader, test_loader, optimizer, num_epochs):
    for i in range(num_epochs):
        train_loss = train(model, train_loader, optimizer)
        test_loss, test_accuracy = test(model, test_loader)
        results.append((test_loss, test_accuracy))
        print(
            f'Epoch: {i + 1} | Train loss: {train_loss:.5f} |',
            f'Test loss: {test_loss:.5f} | Test accuracy: {test_accuracy:.5f}'
        )

if __name__ == "__main__":

    model = CNN()


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    train_model(model, train_loader, test_loader, optimizer, num_epochs)


    epochs = list(range(1, num_epochs + 1))
    test_losses = [result[0] for result in results]
    test_accuracies = [result[1] for result in results]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, test_losses, marker='o', label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Test Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, marker='o', color='r', label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()
