import warnings
import csv
import torch
import time
import cv2
import os
import torch.utils.data as Data
import torchvision.transforms as transforms
import model.ResNet
from torch import nn
from torch import optim

# Setting up the device for GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
print(device)

def dataReader():
    """
    Reads the data from the dataset and labels from the CSV files.
    Applies transformations to the images and returns the input and label tensors.
    """
    Input = []
    label = []

    # Iterate through 10 datasets
    for i in range(1, 11):
        data_path = f'../Dataset/Capture_{i}/'
        label_path = f'../Dataset/KeyCapture_{i}.csv'

        # Load and transform images
        for file in os.listdir(data_path):
            img = cv2.imread(data_path + file)
            transf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
            img = transf(img)  # Convert image to tensor
            Input.append(img)

        # Load labels
        with open(label_path, 'rt') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            rows = [list(map(int, row)) for row in reader]
        label.extend(rows)

    return Input, label

# Data augmentation for imbalance handling
def augmentData(data_x, data_y):
    """
    Augments the data by replicating samples based on certain conditions.
    """
    for i in range(len(data_y)):
        if data_y[i][0] != 0 or data_y[i][1] != 0:
            data_x += [data_x[i]] * 3
            data_y += [data_y[i]] * 3

        if data_y[i][2] != 0 or data_y[i][3] != 0:
            data_x += [data_x[i]] * 2
            data_y += [data_y[i]] * 2

    return data_x, data_y

# Reading and augmenting data
data_x, data_y = dataReader()
data_x, data_y = augmentData(data_x, data_y)

# Converting data to tensors
data_x = torch.stack(data_x, dim=0)
data_y = torch.FloatTensor(data_y)
print(data_x.shape)
print(data_y.shape)

# Model, loss, and optimizer
ResNet = ResNet.ResNet18()
ResNet.to(device)
epochs = 200
optimizer = optim.Adam(ResNet.parameters(), lr=0.0005, weight_decay=0.0)
criterion = nn.BCELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, threshold=1e-3)

# Training loop
batch_size = 64
torch_dataset = Data.TensorDataset(data_x, data_y)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(1, epochs + 1):
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        output = ResNet(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    TimeStr = time.asctime(time.localtime(time.time()))
    print(f'Epoch: {epoch} --- {TimeStr} --- ')
    print(f'Train Loss of the model: {loss}')
    print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
    scheduler.step(loss)

# Save the trained model
torch.save(ResNet.state_dict(), 'ResNet.pkl')