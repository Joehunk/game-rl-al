import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.io import read_image
import os
import pytorch_device


class CustomImageDataset(Dataset):
    def __init__(self, img_dirs_with_labels, transform=None):
        self.img_labels = []
        self.transform = transform
        # img_dirs_with_labels is a list of tuples (directory, label)
        for img_dir, label in img_dirs_with_labels:
            for img_name in os.listdir(img_dir):
                self.img_labels.append((os.path.join(img_dir, img_name), label))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label


class SimpleCNNClassifier(nn.Module):
    def __init__(self):
        super(SimpleCNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        # Adjust the fc1 layer to match the correct flattened size
        self.fc1 = nn.Linear(32 * 100 * 100, 256)  # Adjusted to 32*100*100
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 100 * 100)  # Correctly adjusted flatten size
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid for binary classification
        return x


def train(save_weights_path: str):
    # Define transformations
    # transform = transforms.Compose(
    #     [
    #         lambda x: x.half() / 255,
    #         transforms.Resize((400, 400))  
    #     ]
    # )

    transform = transforms.Resize((400, 400))  

    # Define directories with labels
    img_dirs_with_labels = [
        ("./temp_data/success_detect/yes", 1),
        ("./temp_data/success_detect/no", 0),
    ]

    # Create dataset
    dataset = CustomImageDataset(
        img_dirs_with_labels=img_dirs_with_labels, transform=transform
    )

    # DataLoader
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Define your SimpleCNNClassifier (assuming it's already defined as per your snippet)
    model = SimpleCNNClassifier().to(pytorch_device.device, dtype=torch.float32)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    for epoch in range(20):
        for inputs, labels in data_loader:
            inputs = inputs.to(pytorch_device.device, dtype=torch.float32)
            labels = labels.unsqueeze(1).to(pytorch_device.device, dtype=torch.float32)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    print(f"Saving weights to {save_weights_path}")
    torch.save(model.state_dict(), save_weights_path)


if __name__ == "__main__":
    import model_utils

    model = SimpleCNNClassifier()
    model = model.to(pytorch_device.device, dtype=torch.float16)
    print(f"Model param size {model_utils.get_model_size_mb(model):.2f} MB")
