import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image
import os
import re
import pytorch_device


class WinAndAngleDetectorImageDataset(Dataset):
    def __init__(self, img_dirs_with_labels, transform=None, repeats=40):
        self.img_labels = []
        self.transform = transform
        self.repeats = repeats
        # img_dirs_with_labels is a list of tuples (directory, label)
        for img_dir, class_labeler, qual_labeler in img_dirs_with_labels:
            for img_name in os.listdir(img_dir):
                for _ in range(
                    repeats
                ):  # Assume each image will be augmented into 'repeats' number of crops
                    self.img_labels.append(
                        (
                            os.path.join(img_dir, img_name),
                            class_labeler(img_name),
                            qual_labeler(img_name),
                        )
                    )

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, class_label, qual_label = self.img_labels[idx]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, (class_label, qual_label)


def get_static_labeler(label):
    return lambda unused: label


def degree_labeler(filename):
    match = re.search(r"deg_(\d{3})\.jpg", filename, re.IGNORECASE)

    if match:
        # Extract the numeric portion and convert it to float then
        # normalize to 900 (tenths of a degree from 0-90 degrees)
        return float(match.group(1)) / 900.0
    else:
        # Return None if the pattern does not match
        raise RuntimeError(f"Invalid file name {filename} for degree label")


class WinAndAngleDetector(nn.Module):
    def __init__(self):
        super(WinAndAngleDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(16 * 50 * 50, 128)
        self.fc2 = nn.Linear(128, 1)  # Classification output
        self.fc3 = nn.Linear(16 * 50 * 50, 128)
        self.fc4 = nn.Linear(128, 1)  # Quality output

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(x)  # Additional pooling to reduce dimensions
        x = x.view(-1, 16 * 50 * 50)

        # Original classification output
        classification_output = F.relu(self.fc1(x))
        classification_output = torch.sigmoid(self.fc2(classification_output))

        # New quality metric output, with a linear activation and clamped to [0, 1]
        quality_metric_output = F.relu(self.fc3(x))
        quality_metric_output = self.fc4(quality_metric_output)
        # quality_metric_output = torch.clamp(
        #     quality_metric_output, 0, 1
        # )  # Ensures output is between 0 and 1

        return classification_output, quality_metric_output


def reinitialize_layers(layers):
    for layer in layers:
        if hasattr(layer, "weight"):
            nn.init.xavier_uniform_(layer.weight)
        if hasattr(layer, "bias") and layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


def train(save_weights_path: str, num_epochs: int = 10):
    transform = transforms.Compose(
        [transforms.Resize((440, 440)), transforms.RandomCrop(size=(400, 400))]
    )

    # Define directories with labels
    img_dirs_with_labels = [
        (
            "./temp_data/success_detect/train/yes",
            get_static_labeler(1),
            get_static_labeler(0),
        ),
        (
            "./temp_data/success_detect/train/no",
            get_static_labeler(0),
            get_static_labeler(0),
        ),
        (
            "./temp_data/success_detect/train/angle",
            get_static_labeler(0),
            degree_labeler,
        ),
    ]

    # Create dataset
    dataset = WinAndAngleDetectorImageDataset(
        img_dirs_with_labels=img_dirs_with_labels, transform=transform
    )

    # DataLoader
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Define your SimpleCNNClassifier (assuming it's already defined as per your snippet)
    model = WinAndAngleDetector().to(pytorch_device.device, dtype=torch.float32)

    # Load and re-init some layers
    model.load_state_dict(torch.load("bar_difflayers_goodqual_shitclass.pth"))
    reinitialize_layers([model.fc1, model.fc2])

    criterion_classification = (
        nn.BCELoss()
    )  # Binary Cross-Entropy Loss for classification
    criterion_quality = nn.MSELoss()  # Mean Squared Error Loss for quality metric
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_class_loss = 0.0
        running_qual_loss = 0.0
        for inputs, labels in data_loader:
            class_labels, qual_labels = labels
            inputs = inputs.to(pytorch_device.device, dtype=torch.float32)
            class_labels = class_labels.unsqueeze(1).to(
                pytorch_device.device, dtype=torch.float32
            )
            qual_labels = qual_labels.unsqueeze(1).to(
                pytorch_device.device, dtype=torch.float32
            )

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            class_output, qual_output = model(inputs)

            # Compute loss
            loss_classification = criterion_classification(class_output, class_labels)
            loss_quality = criterion_quality(qual_output, qual_labels)

            # Combine losses
            loss = loss_classification + loss_quality
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_class_loss += loss_classification.item()
            running_qual_loss += loss_quality.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_loader)}")
        print(
            f"Class loss: {running_class_loss/len(data_loader)} Qual loss: {running_qual_loss/len(data_loader)}"
        )

    torch.save(model.state_dict(), save_weights_path)


if __name__ == "__main__":
    import model_utils

    model = WinAndAngleDetector()
    model = model.to(pytorch_device.device, dtype=torch.float32)
    print(f"Model param size {model_utils.get_model_size_mb(model):.2f} MB")
