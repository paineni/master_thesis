import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import fiftyone as fo
import csv
import os
from EMA import (
    EMA_region_begin,
    EMA_region_define,
    EMA_region_end,
    EMA_init,
    EMA_finalize,
)


# Define the custom dataset class
class FiftyOneDataset(Dataset):
    def __init__(self, dataset, transform=None):
        """
        Args:
            dataset (fo.Dataset): FiftyOne dataset object.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.dataset = dataset
        self.transform = transform
        # Create a list of sample IDs
        self.sample_ids = [sample.id for sample in dataset]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        # Use sample ID to access the sample
        sample_id = self.sample_ids[idx]
        sample = self.dataset[sample_id]

        # Extracting the image file path and label from the sample
        image_path = sample.filepath
        class_label = sample.label.label  # Extract the label from the sample
        lst = ["cocoa", "invalid"]
        label = lst.index(class_label)
        # Open the image file and convert to RGB
        image = Image.open(image_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label


# Load your FiftyOne dataset
dt = fo.Dataset.from_dir(
    "cropped_classification_dataset", dataset_type=fo.types.FiftyOneDataset
)
view1 = dt.match_tags("train")
view2 = dt.match_tags("valid")
# Define transformations (assuming images are already 224x224)
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


# Initialize VGG16 with pre-trained weights
vgg16 = models.vgg16(pretrained=True)

# Freeze convolutional layers
for param in vgg16.parameters():
    param.requires_grad = False

# Modify the classifier
num_features = vgg16.classifier[6].in_features
vgg16.classifier[6] = nn.Sequential(
    nn.Linear(num_features, 1024),  # Reduce units to 1024
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    # Consider removing one layer here if needed
    nn.Linear(1024, 2),  # Adjust based on your number of classes
)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    vgg16.classifier.parameters(), lr=2e-5, weight_decay=1e-4
)

# Prepare CSV file for logging
csv_file = "vgg_training_loss_log.csv"
if os.path.exists(csv_file):
    os.remove(csv_file)

with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Training Loss", "Validation Loss"])

# Early stopping parameters
patience = 10
best_loss = float("inf")
patience_counter = 0

# Split the dataset into training and validation sets
train_dataset = FiftyOneDataset(view1, transform=transform)
valid_dataset = FiftyOneDataset(view2, transform=transform)

train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=2
)
val_loader = DataLoader(
    valid_dataset, batch_size=16, shuffle=False, num_workers=2
)

# Training the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16.to(device)

num_epochs = 200  # Set the number of epochs
EMA_init()
region = EMA_region_define("abc")
EMA_region_begin(region)
for epoch in range(num_epochs):
    vgg16.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = vgg16(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_loader.dataset)

    # Validation phase
    vgg16.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = vgg16(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

    val_loss = val_loss / len(val_loader.dataset)

    # Log loss to CSV
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, train_loss, val_loss])

    # Early stopping check
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        # Save the best model weights
        torch.save(vgg16.state_dict(), "best_vgg16_model_weights.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    print(
        f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
    )
EMA_region_end(region)
EMA_finalize()

torch.save(vgg16.state_dict(), "final_vgg16_model_weights.pth")
print("Training complete")
