import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from unet_model import UNet
import torch.nn as nn
import torch.optim as optim



#############
# Define IoU function here first
#############
def compute_iou(preds, masks, num_classes=8):
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        mask_cls = (masks == cls)

        intersection = (pred_cls & mask_cls).sum().item()
        union = (pred_cls | mask_cls).sum().item()

        if union == 0:
            continue
        ious.append(intersection / union)

    if len(ious) == 0:
        return 1.0
    return sum(ious)/len(ious)

#############
# Dataset class
#############
class SeepDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.filenames = [f for f in os.listdir(img_dir) if f.endswith('.tif')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        image_array = np.array(image, dtype=np.float32)
        mask_array = np.array(mask, dtype=np.int64)

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array)

        return image_tensor, mask_tensor

#############
# Set up datasets and loaders
#############
train_dataset = SeepDataset("seep_detection/train_images_256", "seep_detection/train_masks_256")
val_dataset = SeepDataset("seep_detection/val_images_256", "seep_detection/val_masks_256")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

#############
# Initialize model, loss, optimizer
#############
model = UNet(n_channels=3, n_classes=8)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#############
# Training loop
#############
num_epochs = 5
print("Starting training...the bactch zise now is 4, so may take a while")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for i, (images, masks) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    val_steps = 0
    with torch.no_grad():
        for images, masks in val_loader:
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            iou = compute_iou(preds, masks, num_classes=8)
            val_iou += iou
            val_steps += 1

    val_loss /= val_steps
    val_iou /= val_steps

    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")