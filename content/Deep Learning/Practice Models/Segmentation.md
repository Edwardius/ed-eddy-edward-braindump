# VOC Segmentation

## Loading Dataset

```python
from torchvision.datasets import VOCSegmentation
from torchvision import transforms as T

# Transforms to make data consistent
img_transform = T.Compose([
    T.Resize((256, 256)),  # or (512, 512)
    T.ToTensor(),
])

mask_transform = T.Compose([
    T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),  # IMPORTANT!
    T.PILToTensor(),
    T.Lambda(lambda x: x.squeeze(0).long())
])


# Load the dataset - note root should be './data', not './data/VOCdevkit'
train_dataset = VOCSegmentation(root='./data', year='2012', image_set='train', download=False, transform=img_transform, target_transform=mask_transform)
val_dataset = VOCSegmentation(root='./data', year='2012', image_set='val', download=False, transform=img_transform, target_transform=mask_transform)

print(f"Successfully loaded {len(train_dataset)} training images!")
print(f"Testing first sample...")

# Test loading one sample
img, mask = train_dataset[0]
img_v, mask_v = val_dataset[0]
img.shape, mask.shape, img_v.shape, mask_v.shape,
```

**Output:**
```
Successfully loaded 1464 training images!
Testing first sample...

(torch.Size([3, 256, 256]),
 torch.Size([256, 256]),
 torch.Size([3, 256, 256]),
 torch.Size([256, 256]))
```

## Visualizing Dataset

```python
import numpy as np
import matplotlib.pyplot as plt

# Show a single image and values
img, mask = train_dataset[0]
img_v, mask_v = val_dataset[0]

fig, axes = plt.subplots(2, 2, figsize=(12, 5))

axes[0, 0].imshow(img.permute(1, 2, 0))
axes[0, 1].imshow(mask)
axes[1, 0].imshow(img_v.permute(1, 2, 0))
axes[1, 1].imshow(mask_v)

plt.tight_layout()
plt.show()
```

## U-Net Style Segmentation Model

```python
import torch
from torch import nn
from torch.utils.data import DataLoader

import os
from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 32
NUM_CLASSES = 21

LEARNING_RATE = 0.01
EPOCHS = 100
EVAL_INTERVAL = 10

train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

class VOCSegmentation(nn.Module):
  def __init__(self):
    super().__init__()

    # Encoder blocks
    self.enc0_block = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
      nn.BatchNorm2d(num_features=64),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2)  # 256 -> 128
    )

    self.enc1_block = nn.Sequential(
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
      nn.BatchNorm2d(num_features=128),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2)  # 128 -> 64
    )

    self.enc2_block = nn.Sequential(
      nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
      nn.BatchNorm2d(num_features=256),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2)  # 64 -> 32
    )

    # Decoder blocks
    self.dec0_block = nn.Sequential(
      nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),  # 32 -> 64
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
      nn.ReLU(inplace=True)
    )

    self.dec1_block = nn.Sequential(
      nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),  # 64 -> 128
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
      nn.ReLU(inplace=True)
    )

    self.dec2_block = nn.Sequential(
      nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),  # 128 -> 256
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
      nn.ReLU(inplace=True)
    )

    self.final = nn.Conv2d(in_channels=32, out_channels=NUM_CLASSES, kernel_size=1)

  def forward(self, x):
    x = self.enc0_block(x)
    x = self.enc1_block(x)
    x = self.enc2_block(x)
    x = self.dec0_block(x)
    x = self.dec1_block(x)
    x = self.dec2_block(x)
    return self.final(x)

model = VOCSegmentation()
model.cuda()
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
opt = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

def train(model, loss_fn, opt, train_dl):
  model.train()
  losses = []
  for img, mask in train_dl:
    img = img.to(device)
    mask = mask.to(device)

    # TEMPORARY FIX: Force all values into valid range
    mask = torch.where(mask == 255, 255, torch.clamp(mask, 0, 20))

    pred = model(img)
    loss = loss_fn(pred, mask)

    losses.append(loss)

    opt.zero_grad()
    loss.backward()
    opt.step()

  net_loss = sum(losses) / len(losses)
  print(f"TRAIN: NET LOSS {net_loss.detach().cpu()}")


def test(model, val_dl):
  model.eval()
  acc = []
  for img, mask in val_dl:
    img = img.to(device)
    mask = mask.to(device)

    # TEMPORARY FIX: Force all values into valid range
    mask = torch.where(mask == 255, 255, torch.clamp(mask, 0, 20))

    pred = model(img)
    loss = loss_fn(pred, mask)

    acc.append(loss)

  net_acc = sum(acc) / len(acc)
  print(f"TEST NET LOSS {net_acc.detach().cpu()}")

for e in tqdm(range(EPOCHS)):
  train(model, loss_fn, opt, train_dl)

  if e % EVAL_INTERVAL == 0:
    with torch.inference_mode():
      test(model, val_dl)
```

*Note: Training outputs omitted for brevity*

### Architecture Notes:
- Uses a U-Net style encoder-decoder architecture
- **Encoder**: 3 blocks with Conv -> BatchNorm -> ReLU -> MaxPool, progressively downsampling (256 -> 128 -> 64 -> 32)
- **Decoder**: 3 blocks with ConvTranspose (upsampling) -> ReLU -> Conv -> ReLU, progressively upsampling back to 256
- Final 1x1 convolution to produce 21 class predictions (VOC dataset has 21 classes)
- Uses CrossEntropyLoss with `ignore_index=255` for unlabeled pixels
