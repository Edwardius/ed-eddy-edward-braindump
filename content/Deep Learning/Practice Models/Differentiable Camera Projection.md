# Differentiable Camera Projection

## Data Generation

```python
import torch

def generate_projection_data(batch_size=8, num_points=100):
    """
    Generate synthetic data for camera projection

    Returns:
        points_3d: [B, N, 3] - 3D points in camera frame
        K: [B, 3, 3] - Camera intrinsic matrices
        pixels_gt: [B, N, 2] - Ground truth 2D projections
    """
    # Generate random 3D points in front of camera
    points_3d = torch.randn(batch_size, num_points, 3)
    points_3d[:, :, 0] = points_3d[:, :, 0] * 2  # x: [-2, 2]
    points_3d[:, :, 1] = points_3d[:, :, 1] * 2  # y: [-2, 2]
    points_3d[:, :, 2] = torch.abs(points_3d[:, :, 2]) * 2 + 2.0  # z: [2, 6]

    # Camera intrinsics (640x480 image)
    K = torch.zeros(batch_size, 3, 3)
    K[:, 0, 0] = 500  # fx
    K[:, 1, 1] = 500  # fy
    K[:, 0, 2] = 320  # cx
    K[:, 1, 2] = 240  # cy
    K[:, 2, 2] = 1.0

    # Ground truth projection (manual implementation)
    points_homo = points_3d.unsqueeze(-1)  # [B, N, 3, 1]
    K_expanded = K.unsqueeze(1)  # [B, 1, 3, 3]
    projected = torch.matmul(K_expanded, points_homo).squeeze(-1)  # [B, N, 3]
    pixels_gt = projected[:, :, :2] / projected[:, :, 2:3]  # [B, N, 2]

    return points_3d, K, pixels_gt

# Generate data
points_3d, K, pixels_gt = generate_projection_data(batch_size=8, num_points=100)

print("INPUT:")
print("  points_3d:", points_3d.shape)  # [8, 100, 3]
print("  K:", K.shape)                   # [8, 3, 3]
print("\nOUTPUT:")
print("  pixels_gt:", pixels_gt.shape)   # [8, 100, 2]

# YOUR TASK: Build a model that takes (points_3d, K) and outputs pixels
# model = YourProjectionModel()
# pixels_pred = model(points_3d, K)
# loss = F.mse_loss(pixels_pred, pixels_gt)
```

**Output:**
```
INPUT:
  points_3d: torch.Size([8, 100, 3])
  K: torch.Size([8, 3, 3])

OUTPUT:
  pixels_gt: torch.Size([8, 100, 2])
```

## Data Preparation

```python
pixels_gt = pixels_gt.flatten(0, 1)
points_3d = points_3d.flatten(0, 1)
pixels_gt.shape, points_3d.shape
```

**Output:**
```
(torch.Size([800, 2]), torch.Size([800, 3]))
```

## Model Training

```python
from torch.utils.data import TensorDataset, DataLoader
from torch import nn

SPLIT = 0.8
BATCH_SIZE = 8
LEARNING_RATE = 0.05
MOMENTUM = 0.9
EPOCHS = 1000
EVAL_INT = 100

# Making the gt homogenous
pixels_gt = torch.cat((pixels_gt, torch.ones(800).unsqueeze(1)), dim=1)
pixels_gt.shape, points_3d.shape

# Normalizing data
class ZNormalization():
  def __init__(self):
    self.std = None
    self.mean = None

  def fit(self, data: torch.Tensor):
    self.mean = data.mean()
    self.std = data.std()

  def normalize(self, data: torch.Tensor):
    return (data - self.mean) / self.std

  def inverse_norm(self, data: torch.Tensor):
    return data * self.std + self.mean

points_norm = ZNormalization()
pixels_norm = ZNormalization()

points_norm.fit(points_3d)
pixels_norm.fit(pixels_gt)

points_3d = points_norm.normalize(points_3d)
print(points_3d.shape)
pixels_gt = pixels_norm.normalize(pixels_gt)
print(pixels_gt.shape)

# Shuffle data
shuffle = torch.randperm(len(points_3d))
points_3d = points_3d[shuffle]
pixels_gt = pixels_gt[shuffle]

# Instantiate Dataloaders
split_i = int(0.8*len(points_3d))
train_X, test_X = points_3d[:split_i], points_3d[split_i:]
train_y, test_y = pixels_gt[:split_i], pixels_gt[split_i:]
train_dl = DataLoader(TensorDataset(train_X, train_y), batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(TensorDataset(test_X, test_y), batch_size=BATCH_SIZE, shuffle=True)

class IntrinsicModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.lin_block = nn.Sequential(
      nn.Linear(in_features=3, out_features=10),
      nn.ReLU(),
      nn.Linear(in_features=10, out_features=3),
    )

  def forward(self, x):
    x = self.lin_block(x)
    return x

model = IntrinsicModel()
loss_fn = torch.nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

def train(model, loss_fn, opt, train_dl):
  losses = []
  for x_b, y_b in train_dl:
    pred = model(x_b)
    loss = loss_fn(pred, y_b)
    losses.append(loss)

    opt.zero_grad()
    loss.backward()
    opt.step()

  avg_loss = sum(losses) / len(losses)
  print(f"MSE TRAIN set {avg_loss}")

def test(model, test_dl):
  losses = []
  for x_b, y_b in test_dl:
    pred = model(x_b)
    loss = loss_fn(pred, y_b)
    losses.append(loss)

  avg_loss = sum(losses) / len(losses)
  print(f"MSE TEST set {avg_loss}")

for e in range(EPOCHS):
  train(model, loss_fn, opt, train_dl)

  if e % EVAL_INT == 0:
    with torch.inference_mode():
      test(model, test_dl)
```

*Note: Training outputs omitted for brevity*
