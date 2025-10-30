# Learned Kalman Filter

## Data Generation

```python
import torch

def generate_kalman_data(batch_size=16, seq_len=50, dt=1.0):
    """
    Generate data for learning Kalman filter

    Returns:
        observations: [B, T, 2] - Noisy position measurements
        true_states: [B, T, 4] - Ground truth [x, y, vx, vy]
    """
    # Initialize random starting states
    states = torch.randn(batch_size, 4)  # [x, y, vx, vy]
    states[:, 2:] *= 0.5  # Smaller velocities

    # Motion model
    F = torch.tensor([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)

    # Observation model (observe position only)
    H = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=torch.float32)

    # Process noise
    process_noise_std = 0.1
    # Observation noise
    obs_noise_std = 0.5

    true_states_list = []
    observations_list = []

    for t in range(seq_len):
        # Store current state
        true_states_list.append(states.clone())

        # Generate observation (position + noise)
        obs = torch.matmul(states, H.T) + torch.randn(batch_size, 2) * obs_noise_std
        observations_list.append(obs)

        # Propagate state
        states = torch.matmul(states, F.T) + torch.randn(batch_size, 4) * process_noise_std

    true_states = torch.stack(true_states_list, dim=1)    # [B, T, 4]
    observations = torch.stack(observations_list, dim=1)  # [B, T, 2]

    return observations, true_states

# Generate data
observations, true_states = generate_kalman_data(batch_size=16, seq_len=50)

print("INPUT:")
print("  observations:", observations.shape)    # [16, 50, 2]
print("  obs noise level:", observations.std().item())
print("\nOUTPUT:")
print("  true_states:", true_states.shape)      # [16, 50, 4]
print("\nVisualize trajectory:")
print("  Position:", true_states[0, :5, :2])
print("  Velocity:", true_states[0, :5, 2:])

# YOUR TASK: Build a learnable Kalman filter
# model = LearnedKalmanFilter()
# predicted_states = model(observations)
# loss = F.mse_loss(predicted_states, true_states)
```

**Output:**
```
INPUT:
  observations: torch.Size([16, 50, 2])
  obs noise level: 15.132975578308105

OUTPUT:
  true_states: torch.Size([16, 50, 4])

Visualize trajectory:
  Position: tensor([[ 0.5475,  0.6218],
        [ 0.6527, -0.1476],
        [ 1.0890, -0.6857],
        [ 1.2830, -1.0682],
        [ 1.3929, -1.3671]])
  Velocity: tensor([[ 0.1766, -0.6363],
        [ 0.3616, -0.4346],
        [ 0.1175, -0.4036],
        [ 0.1457, -0.3459],
        [ 0.1434, -0.2056]])
```

## RNN-based Kalman Filter Model

```python
import torch
import torch.nn as nn

EPOCHS = 100
LEARNING_RATE = 0.01
EVAL_INT = 10

class RNNKalmannFilter(nn.Module):
  def __init__(self):
    super().__init__()

    self.gru = nn.GRU(input_size=2, hidden_size=16, num_layers=2, batch_first=True)
    self.lin = nn.Linear(in_features=16, out_features=4)

  def forward(self, x):
    x, _ = self.gru(x)
    x = self.lin(x)
    return x

model = RNNKalmannFilter()
loss_fn = torch.nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

def train(model, loss_fn, opt, observations, true_states):
  pred = model(observations)
  loss = loss_fn(pred, true_states)

  opt.zero_grad()
  loss.backward()
  opt.step()

  print(f"loss {loss}")

for e in range(EPOCHS):
  train(model, loss_fn, opt, observations, true_states)
```

**Output:**
```
loss 114.74261474609375
loss 112.75377655029297
loss 110.9104232788086
loss 109.04251861572266
...
(loss values decrease over training)
...
loss 36.4130744934082
loss 36.129634857177734
```
