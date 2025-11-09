#stateEstimation 
You have a [[Kalman Filter]] or [[Extended Kalman Filter]], how do you fuse multiple measurements together?

Its actually pretty simple. To handle multiple measurement types coming in at once. We simply **sequentially update our prediction with the respective measurement type one by one**.

> [!info] This is assuming that we know a mapping between the given measurement and a **common state vector**
## Asynchronous Operation

> [!error] Each measurement type can be a predict - update step if asynchronous and different rates!

```
t=0.000s: [State estimate #1] ← Initial state
   ↓
t=0.005s: IMU message arrives
  → Predict from t=0.000 to t=0.005
  → Update with IMU
  [State estimate #2] ← NEW estimate
   ↓
t=0.010s: IMU message arrives
  → Predict from t=0.005 to t=0.010
  → Update with IMU
  [State estimate #3] ← NEW estimate
   ↓
t=0.012s: Wheel odometry arrives
  → Predict from t=0.010 to t=0.012
  → Update with odometry
  [State estimate #4] ← NEW estimate
   ↓
t=0.015s: IMU message arrives
  → Predict from t=0.012 to t=0.015
  → Update with IMU
  [State estimate #5] ← NEW estimate
   ↓
t=0.020s: IMU message arrives
  → Predict from t=0.015 to t=0.020
  → Update with IMU
  [State estimate #6] ← NEW estimate
   ↓
t=0.020s: GPS arrives (same timestamp!)
  → No predict needed (already at t=0.020)
  → Update with GPS
  [State estimate #7] ← NEW estimate
```

## Batched Operation

> [!error] If they arrive in a batch, **we can do a sequence of updates on a single prediction as well**

```
t=0.000s: [State estimate #1] ← Initial state
   ↓
t=0.005s: IMU message and Wheel odometry and GPS arrives arrives
  → Predict from t=0.000 to t=0.005
  → Update with IMU
  → Update with Wheel odometry
  → Update with GPS
  [State estimate #2] ← NEW estimate
   ↓
t=0.010s: IMU message arrives
  → Predict from t=0.005 to t=0.010
  → Update with IMU
  [State estimate #3] ← NEW estimate
   ↓
```

#worldModeling