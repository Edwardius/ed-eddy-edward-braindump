These relate to algorithms that have to do with robot's processing and acting on the world around it. It is an unorganized compilation of our attempts to mimic [[00 - Physiological Psychology Table of Contents|the processes of the human brain]]. This does not include implementation specific details like tools and coding standards. For that see [[00 - Roboting]]

# Table of Contents

[[General Architecture of Robotics Systems]]
## Perception

## World Modeling
**ICP**
[[Iterative Closest Point (ICP)]]
[[Generalized Iterative Closest Point (GICP)]]
[[Mahalanobis Distance]]

**Pose Graphs**
[[Pose Graph]]
[[Pose Graph Optimization]]

**Special Groups**
[[Lie Algebra for EKF]]
[[SE(3)]]
[[SO(3)]]
### State Estimation
This stuff was referencing http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf

[[What is State Estimation]]
[[Applications of Deep Learning in State Estimation]]
[[Measurement Fusion]]

**3D Geometries**
[[3D Numanclature]]
[[Key Identities]]
[[Poses and Transformations]]
[[Rotation Representations]]
[[Rotational Kinematics]]

**Calibration**
[[Camera Calibration]]

**Handling Non-Idealities**
[[Finding Covariance W]]
[[Internal vs External Data Association]]
[[RANSAC]]
[[Robust Loss Functions]]
[[Unbiased and Consistent]]

**Linear-Gaussian Estimation**
[[Robot Embodiment/Robot Emb - World Modeling/State Estimation/Non-linear Non-Gaussian Estimation/Bayesian Inference|Bayesian Inference]]
[[Exploiting Sparsity in Batch Solution]]
[[Kalman Filter]]
[[LG Problem Statement]]
[[Robot Embodiment/Robot Emb - World Modeling/State Estimation/Non-linear Non-Gaussian Estimation/Maximum A Posteriori|Maximum A Posteriori]]
[[Rauch-Tung-Striebel Smoother]]

**Non-linear Non-Gaussian Estimation**
[[Bayes Filter]]
[[Robot Embodiment/Robot Emb - World Modeling/State Estimation/Non-linear Non-Gaussian Estimation/Bayesian Inference|Bayesian Inference]]
[[Extended Kalman Filter]]
[[Generalized Gaussian Filter]]
[[Iterative Extended Kalman Filter]]
[[Iterative Sigmapoint (Unscented) Kalman Filter]]
[[Robot Embodiment/Robot Emb - World Modeling/State Estimation/Non-linear Non-Gaussian Estimation/Maximum A Posteriori|Maximum A Posteriori]]
[[Monte Carlo Method]]
[[NLNG Problem Statement]]
[[Particle Filter]]
[[Sigmapoint (Unscented) Kalman Filter]]
[[Sigmapoint (Unscented) Transformation]]
[[Sliding-Window Filter]]
[[Ways to Handle Non-linearities]]
## Memory
(this one is harder to see, but what im trying to get here is that there is some sort of memory management here, something that I feel like exceeds world modeling)
## Configuration


## Action

## End to End

 #memory #worldModeling #perception #configuration #action #endToEnd