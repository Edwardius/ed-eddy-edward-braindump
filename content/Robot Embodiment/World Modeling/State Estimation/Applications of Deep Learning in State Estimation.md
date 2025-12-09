# Learning-Based Feature Detection and Matching
Learn to detect features
Learn to correspond features between frames via their descriptors

# Deep Visual Odometry
Monocular Depth Estimation
End-to-End Visual Odometry (CNN-LSTM architecture capable of predicting pose from image sequences)

# Learning-Based Calibration
Predict camera intrinsics from a single image, collects features prone to distortion and FOV (outputs intrinsics).
- trains on synthetic data with known intrinsics
- real data with already estimated intrinsics from SfM
- useful for autocalibration
Extrinsic Calibration (predict extrinsics of camera in lidar frame)

# Loop Closure Detection
Bag of words
Place recognition 

# End-to-End Learning for SLAM
Take in an image sequence and predict depth and pose for each frame. 
- Build a map from those depths and poses which can be differentiable
Learn a map representation through pose net and a branched off NeRF model to do neural rendering

# Outlier Rejection
Learned RANSAC