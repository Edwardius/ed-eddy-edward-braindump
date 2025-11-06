A graphical representation of a robot's trajectory through space. It consists of:
- **nodes** which are the pose of the robot (position and orientation) at different time steps
- **edges** constraints between the nodes which we've determined from various measurements

>[!info] Each edge has a covariance matrix representing the uncertainty of the constraint.

To reconcile these contraints, we use [[Pose Graph Optimization]].