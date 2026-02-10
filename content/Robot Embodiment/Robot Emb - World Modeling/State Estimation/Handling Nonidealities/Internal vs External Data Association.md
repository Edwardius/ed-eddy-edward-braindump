When getting data measurements, we need to make sure that the measurement we are taking now is of the same thing we are measuring right before.

IE. GPS satellite measurement is coming from the same satellite as before

**External Data Association** is where the data is being associated **before** it is being considered in our state estimator.
- GPS, Visual Feature Matching, (loop closure detection, feature-based initial alignment), Apriltags

**Internal Data Association** is where data is being associating while inside our state estimator
- only the **measurements/model** are used to do data association
- ICP is one of them, as it relies on a good initial guess AND it solve association and state optimization together