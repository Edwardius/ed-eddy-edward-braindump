#action #robotEmbodiment 

TODO: this needs to be more complete. Time time time time time, i need more time

This is following https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5509799

As well as https://www.ri.cmu.edu/pub_files/2011/5/20100914_icra2011-mcnaughton.pdf

Generates polynomial trajectories to various sampled points. These points are selected either via centerline of a reference path, or something else that is arbitrary. 

We can compute these polynomials in the Frenet Frame. Which is a coordinate system defined relative to a reference path (warped to the path's curvature)

![[Pasted image 20260209143210.png]]

There is a transformation between coordinates in the frenet frame and a basic coordinate frame. 

![[Pasted image 20260209143230.png]]

