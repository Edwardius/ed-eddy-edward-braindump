#proprioception #stateEstimation

see [[Kalman Filter]] for basic explanation of the Kalman Filter.

The Basic Kalman Filter only functions on linear systems.  It assumes that your state can be modeled as:
$$
\mathbf{x}_{k}=\mathbf{F}_{k}\mathbf{x}_{k-1}+\mathbf{B}_{k}\mathbf{u}_{k}+\mathbf{W}_{k}
$$
$$
\mathbf{z}_{k}=\mathbf{H}_{k}\mathbf{x}_{k}+\mathbf{v}_{k}
$$
So as a linear system (in matrix form).

**The Extended Kalman Filter expands on the capabilities of the basic Kalman Filter to handle non-linear systems modeled by**
$$
\mathbf{x}_{k}=f(\mathbf{x}_{k-1},\mathbf{u}_{k})+\mathbf{w}_{k}
$$
$$
\mathbf{z}_{k}=h(\mathbf{x}_{k})+\mathbf{v}_{k}
$$
