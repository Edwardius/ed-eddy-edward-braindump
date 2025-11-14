#proprioception #stateEstimation

see [[Kalman Filter]] for basic explanation of the Kalman Filter.
# Setup

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
# Relationship with Bayes Filter
Coming from [[Bayes Filter]]. We make alot of assumptions to make the math easier to handle.

$$
\underbrace{ p(\mathbf{x}_{k}|\check{\mathbf{x}}_{0},\mathbf{v}_{1:k},\mathbf{y}_{{0:k}}) }_{ \text{posterior belief} }=\eta \underbrace{ p(\mathbf{y}_{k}|\mathbf{x}_{k}) }_{ \substack{\text{observation} \\ \text{correction} \\ \text{using}\;\mathbf{g}(\cdot)} }\int \underbrace{ p(\mathbf{x}_{k}|\mathbf{x}_{k-1},\mathbf{v}_{k}) }_{ \substack{\text{motion prediction} \\ \text{using }\mathbf{f}(\cdot)} }\underbrace{ p(\mathbf{x}_{k-1}|\check{\mathbf{x}}_{0},\mathbf{v}_{1:k-1},\mathbf{y}_{0:k-1}) }_{ \text{prior belief} }d\mathbf{x}_{k-1}
$$
We assume that our posterior belief is **Gaussian**, and our noise variables are **Gaussian** and **outside our non-linear state transition and observation models**.
- Why Gaussians? Two reasons
	- Its a way to represent a continuous PDF in a finite way
	- The normal product of two gaussians is also a gaussian
	- A large portion of randomness in our world is Gaussian in nature (its a safe-enough approximation)
	- **Note:** Passing our Gaussians through non-linear functions cause the output PDF to be non-Gaussian.

You can draw connections better if you take a look at the [[Generalized Gaussian Filter]]
# Linearization
The way an EKF gets around this is by **linearizing the non-linear system**! This is done via a [[Taylor Series]] (only using a first-order approximation).

$$
\mathbf{F}_{k}=\frac{ \partial f }{ \partial \mathbf{x} } \bigg|_{\hat{\mathbf{x}}_{k-1|k-1},\mathbf{u}_{k}} \;\;\; \mathbf{H}_{k}=\frac{ \partial h }{ \partial \mathbf{x} } \bigg|_{\hat{\mathbf{x}}_{k|k-1}}
$$
These both become [[Jacobian|Jacobians]] where...
$$
[\mathbf{F}_k]_{ij} = \frac{\partial f_i}{\partial x_j}, \quad [\mathbf{H}_k]_{ij} = \frac{\partial h_i}{\partial x_j}
$$

# Predict
First compute the state estimate prior with the non-linear state transition function
$$
\hat{\mathbf{x}}_{k|k-1}=f(\hat{\mathbf{x}}_{k-1|k-1},\mathbf{u}_{k})
$$
We derive the first order approximation of the non-linear state about the paste state estimate and control input.
$$
\mathbf{F}_{k}=\frac{ \partial f }{ \partial \mathbf{x} } \bigg|_{\hat{\mathbf{x}}_{k-1|k-1},\mathbf{u}_{k}}
$$
We then use that to predict our state covariance prior.
$$
\mathbf{P}_{k|k-1}=\underbrace{ \mathbf{F}_{k}\mathbf{P}_{k-1|k-1}\mathbf{F}_{k}^{T} }_{ \text{We use jacobian here} }+\mathbf{Q}_{k}
$$

# Update
Compute the first order approximation of the non-linear measurement function using our predicted state prior.
$$
\mathbf{H}_{k}=\frac{ \partial h }{ \partial \mathbf{x} } \bigg|_{\hat{\mathbf{x}}_{k|k-1}}
$$
Compute the Kalman Gain
$$
\mathbf{K}_{k}=\frac{\mathbf{P}_{k|k-1}\mathbf{H}_{k}^{T}}{\mathbf{H}_{k}\mathbf{P}_{k|k-1}\mathbf{H}_{k}^{T}+\mathbf{R}_{k}}
$$
Update the priors with the measurement using the Kalman Gain (this becomes the posterior)
$$
\hat{\mathbf{x}}_{k|k}=\hat{\mathbf{x}}_{k|k-1}+\mathbf{K}_{k}(\mathbf{z}_{k}-\underbrace{ h(\hat{\mathbf{x}}_{k|k-1}) }_{ \substack{\text{We use} \\ \text{non-linear} \\ \text{here} }})
$$
$$
\mathbf{P}_{k|k}=(\mathbf{I}-\mathbf{H}_{k}\mathbf{K}_{k})\mathbf{P}_{k|k-1}
$$
# Another way to Represent this
There's just some notation mess because I learned these two concepts from two different sources. The Extended Kalman Filter can also be defined as the following:
$$
\text{predictor:} \quad 
\check{\mathbf{P}}_k = \mathbf{F}_{k-1}\hat{\mathbf{P}}_{k-1}\mathbf{F}_{k-1}^T + \mathbf{Q}'_k, \tag{4.32a}
$$
$$
\check{\mathbf{x}}_k = \mathbf{f}(\hat{\mathbf{x}}_{k-1}, \mathbf{v}_k, \mathbf{0}), \tag{4.32b}
$$
$$
\text{Kalman gain:} \quad 
\mathbf{K}_k = \check{\mathbf{P}}_k \mathbf{G}_k^T \left(\mathbf{G}_k \check{\mathbf{P}}_k \mathbf{G}_k^T + \mathbf{R}'_k\right)^{-1}, \tag{4.32c}
$$
$$
\text{corrector:} \quad 
\hat{\mathbf{P}}_k = (\mathbf{I} - \mathbf{K}_k \mathbf{G}_k) \check{\mathbf{P}}_k, \tag{4.32d}
$$
$$
\hat{\mathbf{x}}_k = \check{\mathbf{x}}_k + \mathbf{K}_k \underbrace{\left(\mathbf{y}_k - \mathbf{g}(\check{\mathbf{x}}_k, \mathbf{0})\right)}_{\text{innovation}}. \tag{4.32e}
$$
>[![error] Vee is the predicted prior. hat is the corrected posterior
# When do we have a non-linear system?
Its pretty often. 

*IE. an Ackermann Model is an example of a non-linear state transition model*
*IE. landmark detection would be an example of a non-linear measurement model*

---
**EXAMPLE (2D ROBOT LOCALIZATION)**

Say a robot moves in 2D with state
$$
\mathbf{x}_{k}=\begin{bmatrix}
x_{k} \\
y_{k} \\
\theta_{k}
\end{bmatrix}
$$
The **Motion Model** of the robot is differential drive and is given by
$$
f(\mathbf{x}_{k-1},\mathbf{u}_{k})=\begin{bmatrix}
x_{k-1}+v_{k}\cos(\theta_{k-1})\Delta t \\
y_{k-1}+v_{k}\sin(\theta_{k-1})\Delta t \\
\theta_{k-1}+\omega _{k}\Delta t
\end{bmatrix}
\;\; \mathbf{u}_{k}=\begin{bmatrix}
v_{k} \\
\omega_{k}
\end{bmatrix}
$$
Similarly, its measurement model is given by landmark detection (the landmark being in the same coordinate frame as the robot)
$$
h(\mathbf{x}_{k})|_{p=(m_{x},m_{y})}=\begin{bmatrix}
\sqrt{ (m_{x}-x_{k})^{2}+(m_{y}-y_{k})^2 } \\
\underbrace{ \text{atan2} }_{ \text{a func} }(m_{y}-y_{k}, m_{x}-x_{k})
\end{bmatrix}
\;\; \mathbf{z}_{k}=\begin{bmatrix}
D_{k} \\
\gamma_{k}
\end{bmatrix}
$$
**During Prediction**
$$
\hat{\mathbf{x}}_{k|k-1}=f(\hat{\mathbf{x}}_{k-1|k-1},\mathbf{u}_{k})
$$
$$
\mathbf{F}_k = \begin{bmatrix} 1 & 0 & -v_k \sin(\theta_{k-1}) \Delta t \\ 0 & 1 & v_k \cos(\theta_{k-1}) \Delta t \\ 0 & 0 & 1 \end{bmatrix} \;\; so \;\; 

\mathbf{F}_k|_{\hat{x}_{k-1|k-1},\mathbf{u}_{k}} = \begin{bmatrix} 1 & 0 & -v_k \sin(\hat{\theta}_{k-1|k-1}) \Delta t \\ 0 & 1 & v_k \cos(\hat{\theta}_{k-1|k-1}) \Delta t \\ 0 & 0 & 1 \end{bmatrix}
$$
$$
\mathbf{P}_{k|k-1}=\mathbf{F}_{k}\mathbf{P}_{k-1|k-1}\mathbf{F}_{k}^{T}+\mathbf{Q}_{k}
$$
**During Update**
Given that $q=(m_{x}-\hat{x}_{k|k-1})^{2}+(m_{x}-\hat{y}_{k|k-1})^{2}$
$$
\mathbf{H}_k = \begin{bmatrix} \frac{-(m_x - x_k)}{\sqrt{q}} & \frac{-(m_y - y_k)}{\sqrt{q}} & 0 \\ \frac{m_y - y_k}{q} & \frac{-(m_x - x_k)}{q} & -1 \end{bmatrix} \;\;so\;\; 
\mathbf{H}_k|_{\hat{x}_{k|k-1}} = \begin{bmatrix} \frac{-(m_x - \hat{x}_{k|k-1})}{\sqrt{q}} & \frac{-(m_y - \hat{y}_{k|k-1})}{\sqrt{q}} & 0 \\ \frac{m_y - \hat{y}_{k|k-1}}{q} & \frac{-(m_x - \hat{x}_{k|k-1})}{q} & -1 \end{bmatrix} 
$$
$$
\mathbf{K}_{k}=\frac{\mathbf{P}_{k|k-1}\mathbf{H}_{k}^{T}}{\mathbf{H}_{k}\mathbf{P}_{k|k-1}H_{k}^{T}+\mathbf{R}_{k}}
$$
$$
\mathbf{P}_{k|k}=(\mathbf{I}-\mathbf{H}_{k}\mathbf{K}_{k})\mathbf{P}_{k|k-1}
$$
$$
\hat{\mathbf{x}}_{k|k}=\hat{\mathbf{x}}_{k|k-1}+\mathbf{K}_{k}(\mathbf{z}_{k}-h(\hat{\mathbf{x}}_{k|k-1}))
$$
Rinse and repeat

#worldModeling