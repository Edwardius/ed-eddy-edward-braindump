Using what we discussed in [[Rotation Representations]]

# Angular Velocity
The angular velocity of frame 2 with respect to frame 1 is denoted as $\vec{\omega_{21}}$
The angular velocity of frame 1 with respect to frame 2 is denoted as $\vec{\omega_{12}}=-\vec{\omega_{21}}$

![[Pasted image 20251115164029.png]]
## Important Applications
A *vector time derivative* is the instantaneous rate of change of a vector. **Different frames of reference see different motion** so we need to analyze the motion of two different frames separately.

### In Frame  1
We can define the vector time derivative in frame 1 as $\dot{(\cdot)}$. Hence the vector time derivative in frame 1 of frame 1 itself is nothing
$$
\dot{\vec{\mathcal{F}}}_{1}=\vec{0}
$$
Given a angular velocity, $\vec{\omega_{21}}$, the vector time derivative of frame 2 is given by
$$
\dot{\vec{\mathcal{F}}}_{2}=\vec{\omega_{21}}\times\vec{\mathcal{F}}_{2}
$$
**Say we have a vector expressed in both frames**
$$
\vec{r}=\vec{\mathcal{F}}_{1}^{T}\mathbf{r}_{1}=\vec{\mathcal{F}}_{2}^{T}\mathbf{r}_{2}
$$
The time derivative of such vector **from the perspective of Frame 1** is
$$
\dot{\vec{r}}=\cancelto{ 0 }{ \dot{\vec{\mathcal{F}}}_{1}^{T}\mathbf{r}_{1} }+\vec{\mathcal{F}}_{1}^{T}\dot{\mathbf{r_{1}}}=\vec{\mathcal{F}}_{1}^{T}\dot{\mathbf{r_{1}}}
$$
#### In Frame 2
We will define the vector time derivative in frame 2 as $\mathring{(\cdot)}$.
$$
\mathring{\vec{r}}=\cancelto{ 0 }{ \mathring{\vec{\mathcal{F}}}_{2}^{T}\mathbf{r}_{2} }+\vec{\mathcal{F}}_{2}^{T}\mathring{\mathbf{r}_{2}}=\vec{\mathcal{F}}_{2}^{T}\mathring{\mathbf{r}_{2}}
$$
### Combining Perspectives (Relationship of Vector Time Derivative)
$$
\vec{\mathcal{F}}_{2}^{T}\dot{\mathbf{r}}_{2}=\vec{\mathcal{F}}_{2}^{T}\mathring{\mathbf{r}}_{2} \quad \dot{\mathbf{r}}_{2}=\mathring{\mathbf{r}}_{2}
$$
The vector time derivative of $\vec{r}$ in expressed in terms of frame 2 **but as seen in frame 1** is given by
$$
\dot{\vec{r}}=\mathring{\vec{r}}+\vec{\omega}_{21}\times \vec{r}
$$
>[!error] This is telling us how the time derivative of a vector witnessed between two different frames changes based on the angular velocity between two different frames

## Inertial Time Derivative
A useful application is when we detect something in a moving frame, and we want to transform it to a non-rotating inertial reference frame (a frame that does not rotate or accelerate, it is often something like the map origin or map reference frame).

Say our rotating frame $\vec{\mathcal{F}}_{2}$ is undergoing a angular velocity $\vec{\omega}_{21}$. 

**First we express the angular velocity**
$$
\vec{\omega}_{21}=\vec{\mathcal{F}}^{T}_{2}\mathbf{\omega}_{2}^{21}
$$
We detected an object in Frame 2 with position $\vec{\mathcal{F}}_{2}^{T}\mathbf{r}_{2}$ . Because Frame 2 is undergoing an angular velocity, the change in the position of the object in the non-rotation inertial frame is given by
$$\dot{\mathbf{r}}_{1}=\mathbf{C}_{12}(\dot{\mathbf{r}}_{2}+\boldsymbol{\omega}_{2}^{21\times}\mathbf{r}_{2})
$$
# Acceleration
A similar formula can be derived for acceleration
$$
\ddot{\vec{r}}=\mathring{\mathring{\vec{r}}}+2\vec{\omega}_{21}\times \mathring{\vec{r}}+\mathring{\vec{\omega}}_{21}\times \vec{r}+ \vec{\omega}_{21}\times(\vec{\omega}_{21}\times \vec{r})
$$
$$
\ddot{\mathbf{r}_{1}}=\mathbf{C}_{12}[\ddot{\mathbf{r}_{2}}+2\boldsymbol{\omega}_{2}^{21\times}\dot{\mathbf{r}}+\dot{\boldsymbol{\omega}}_{2}^{21\times}\mathbf{r}_{2}+\boldsymbol{\omega}_{2}^{21\times}\boldsymbol{\omega}_{2}^{21\times}\mathbf{r}_{2}]
$$
>[!error] note that we are only looking at the instantaneous velocities and accelerations of $\vec{r}$ when the two reference frames are on top of each other and frame 2 in induced with a angular velocity

# Angular Velocity Given Rotation Matrix
When our rotation matrix is given as a function of time, our angular velocity is given by:
$$
\boldsymbol{\omega}_{2}^{21\times}=-\dot{\mathbf{C}}_{21}\mathbf{C}_{21}^{-1}=-\dot{\mathbf{C}}_{21}\mathbf{C}_{21}^{T}
$$
