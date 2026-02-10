**Measures linear acceleration and angular velocity** with 3 orthogonal accelerometers and 3 orthogonal gyros.

>[!error] Modern IMUs use MEMs Technology (Micro-Electro-Mechanical Systems). 

**Accelerometers** are proof masses and springs 
**Gyros** are vibrating proof masses that detect **Coriolis Force** when rotating

If you have a system
![[Pasted image 20251115205338.png]]

Its sensor model (assuming we want the acceleration and angular velocity of the vehicle not the sensor) is
$$ 
\begin{bmatrix} \mathbf{a} \\ \boldsymbol{\omega} \end{bmatrix} = \mathbf{s}\left(\mathbf{r}_i^{vi}, \mathbf{C}_{vi}, \boldsymbol{\omega}_i^{vi}, \mathbf{r}_i^{vi}, \dot{\boldsymbol{\omega}}_i^{vi}\right) 
$$ $$ 
= \begin{bmatrix} \mathbf{C}_{vi}\left(\mathbf{C}_{iv}\left(\ddot{\mathbf{r}}_i^{vi} - \mathbf{g}_i\right) + \dot{\boldsymbol{\omega}}_i^{vi} \times \mathbf{r}_i^{vi} + \boldsymbol{\omega}_i^{vi} \times \boldsymbol{\omega}_i^{vi} \times \mathbf{r}_i^{vi}\right) \\ \mathbf{C}_{vi}\boldsymbol{\omega}_i^{vi} \end{bmatrix} 
$$
