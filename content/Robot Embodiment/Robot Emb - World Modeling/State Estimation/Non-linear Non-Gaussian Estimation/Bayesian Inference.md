Extension of [[Robot Embodiment/Robot Emb - World Modeling/State Estimation/Linear-Gaussian Estimation/Bayesian Inference|Bayesian Inference]] except for [[NLNG Problem Statement]]. 

See [[Robot Embodiment/Robot Emb - World Modeling/State Estimation/Non-linear Non-Gaussian Estimation/Maximum A Posteriori|Maximum A Posteriori]] perspective of inference as well. We will be using [[Gauss-Newton Method]]

# Predict 
Following our model, we can linearize it about an operating point.
$$
\mathbf{x}_k \approx \mathbf{f}(\mathbf{x}_{\text{op},k-1}, \mathbf{v}_k, \mathbf{0}) + \mathbf{F}_{k-1}(\mathbf{x}_{k-1} - \mathbf{x}_{\text{op},k-1}) + \mathbf{w}_k'
$$
$$
\mathbf{F}_{k-1} = \frac{\partial \mathbf{f}(\mathbf{x}_{k-1}, \mathbf{v}_k, \mathbf{w}_k)}{\partial \mathbf{x}_{k-1}}\bigg|_{\mathbf{x}_{\text{op},k-1}, \mathbf{v}_k, 0}, \quad \mathbf{G}_k = \frac{\partial \mathbf{g}(\mathbf{x}_k, \mathbf{n}_k)}{\partial \mathbf{x}_k}\bigg|_{\mathbf{x}_{\text{op},k}, 0}
$$

>[!info] The $(·)'$ notation is used to indicate that the Jacobian with respect to the noise is incorporated into the quantity.


We can lift this into lifted matrix form like in the previous [[Robot Embodiment/Robot Emb - World Modeling/State Estimation/Linear-Gaussian Estimation/Bayesian Inference|Bayesian Inference]]
$$
\mathbf{x} = \mathbf{F}(\boldsymbol{\nu} + \mathbf{w}')
$$
$$
\boldsymbol{\nu} = \begin{bmatrix} \mathbf{f}(\mathbf{x}_{\text{op},0}, \mathbf{v}_1, \mathbf{0}) - \mathbf{F}_0\mathbf{x}_{\text{op},0} \\ \mathbf{f}(\mathbf{x}_{\text{op},1}, \mathbf{v}_2, \mathbf{0}) - \mathbf{F}_1\mathbf{x}_{\text{op},1} \\ \vdots \\ \mathbf{f}(\mathbf{x}_{\text{op},K-1}, \mathbf{v}_K, \mathbf{0}) - \mathbf{F}_{K-1}\mathbf{x}_{\text{op},K-1} \end{bmatrix}
$$ $$
\mathbf{F} = \begin{bmatrix} \mathbf{1} & & & & \\ \mathbf{F}_0 & \mathbf{1} & & & \\ \mathbf{F}_1\mathbf{F}_0 & \mathbf{F}_1 & \mathbf{1} & & \\ \vdots & \vdots & \vdots & \ddots & \\ \mathbf{F}_{K-2}\cdots\mathbf{F}_0 & \mathbf{F}_{K-2}\cdots\mathbf{F}_1 & \mathbf{F}_{K-2}\cdots\mathbf{F}_2 & \cdots & \mathbf{1} \\ \mathbf{F}_{K-1}\cdots\mathbf{F}_0 & \mathbf{F}_{K-1}\cdots\mathbf{F}_1 & \mathbf{F}_{K-1}\cdots\mathbf{F}_2 & \cdots & \mathbf{F}_{K-1} & \mathbf{1} \end{bmatrix}
$$ $$
\mathbf{Q}' = \text{diag}(\bar{\mathbf{P}}_0, \mathbf{Q}_1', \mathbf{Q}_2', \ldots, \mathbf{Q}_K')
$$and $\mathbf{w}' \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}')$. 

From this, we can derive the mean and covariance
$$
\check{\mathbf{x}} = E[\mathbf{x}] = E[\mathbf{F}(\boldsymbol{\nu} + \mathbf{w}')] = \mathbf{F}\boldsymbol{\nu}
$$
$$
\check{\mathbf{P}} = E\left[(\mathbf{x} - E[\mathbf{x}])(\mathbf{x} - E[\mathbf{x}])^T\right] = \mathbf{F} E\left[\mathbf{w}\mathbf{w}'^T\right] \mathbf{F}^T = \mathbf{F}\mathbf{Q}'\mathbf{F}^T
$$
Thus, the prior can be summarized as $\mathbf{x} \sim \mathcal{N}(\mathbf{F}\boldsymbol{\nu}, \mathbf{F}\mathbf{Q}'\mathbf{F}^T)$. 
# Update
$$
\mathbf{y}_k \approx \mathbf{g}(\mathbf{x}_{\text{op},k}, \mathbf{0}) + \mathbf{G}_k(\mathbf{x}_{k-1} - \mathbf{x}_{\text{op},k-1}) + \mathbf{n}_k'
$$ which can be written in lifted form as 
$$
\mathbf{y} = \mathbf{y}_{\text{op}} + \mathbf{G}(\mathbf{x} - \mathbf{x}_{\text{op}}) + \mathbf{n}'
$$ where 
$$
\mathbf{y}_{\text{op}} = \begin{bmatrix} \mathbf{g}(\mathbf{x}_{\text{op},0}, \mathbf{0}) \\ \mathbf{g}(\mathbf{x}_{\text{op},1}, \mathbf{0}) \\ \vdots \\ \mathbf{g}(\mathbf{x}_{\text{op},K}, \mathbf{0}) \end{bmatrix}
$$ $$
\mathbf{G} = \text{diag}(\mathbf{G}_0, \mathbf{G}_1, \mathbf{G}_2, \ldots, \mathbf{G}_K)
$$ $$
\mathbf{R} = \text{diag}(\mathbf{R}_0', \mathbf{R}_1', \mathbf{R}_2', \ldots, \mathbf{R}_K')
$$ and $\mathbf{n}' \sim \mathcal{N}(\mathbf{0}, \mathbf{R}')$.

$$
E[\mathbf{y}] = \mathbf{y}_{\text{op}} + \mathbf{G}(\bar{\mathbf{x}} - \mathbf{x}_{\text{op}})
$$ $$
E\left[(\mathbf{y} - E[\mathbf{y}])(\mathbf{y} - E[\mathbf{y}])^T\right] = \mathbf{G}\bar{\mathbf{P}}\mathbf{G}^T + \mathbf{R}'
$$ $$
E\left[(\mathbf{y} - E[\mathbf{y}])(\mathbf{x} - E[\mathbf{x}])^T\right] = \mathbf{G}\bar{\mathbf{P}}
$$ With these quantities in hand, we can write a [[Joint Gaussian PDFs]] for the lifted trajectory and measurements as 
$$
p(\mathbf{x}, \mathbf{y}|\boldsymbol{\nu}) = \mathcal{N}\left(\begin{bmatrix} \check{\mathbf{x}} \\ \mathbf{y}_{\text{op}} + \mathbf{G}(\check{\mathbf{x}} - \mathbf{x}_{\text{op}}) \end{bmatrix}, \begin{bmatrix} \check{\mathbf{P}} & \check{\mathbf{P}}\mathbf{G}^T \\ \mathbf{G}\check{\mathbf{P}} & \mathbf{G}\check{\mathbf{P}}\mathbf{G}^T + \mathbf{R}' \end{bmatrix}\right)
$$ From the factorization we did with [[Joint Gaussian PDFs]]
$$
p(\mathbf{x}|\boldsymbol{\nu}, \mathbf{y}) = \mathcal{N}(\hat{\mathbf{x}}, \hat{\mathbf{P}})
$$
where 
$$
\mathbf{K} = \check{\mathbf{P}}\mathbf{G}^T(\mathbf{G}\check{\mathbf{P}}\mathbf{G}^T + \mathbf{R}')^{-1}
$$$$
\hat{\mathbf{P}} = (\mathbf{I} - \mathbf{K}\mathbf{G})\check{\mathbf{P}}
$$$$
\hat{\mathbf{x}} = \check{\mathbf{x}} + \mathbf{K}(\mathbf{y} - \mathbf{y}_{\text{op}} - \mathbf{G}(\check{\mathbf{x}} - \mathbf{x}_{\text{op}}))
$$Using the SMW identity from (2.124), we can rearrange the equation for the posterior mean to be 
$$
\left(\check{\mathbf{P}}^{-1} + \mathbf{G}^T\mathbf{R}'^{-1}\mathbf{G}\right) \delta\mathbf{x}^* = \check{\mathbf{P}}^{-1}(\check{\mathbf{x}} - \mathbf{x}_{\text{op}}) + \mathbf{G}^T\mathbf{R}'^{-1}(\mathbf{y} - \mathbf{y}_{\text{op}})
$$ where $\delta\mathbf{x}^* = \hat{\mathbf{x}} - \mathbf{x}_{\text{op}}$. Inserting the details of the prior, this becomes $$
\underbrace{\left(\mathbf{F}^{-T}\mathbf{Q}'^{-1}\mathbf{F}^{-1} + \mathbf{G}^T\mathbf{R}'^{-1}\mathbf{G}\right)}_{\text{block-tridiagonal}} \delta\mathbf{x}^* = \mathbf{F}^{-T}\mathbf{Q}'^{-1}(\boldsymbol{\nu} - \mathbf{F}^{-1}\mathbf{x}_{\text{op}}) + \mathbf{G}^T\mathbf{R}'^{-1}(\mathbf{y} - \mathbf{y}_{\text{op}})
$$ Then, under the definitions
$$
\mathbf{H} = \begin{bmatrix} \mathbf{F}^{-1} \\ \mathbf{G} \end{bmatrix}, \quad \mathbf{W} = \text{diag}(\mathbf{Q}', \mathbf{R}'), \quad \mathbf{e}(\mathbf{x}_{\text{op}}) = \begin{bmatrix} \boldsymbol{\nu} - \mathbf{F}^{-1}\mathbf{x}_{\text{op}} \\ \mathbf{y} - \mathbf{y}_{\text{op}} \end{bmatrix}
$$ we can rewrite this as 
$$
\underbrace{(\mathbf{H}^T\mathbf{W}^{-1}\mathbf{H})}_{\text{block-tridiagonal}} \delta\mathbf{x}^* = \mathbf{H}^T\mathbf{W}^{-1}\mathbf{e}(\mathbf{x}_{\text{op}})
$$
Which is the same as what we got with [[Robot Embodiment/World Modeling/State Estimation/Non-linear Non-Gaussian Estimation/Maximum A Posteriori|Maximum A Posteriori]] , except this time we explicitly show that the covariance of the estimate can be approximated with 
$$
\hat{\mathbf{P}}=(\mathbf{H}^T\mathbf{W}^{-1}\mathbf{H})^{-1}
$$