When a [[Lie Group]] is not commutative ($A*B\neq B*A$), then we end up having to account for this when we do [[Lie Algebra]]. This is because **we want to do nice vector addition in the Lie Algebra**, but if the Lie Group is not commutative, then we end up with error.

# Core
$$
X+Y=Y+X\;\;\text{(vectors commute!)}
$$
$$
g*h\neq h*g\;\;(\text{group elements often don't commute})
$$
# Result
As a result, when you try to map from Lie Algebra to Lie Group with the exponential:
$$
\exp(X+Y)\neq \exp(X)\cdot \exp(Y)\;\;(\text{for non-commutative Lie Groups})
$$
So we need a formula that accounts for this distortion:
$$\exp(X) \cdot \exp(Y) = \exp\left(X + Y + \frac{1}{2}[X,Y] + \frac{1}{12}[X,[X,Y]] - \frac{1}{12}[Y,[X,Y]] + \cdots\right)$$
Where $[X,Y]$ is the **Lie Bracket** and its a common function that deals with the distortion between moving in Lie Algebra and the Lie Group.