Measures how two variables vary with each other.

Given two variables $X$ and $Y$:
$$
Cov(X,Y)=E[(X-\mu_{X})(Y-\mu_{Y})]
$$
For two samples of data:
$$
Cov(X,Y)=\frac{1}{n-1}\sum ^{n}_{i=1}(x_{i}-\bar{x})(y_{i}-\bar{y})
$$
# What it tells us
- **Positive** tells us that when one variable **goes up**, the other variable tends to **go up**
- **Negative** tells us that as one variable **goes up**, the other tends to **go down**
- **Zero** tells us that there is no linear relationship between the two variables

![[Pasted image 20251104105445.png]]

>[!info] It **DOES NOT** tell us if two variables are connected in some way, its just used to make a **simple observation**.

# Limitations
Covariance can be heavily skewed if the two variables have different scales. To counteract this, you can either normalize the data (partially the reasoning behind [[Normalization]] techniques), or compute [[Correlation]].
#probability 