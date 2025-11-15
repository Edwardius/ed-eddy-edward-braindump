1. Select a small random subset of the original data.
2. Fit a hypothesis to that sample
3. Analyze the hypothesis with the rest of the data, classify which data points are inliers and outliers
4. Refit the model using both the hypothesized and classified inliers
5. Evaluate the refit model in terms of the residual error with respect to all of the inliers
6. Repeat 4-5 as needed (This is a optimization of the model), once ready, **return the model and its residual error**

>[!error] You do steps 1-6 on **multiple** samples of random data until you find a model you are happy with
