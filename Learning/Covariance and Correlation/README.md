# Covariance
Covariance refers to a relationship between two variable with respect to effect of change in one variable over other

$$Covariance(x,y) = \frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})((y_i-\bar{y}))$$

## Use cases
1. Dimensionality Reduction

# Correlation
Correlation of a measure of the degree to which two random variables are correlated or go on in a sequence. When a change in one variable cause equivalent change in the second random variable

$$Correlation(x,y) = \frac{covariance(x,y)}{\sqrt{\sigma_x\sigma_y}}$$

## Use Cases
1. Finding patterns
2. Dimensionality Reduction

Side note, that's something that you would usually use to plot the heatmap (reference hackathon)

## Questions
1. What is the range of correlation b/w two random variables?
2. What is the range of covariance b/w two random variables?
3. which is unit free of the two?
4. Does change in scale affect covariance?
5. Does change in scale affect correlation?

## Answers
1. $[-1, 1]$
2. $(-\infty, \infty)$
3. Correlation
4. Yes
5. No
