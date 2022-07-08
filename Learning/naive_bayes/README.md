# Naive Bayes
Naive Bayes is a classifier that uses the Bayes Theorem

## Bayes Theorem
$$P\left(A|B\right) = \frac{P\left(B|A\right)P(A)}{P(B)}$$

## Algorithm
Consider you have a bunch of features $X = \{x_1, x_2, \dots, x_n\}$ and label $y$, the Bayes Theorem states

$$P\left(y|X\right) = \frac{P\left(X|y\right)P(y)}{P(X)}$$

This can be made into something nice if only all the features were independent, that is the assumption of Naive Bayes, it is also the reason why nobody really uses naive bayes. In general you would have features that have some correlation with other features. Anyways, following the assumption we get

$$P\left(y|(x_1,x_2,\dots,x_n)\right) = \frac{P\left(x_1|y\right)P\left(x_2|y\right)\dots P\left(x_n|y\right)P(y)}{P(x_1)P(x_2)\dots P(x_n)}$$

Given some training data, you can calculate all the things on the right.

## Gaussian Naive Bayes
This is naive bayes with just some features being continuous instead of categorical or ordinal. You assume that the feature has a latent gaussian distribution and calculate variance and mean from the data.

## Questions
1. Give some examples where naive bayes could be useful
2. define posterior probability
3. Define prior probability
4. is Naive bayes discriminative or generative?
