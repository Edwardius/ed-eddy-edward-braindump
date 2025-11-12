#probability 

>[!error] this is a sort of looseness in the definition of Bayes' Theorem because we can kinda define things as probabilities however we want. For me, I find it better to mentally model this as both a mathematical and philisophical concept
# Basic Bayes' Theorem
Also known as Bayes' rule or Bayes' law  is a result in probability theory that relates conditional probabilities. 

$$
P(A|B)=\frac{P(B|A)P(A)}{P(B)}
$$
The basic Bayes' Theorem relates probabilities of **events**. ie. "its cloudy" "there's rain".

>[!error] There are different interpretations of Bayes Theorem

# Bayesian Inference
Same equation, but we are now thinking in terms of **parameter estimation**

$$
P(\theta|D)=\frac{P(D|\theta)P(\theta)}{P(D)}=\frac{P(D|\theta)P(\theta)}{\int P(D|\theta)P(\theta)d\theta}
$$

Where:
- **$\theta$** (your "A") = **parameters** (unknown quantities you want to estimate)
- **$D$** (your "B") = **data/observations** (known, fixed quantities you've observed)

This interpretation has **special terminology**:
- **$P(\theta)$** = **prior** (your beliefs about parameters before data) (the probability distribution of the parameters them selves)
- **$P(D|\theta)$** = **likelihood** (probability of data given parameters)
- **$P(\theta|D)$** = **posterior** (updated beliefs about parameters after seeing data)
- **$P(D)$** = **evidence/marginal likelihood** (normalizing constant)