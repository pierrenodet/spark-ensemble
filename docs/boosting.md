---
id: boosting
title: Boosting
---

The old Boosting, à la papa, from Freund and Schapire [[1](#references)].

For classification, SAMME (Multi-class AdaBoost) [[2](#references)] from Ji Zhu is implemented.

For regression, R2 (Improving Regressors using Boosting Techniques) [[3](#references)] from H. Drucker has been chosen.

For convenience, a N Round early stop is available.

## Parameters

The parameters available for Boosting are related to early stop and the loss function for weight computation.

```scala
import org.apache.spark.ml.classification.{BoostingClassifier, DecisionTreeClassifier}
        
new BoostingClassifier()
        .setBaseLearner(new DecisionTreeClassifier()) //Base learner used by the meta-estimator.
        .setNumBaseLearners(10) //Number of base learners.
        .setLoss("exponential") //Loss function used for weight computation.
        .setValidationIndicatorCol("val") //Column name that contains true or false for the early stop data set.
        .setTol(1E-3) //Tolerance for optimized step size and gain in loss on early stop set.
        .setNumRound(8) //Number of rounds to wait for the loss on early stop set to decrease.               
```

## References

 * [[1](https://pdf.sciencedirectassets.com/272574/1-s2.0-S0022000000X00384/1-s2.0-S002200009791504X/main.pdf?X-Amz-Security-Token=AgoJb3JpZ2luX2VjEKP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCID7xdi4RVOyBg2JyJaKf%2Bo1b0VB2lLHjOH7N4qiYc1zjAiEA1jYOFh6CjJQZmZ3QcaiDrtaD1P9fYiZkPZM6gR%2B%2FKF8q2gMIXBACGgwwNTkwMDM1NDY4NjUiDAVj5twFPKo1W86Ylyq3A1nD9LiuPFB7iWNzcbJyfjY0ZQjoHwoUo4yrPs9kyH3qntJCFhwM8v1I3278TKFu%2BAtZU%2BJP3OxpJeeXYZ5MPe5g8eYKuwDpdT9mubV3aWr2Vw3EjEkHrVBFE1%2B%2B8Ds3dc9mYqcV87AJCns5uL9mQbh3JTFGuuubYMLkQssmVky%2B3SUvOpW%2Bnl5BTq%2FPqaShPUVW7ky1CLk8%2B3INirdGvWsTeU5GZJRiJqpWYpAS9Qa0Km5BkIPDSHKh5u53tTIUXPqBW6P3MXr2k0XLqpFEIi0%2F8BHDaP%2FcI5EsMvsCFZDsZZAlXZn2Vm8MNZUOnRhOC%2BE2Q1R11o3hly2LGDfz74IihRvXDY40kHvfEwNmeK8y9p7j2NTVUeiNvdjdXpByoEJkmJduPiBVpsQ3SMM3Q6dIm%2BNVzJwMJyQioLcKI7kyC%2FvG6hF9z%2FGRAu7K7hRcbdW5XX2pTES9A5AK9LdeGxvhFThiGfODJaCTPwccTn%2Fw2gDP23uETJ0ldmaqRUJo6TB5LqgeoE6Ll0BwWRJSeUUHybTcVfbFmf6S0ItX42eM0%2Fv7qMsIIWU%2FUV6QRLpth7InXxm7KbUwqurI6wU6tAGcV%2FHkqjW5CYxpXREYK2hHWz12ZKxPV13aBjDyEjTvd85BW0VPpmOixpSlBdV67AnrWSBo1Coo0DNkscAwWepWNDTbZfwHaCd6q7pAyb0RvuD4URqwi2WDTahX9bRK%2BNTAA7vpnfSmv0qLqir02wSLYIP%2Fzf%2FXlhAKyb%2BPTaPhY1Y2JGWkqlykiOMsG2oP42c9LKEVDYkn7y%2Bv62TwYYGQylb%2By1xVnCY3cm9rAFgEGO%2BSWk4%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20190906T114916Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY4UOJFHA3%2F20190906%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=8bfc5bbd29b405a4b4d174cb515bdc2621ac4246c4e30f9a108b143169c0c6c2&hash=d85f9856ba204a5941472cb769535e404f48935a697c8f25a9d241024018527f&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S002200009791504X&tid=spdf-9461bd1f-a6a2-4a1d-9bb1-a6ada11fbdc2&sid=f588c56f21cb124fa58b550402b796a79eb7gxrqb&type=client)] Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. Journal of computer and system sciences, 55(1), 119-139.
 * [[2](https://web.stanford.edu/~hastie/Papers/samme.pdf)] Hastie, T., Rosset, S., Zhu, J., & Zou, H. (2009). Multi-class adaboost. Statistics and its Interface, 2(3), 349-360.
 * [[3](https://pdfs.semanticscholar.org/8d49/e2dedb817f2c3330e74b63c5fc86d2399ce3.pdf)] Drucker, H. (1997, July). Improving regressors using boosting techniques. In ICML (Vol. 97, pp. 107-115).