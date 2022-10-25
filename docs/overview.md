---
id: overview
title: Overview
---

## Why?

Want to switch Decision Trees to Neural Net in Random Forest to make avNNet?

There is no curse in elvish, entish, or the tongues of men for this treachery.

But it's easy now!

```scala
val randomForest = new BaggingClassifier()
  .setBaseLearner(new DecisionTreeClassifier())
```
  
```scala
val avNNet = new BaggingClassifier()
   .setBaseLearner(new MultiLayerPerceptronClassifier())
```

## Implemented Meta-Algorithms

Le classico of meta-algorithms have been implemented in this library as closely as described in the papers of the original authors.

Here is the current list :

 * **Stacking**
 * **Bagging**
 * **Boosting**
 * **GBM**
 
They all work with Multiclass Classification and Regression.

## Ready to Install?

Download the dependency from Maven Central

**SBT**

```scala
libraryDependencies += "com.github.pierrenodet" %% "spark-ensemble" % "@VERSION@"
```

**Maven**

```maven-pom
<dependency>
    <groupId>com.github.pierrenodet</groupId>
    <artifactId>spark-ensemble_2.13</artifactId>
    <version>@VERSION@</version>
</dependency>
```