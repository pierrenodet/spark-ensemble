# Ensemble Estimators for Apache Spark
[![Build Status](https://travis-ci.com/pierrenodet/spark-ensemble.svg?branch=master)](https://travis-ci.com/pierrenodet/spark-ensemble)
[![codecov](https://codecov.io/gh/pierrenodet/spark-ensemble/branch/master/graph/badge.svg)](https://codecov.io/gh/pierrenodet/spark-ensemble)
[![Gitter](https://badges.gitter.im/spark-ensemble/community.svg)](https://gitter.im/spark-ensemble/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/pierrenodet/spark-ensemble/blob/master/LICENSE)
[![Maven Central](https://img.shields.io/maven-central/v/com.github.pierrenodet/spark-ensemble_2.12.svg?label=maven-central&colorB=blue)](https://search.maven.org/search?q=g:%22com.github.pierrenodet%22%20AND%20a:%22spark-ensemble_2.12%22)

Library of Meta-Estimators à la scikit-learn for Ensemble Learning for Apache Spark ML

## Getting Started

Download the dependency from Maven Central

**SBT**

```scala
libraryDependencies += "com.github.pierrenodet" %% "spark-ensemble" % "@VERSION@"
```

**Maven**

```maven-pom
<dependency>
  <groupId>com.github.pierrenodet</groupId>
  <artifactId>spark-ensemble_2.12</artifactId>
  <version>@VERSION@</version>
</dependency>
```