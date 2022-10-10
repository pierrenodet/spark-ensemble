import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import { Redirect } from '@docusaurus/router';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './styles.module.css';

import MarkdownBlock from '../../core/MarkdownBlock';

function Home() {
  const { siteConfig } = useDocusaurusContext();
  const index = `[![Build Status](https://travis-ci.com/pierrenodet/spark-ensemble.svg?branch=master)](https://travis-ci.com/pierrenodet/spark-ensemble) [![codecov](https://codecov.io/gh/pierrenodet/spark-ensemble/branch/master/graph/badge.svg)](https://codecov.io/gh/pierrenodet/spark-ensemble) [![Gitter](https://badges.gitter.im/spark-ensemble/community.svg)](https://gitter.im/spark-ensemble/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) [![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/pierrenodet/spark-ensemble/blob/master/LICENSE) [![Maven Central](https://img.shields.io/maven-central/v/com.github.pierrenodet/spark-ensemble_2.12.svg?label=maven-central&colorB=blue)](https://search.maven.org/search?q=g:%22com.github.pierrenodet%22%20AND%20a:%22spark-ensemble_2.12%22)
  `.trim();
  return (
    <Layout
      title="Home"
      description={`${siteConfig.tagline}`}>
      <header className={clsx('hero hero--primary', styles.heroBanner)}>
        <div className="container">
          <h1 className="hero__title"><img src={siteConfig.favicon}></img>{siteConfig.title}</h1>
          <p className="hero__subtitle">{siteConfig.tagline}</p>
          <div className={styles.buttons}>
            <Link className={clsx(
              'button button--outline button--primary button--lg',
              styles.getStarted,
            )} to={useBaseUrl('api/org/apache/spark/ml')} target="_parent">API Docs</Link>
            <Link className={clsx(
              'button button--outline button--primary button--lg',
              styles.getStarted,
            )} to={useBaseUrl('docs/overview')}>Documentation</Link>
            <Link className={clsx(
              'button button--outline button--primary button--lg',
              styles.getStarted,
            )} to='https://github.com/pierrenodet/spark-ensemble'>View on GitHub</Link>
          </div>
        </div>
      </header>
      <main align="center">
        <MarkdownBlock>{index}</MarkdownBlock>
        <p>Library of Meta-Estimators à la scikit-learn for Ensemble Learning for Apache Spark MLLib</p>
      </main>
    </Layout>
  );
}

export default Home;
