const repoUrl = "https://github.com/pierrenodet/spark-ensemble";

const apiUrl = "/spark-ensemble/api/org/apache/spark/index.html";

const siteConfig = {
  title: 'spark-ensemble',
  tagline: 'Ensemble Learning for Apache Spark',
  url: 'https://pierrenodet.github.io/spark-ensemble',
  baseUrl: '/spark-ensemble/',

  projectName: "spark-ensemble",
  organizationName: "pierrenodet",

  customDocsPath: "spark-ensemble-docs/target/mdoc",

  headerLinks: [
    { href: apiUrl, label: "API Docs" },
    { doc: "starting", label: "Documentation" },
    { href: repoUrl, label: "GitHub" }
  ],

  headerIcon: "img/wood.png",
  footerIcon: "img/wood.png",
  favicon: "img/wood.png",

  colors: {
    primaryColor: '#443e8a',
    secondaryColor: '#2f2b60',
  },

  copyright: `Copyright © ${new Date().getFullYear()} Pierre Nodet`,

  highlight: {
    theme: 'github',
  },

  scripts: ['https://buttons.github.io/buttons.js'],

  onPageNav: 'separate',

  separateCss: ["api"],

  cleanUrl: true,

  repoUrl,

  apiUrl
};

module.exports = siteConfig;
