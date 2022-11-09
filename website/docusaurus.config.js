module.exports = {
    title: 'spark-ensemble',
    tagline: 'Ensemble Learning for Apache Spark',
    url: 'https://pierrenodet.github.io',
    baseUrl: 'spark-ensemble',
    favicon: "img/wood-color.png",
    organizationName: 'pierrenodet',
    projectName: 'spark-ensemble',
    themeConfig: {
        navbar: {
            title: 'spark-ensemble',
            logo: {
                src: 'img/wood-color.svg'
            },
            items: [
                { href: '/spark-ensemble/api/org/apache/spark/ml/index.html', label: 'API', position: 'right', target: '_parent' },
                { to: 'docs/overview', label: 'Documentation', position: 'right' },
                {
                    href: "https://github.com/pierrenodet/spark-ensemble",
                    label: 'GitHub',
                    position: 'right'
                },
            ],
        },
        footer: {
            copyright: `Copyright © ${new Date().getFullYear()} Pierre Nodet <br> Icon made by <a href="https://www.flaticon.com/authors/photo3idea-studio" title="photo3idea_studio">photo3idea_studio</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a> licensed by <a href="http://creativecommons.org/licenses/by/3.0/" title="Creative Commons BY 3.0" target="_blank">CC 3.0 BY</a>`,
        },
        colorMode: {
            disableSwitch: true,
        },
        prism: {
            theme: require('prism-react-renderer/themes/github'),
            additionalLanguages: ['java', 'scala'],
        }
    },
    presets: [
        [
            '@docusaurus/preset-classic',
            {
                docs: {
                    path: "../spark-ensemble-docs/target/mdoc",
                    include: ['*.md', '*.mdx'],
                    sidebarPath: require.resolve('./sidebars.js'),
                    editUrl: params =>
                        'https://github.com/pierrenodet/spark-ensemble/edit/master/docs/' + params.docPath,
                },
                theme: {
                    customCss: require.resolve('./src/css/custom.css'),
                },
            },
        ],
    ],
};