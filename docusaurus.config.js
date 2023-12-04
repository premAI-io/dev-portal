// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const darkCodeTheme = require('prism-react-renderer/themes/vsDark');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Prem - Developer Portal',
  tagline: '🤖 Self-Sovereign AI Infrastructure',
  favicon: 'img/favicon.png',

  // Set the production url of your site here
  url: 'https://dev.premai.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'premai-io', // Usually your GitHub org/user name.
  projectName: 'dev-portal', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: false,
        blog: {
          routeBasePath: '/blog',
          showReadingTime: true,
          // Remove this to remove the "edit this page" links.
          editUrl: 'https://github.com/premAI-io/dev-portal/blob/main/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  plugins: [],
  themes: [],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/prem-social-card.jpg',
      metadata: [
        {name: 'title', content: 'Prem - Developer Portal.'},
        {name: 'description', content: 'Learn how to Deploy Prem in your Infrastructure or Contribute to Prem Ecosystem.'}],
      navbar: {
        title: 'Developer Portal',
        logo: {
          alt: 'Prem AI logo',
          src: 'img/logo.svg',
        },
        items: [
          {to: '/blog', label: 'Blog', position: 'left'},
          {
            href: 'https://github.com/premAI-io/prem-app',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Community',
            items: [
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/prem',
              },
              {
                label: 'Discord',
                href: 'https://discord.com/invite/kpKk6vYVAn',
              },
              {
                label: 'Twitter',
                href: 'https://twitter.com/premai_io',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'Blog',
                to: '/blog',
              },
              {
                label: 'GitHub',
                href: 'https://github.com/premAI-io',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Prem Labs, Inc.`,
      },
      prism: {
        additionalLanguages: ["ruby", "csharp", "php", "java", "powershell"],
      },
      colorMode: {
        defaultMode: 'dark',
        disableSwitch: true,
      },
    }),
    scripts: [{src: 'https://plausible.io/js/script.js"', defer: true, 'data-domain': 'dev.premai.io'}],
};

async function createConfig() {
  const darkTheme = (await import("./src/utils/prismDark.mjs")).default;
  // @ts-expect-error: we know it exists, right
  config.themeConfig.prism.theme = darkTheme;
  // @ts-expect-error: we know it exists, right
  config.themeConfig.prism.darkTheme = darkTheme;
  return config;
}

module.exports = createConfig;
