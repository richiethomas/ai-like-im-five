export const siteConfig = {
  /** Wordmark shown in the header and footer. Monograph uses text, never a logo image. */
  name: "AI Like I'm Five",
  tagline: "AI research explained plainly",
  title: "AI Like I'm Five - AI papers explained plainly",
  description:
    "Explanations of AI research papers for everyone. No PhD required, no gatekeeping. Written by someone learning in public.",
  siteUrl: "https://ailikeimfive.com",
  authorName: "Richie Thomas",
  email: "",
  language: "en",
  dateLocale: "en-US",
  locale: "en_US",
  socialImage: "/og-image.png",
  /** Shown in the home sidebar "About" card. */
  about:
    "My learning log for working through 30papers.com, the reading list Ilya Sutskever reportedly gave John Carmack. Written first for an audience of me: I'm autistic with ADHD, and these notes are built to make sense to my brain. If they make sense to yours too, even better.",
  /**
   * Both forms below ship enabled with an empty `action`, which makes them fully
   * interactive demos that submit nowhere: a small script confirms the submit
   * and clears the fields. Paste your provider's endpoint into `action` to send
   * real submissions, or set `enabled: false` to disable the controls outright.
   */
  newsletter: {
    enabled: true,
    action: "https://buttondown.com/api/emails/embed-subscribe/ailikeimfive",
    method: "post",
    emailFieldName: "email",
    title: "Get new posts by email",
    description: "One email when a new explanation goes up. No spam, unsubscribe anytime.",
  },
  contact: {
    enabled: true,
    action: "",
    method: "post",
    responseTime: "I read everything and reply when I can.",
  },
  socials: [
    { label: "LinkedIn", href: "https://linkedin.com/in/heyrichie" },
    { label: "RSS", href: "/rss.xml" },
  ],
};

/** Header navigation. Add or remove entries freely; the header renders them in order. */
export const navigation = [
  { label: "Archive", href: "/posts/" },
  { label: "Categories", href: "/categories/" },
  { label: "Tags", href: "/tags/" },
  { label: "About", href: "/about/" },
];

/** Secondary navigation rendered in the footer. */
export const footerNavigation = [
  { label: "Contact", href: "/contact/" },
  { label: "Privacy", href: "/privacy/" },
  { label: "RSS", href: "/rss.xml" },
];
