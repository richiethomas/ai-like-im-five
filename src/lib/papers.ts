import { chapters, papers, type Chapter, type Paper, type PaperStatus } from "@/config/papers";
import { postSlug, visiblePosts, type Post } from "@/lib/posts";

export { chapters, papers, type Chapter, type Paper, type PaperStatus };

const matchesPaper = (paper: Paper, slug: string) => {
  const byPrefix = paper.slugPrefix ? slug.startsWith(paper.slugPrefix) : false;
  const byExtra = paper.extraPostSlugs?.includes(slug) ?? false;
  return byPrefix || byExtra;
};

/** Posts belonging to a paper, in reading order (oldest first). */
export const paperPosts = (paper: Paper, posts: Post[]): Post[] =>
  visiblePosts(posts)
    .filter((post) => matchesPaper(paper, postSlug(post)))
    .reverse();

export const paperHref = (paper: Paper) => `/papers/${paper.slug}/`;

/** Papers that have at least one published post, in reading order. */
export const papersWithPosts = (posts: Post[]) =>
  papers
    .map((paper) => ({ paper, posts: paperPosts(paper, posts) }))
    .filter((entry) => entry.posts.length > 0);

/** The paper a given post belongs to, if any. */
export const paperForPost = (post: Post): Paper | undefined => {
  const slug = postSlug(post);
  return papers.find((paper) => matchesPaper(paper, slug));
};

/** Chapters paired with the papers assigned to them, in config order. */
export const chaptersWithPapers = () =>
  chapters.map((chapter, index) => ({
    ...chapter,
    index: index + 1,
    papers: papers.filter((paper) => paper.chapter === chapter.key),
  }));

export const statusLabel: Record<PaperStatus, string> = {
  done: "Complete",
  next: "Up next",
  upcoming: "Planned",
};
