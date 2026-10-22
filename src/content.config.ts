import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";
import { categories } from "@/config/categories";

const posts = defineCollection({
  loader: glob({ pattern: "**/*.{md,mdx}", base: "./src/content/posts" }),
  schema: ({ image }) =>
    z.object({
      title: z.string(),
      excerpt: z.string(),
      /** Must match one of the entries in src/config/categories.ts. */
      category: z.enum(categories),
      date: z.coerce.date(),
      updatedDate: z.coerce.date().optional(),
      author: z.object({
        name: z.string(),
        role: z.string(),
      }),
      /**
       * Optional feature image. Monograph's post feeds are deliberately
       * text-only, so a cover is only ever shown on the post itself.
       */
      cover: z
        .object({
          src: image(),
          alt: z.string(),
          creditName: z.string().optional(),
          creditUrl: z.url().optional(),
        })
        .optional(),
      /** Surfaces the post in the "Featured" list in the home sidebar. */
      featured: z.boolean().default(false),
      draft: z.boolean().default(false),
      /**
       * Publication status, used mainly for Notes (posts outside a paper series):
       *   idea      - an outline or bullet points, not written yet
       *   draft     - being written and wordsmithed
       *   published - live on the site
       * Anything not "published" is left out of the production build entirely; on
       * the dev server it appears only on /notes/ and its own page, with a banner.
       */
      status: z.enum(["idea", "draft", "published"]).default("published"),
      /** Freeform topic tags for cross-cutting organization (e.g. "gradient descent", "CNNs"). */
      tags: z.array(z.string()).default([]),
    }),
});

export const collections = { posts };
