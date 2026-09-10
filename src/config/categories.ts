/**
 * The site's categories. Every post belongs to exactly one of these, so keep the
 * list short — six is the practical ceiling before the sidebar stops reading as
 * a menu. Rename or replace entries here, then update the `category` value in
 * each post's frontmatter to match; the build fails on any mismatch.
 *
 * Order matters: it is the order used on the categories index and in the home
 * sidebar.
 */
export const categories = [
  "Attention",
  "Transformers",
  "Embeddings",
  "Applications",
  "Bias",
  "Alignment",
  "Responsible AI",
  "Vision",
  "NLP",
  "Reasoning",
  "Training",
  "Scaling",
  "Fine-tuning",
  "Prompting",
  "Memory",
  "Multimodal",
  "Evaluation",
  "Inference",
  "Efficiency",
  "News",
] as const;

export type Category = (typeof categories)[number];

export const categorySlug = (category: string) =>
  category
    .toLowerCase()
    .replace(/&/g, "and")
    .replace(/[^a-z0-9\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-");

/** One line per category, shown on its archive page and in listings. */
export const categoryDescriptions: Record<Category, string> = {
  Attention: "How transformers learn to focus on what matters in the input.",
  Transformers: "The architecture that powers modern large language models.",
  Embeddings: "Turning words, images, and data into vectors machines can understand.",
  Applications: "How AI is being used in the real world, right now.",
  Bias: "Where AI goes wrong, and why it matters.",
  Alignment: "Making AI systems do what we actually want them to do.",
  "Responsible AI": "Building AI thoughtfully, with ethics and safety in mind.",
  Vision: "Image models, computer vision, and how AI understands pictures.",
  NLP: "Natural language processing: how AI reads and generates text.",
  Reasoning: "Step-by-step thinking, chain-of-thought, and problem solving.",
  Training: "How models are trained, loss functions, and optimization.",
  Scaling: "Why bigger models are better, and what changes when you scale.",
  "Fine-tuning": "Adapting existing models to do specific things well.",
  Prompting: "Prompt engineering, in-context learning, and talking to AI.",
  Memory: "Long-term memory, retrieval, and extending what models remember.",
  Multimodal: "Models that work with text, images, audio, and more together.",
  Evaluation: "Testing AI, benchmarks, and measuring what models can actually do.",
  Inference: "How models generate output, token by token, under the hood.",
  Efficiency: "Making AI faster and cheaper: quantization, distillation, and optimization.",
  News: "Breaking developments and new papers in AI.",
};
