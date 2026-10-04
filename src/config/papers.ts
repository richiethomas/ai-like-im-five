/**
 * The reading path through the 30papers.com list, grouped into chapters.
 *
 * This is the source of truth for the /papers index, the per-paper blogrolls,
 * and the homepage roadmap. A post is matched to its paper by `slugPrefix`
 * (the `paper-N-` prefix on its directory name), so posts can be split,
 * combined, or renamed without touching this file as long as the prefix holds.
 *
 * `status` drives how a paper is shown: "done" papers link to their blogroll,
 * "next" is highlighted as the one being written now, "upcoming" is a planned
 * entry with no posts yet. The order here is a planned reading path and can
 * change as the series goes on.
 */

export interface Chapter {
  key: string;
  title: string;
  blurb: string;
}

export const chapters: Chapter[] = [
  {
    key: "seeing",
    title: "Seeing",
    blurb: "How machines learn to recognize images: convolutional neural networks.",
  },
  {
    key: "remembering",
    title: "Remembering",
    blurb: "Networks that handle sequences and hold on to context: RNNs and LSTMs.",
  },
  {
    key: "attention",
    title: "Attention and Transformers",
    blurb: "The road to large language models, from the first attention mechanism onward.",
  },
  {
    key: "structure",
    title: "Memory, reasoning, structure",
    blurb: "Giving networks an external memory, relations, and graphs to reason over.",
  },
  {
    key: "scale",
    title: "Scale and systems",
    blurb: "What changes when the models, and the machines training them, get big.",
  },
  {
    key: "why",
    title: "The deep why",
    blurb: "Information, compression, and complexity: why learning works at all.",
  },
];

export type PaperStatus = "done" | "next" | "upcoming";

export interface Paper {
  /** Position in the reading path (1-27). */
  number: number;
  /** URL slug for the blogroll at /papers/<slug>/. */
  slug: string;
  /** Short label, e.g. "AlexNet". */
  title: string;
  /** Fuller title shown on the blogroll header. */
  full: string;
  year?: string;
  blurb: string;
  /** Matches a `Chapter.key`. */
  chapter: string;
  status: PaperStatus;
  /** Posts whose slug starts with this belong to the paper. */
  slugPrefix?: string;
  /** Extra post slugs to include (for asides that break the prefix pattern). */
  extraPostSlugs?: string[];
  /** Link to the paper or course itself. */
  sourceUrl?: string;
}

export const papers: Paper[] = [
  // Chapter 1 - Seeing
  {
    number: 1,
    slug: "cs231n",
    title: "CS231n",
    full: "CS231n: Convolutional Neural Networks for Visual Recognition",
    year: "Stanford course",
    blurb:
      "A Stanford course that builds computer vision from a single pixel up to deep convolutional networks.",
    chapter: "seeing",
    status: "done",
    slugPrefix: "paper-1-",
    extraPostSlugs: ["why-these-loss-formulas-look-the-way-they-do"],
    sourceUrl: "https://cs231n.github.io/",
  },
  {
    number: 2,
    slug: "alexnet",
    title: "AlexNet",
    full: "ImageNet Classification with Deep Convolutional Neural Networks",
    year: "2012",
    blurb: "The paper that proved deep CNNs work at scale and set off the deep-learning boom.",
    chapter: "seeing",
    status: "done",
    slugPrefix: "paper-2-",
    sourceUrl: "https://30papers.com/papers/alexnet/",
  },
  {
    number: 3,
    slug: "resnet",
    title: "ResNet",
    full: "Deep Residual Learning for Image Recognition",
    year: "2015",
    blurb: "Skip connections that let networks go hundreds of layers deep without breaking.",
    chapter: "seeing",
    status: "next",
    sourceUrl: "https://30papers.com/papers/deep-residual-learning/",
  },
  {
    number: 4,
    slug: "identity-mappings",
    title: "Identity Mappings",
    full: "Identity Mappings in Deep Residual Networks",
    year: "2016",
    blurb: "A closer look at why skip connections work as well as they do.",
    chapter: "seeing",
    status: "upcoming",
  },
  {
    number: 5,
    slug: "dilated-convolutions",
    title: "Dilated Convolutions",
    full: "Multi-Scale Context Aggregation by Dilated Convolutions",
    year: "2016",
    blurb: "Seeing wider context in an image without throwing away resolution.",
    chapter: "seeing",
    status: "upcoming",
  },

  // Chapter 2 - Remembering
  {
    number: 6,
    slug: "rnn-effectiveness",
    title: "RNNs",
    full: "The Unreasonable Effectiveness of Recurrent Neural Networks",
    year: "2015",
    blurb: "Networks that read and write sequences one step at a time.",
    chapter: "remembering",
    status: "upcoming",
  },
  {
    number: 7,
    slug: "understanding-lstms",
    title: "LSTMs",
    full: "Understanding LSTM Networks",
    year: "2015",
    blurb: "The gated memory cell that made recurrent networks practical.",
    chapter: "remembering",
    status: "upcoming",
  },
  {
    number: 8,
    slug: "rnn-regularization",
    title: "RNN Regularization",
    full: "Recurrent Neural Network Regularization",
    year: "2014",
    blurb: "Keeping big recurrent networks from memorizing instead of learning.",
    chapter: "remembering",
    status: "upcoming",
  },
  {
    number: 9,
    slug: "deep-speech-2",
    title: "Deep Speech 2",
    full: "Deep Speech 2: End-to-End Speech Recognition in English and Mandarin",
    year: "2015",
    blurb: "Turning raw audio into text with one big network.",
    chapter: "remembering",
    status: "upcoming",
  },

  // Chapter 3 - Attention and Transformers
  {
    number: 10,
    slug: "attention-nmt",
    title: "Attention",
    full: "Neural Machine Translation by Jointly Learning to Align and Translate",
    year: "2014",
    blurb: "The paper where the attention mechanism is born.",
    chapter: "attention",
    status: "upcoming",
  },
  {
    number: 11,
    slug: "pointer-networks",
    title: "Pointer Networks",
    full: "Pointer Networks",
    year: "2015",
    blurb: "Letting a network point back at positions in its own input.",
    chapter: "attention",
    status: "upcoming",
  },
  {
    number: 12,
    slug: "order-matters",
    title: "Order Matters",
    full: "Order Matters: Sequence to Sequence for Sets",
    year: "2015",
    blurb: "When the order you feed inputs and read outputs changes the answer.",
    chapter: "attention",
    status: "upcoming",
  },
  {
    number: 13,
    slug: "attention-is-all-you-need",
    title: "Transformer",
    full: "Attention Is All You Need",
    year: "2017",
    blurb: "The Transformer: the architecture behind modern large language models.",
    chapter: "attention",
    status: "upcoming",
  },
  {
    number: 14,
    slug: "annotated-transformer",
    title: "Annotated Transformer",
    full: "The Annotated Transformer",
    year: "2018",
    blurb: "The Transformer walked through line by line, in code.",
    chapter: "attention",
    status: "upcoming",
  },

  // Chapter 4 - Memory, reasoning, structure
  {
    number: 15,
    slug: "neural-turing-machines",
    title: "Neural Turing Machines",
    full: "Neural Turing Machines",
    year: "2014",
    blurb: "A network with an external memory it learns to read from and write to.",
    chapter: "structure",
    status: "upcoming",
  },
  {
    number: 16,
    slug: "relational-recurrent",
    title: "Relational Memory",
    full: "Relational Recurrent Neural Networks",
    year: "2018",
    blurb: "Memory that can relate its own stored pieces to each other.",
    chapter: "structure",
    status: "upcoming",
  },
  {
    number: 17,
    slug: "relation-networks",
    title: "Relational Reasoning",
    full: "A Simple Neural Network Module for Relational Reasoning",
    year: "2017",
    blurb: "A building block for reasoning about how objects relate.",
    chapter: "structure",
    status: "upcoming",
  },
  {
    number: 18,
    slug: "message-passing",
    title: "Graph Networks",
    full: "Neural Message Passing for Quantum Chemistry",
    year: "2017",
    blurb: "Learning on graphs by passing messages between connected nodes.",
    chapter: "structure",
    status: "upcoming",
  },

  // Chapter 5 - Scale and systems
  {
    number: 19,
    slug: "scaling-laws",
    title: "Scaling Laws",
    full: "Scaling Laws for Neural Language Models",
    year: "2020",
    blurb: "The math behind why bigger models keep getting better.",
    chapter: "scale",
    status: "upcoming",
  },
  {
    number: 20,
    slug: "gpipe",
    title: "GPipe",
    full: "GPipe: Efficient Training of Giant Neural Networks Using Pipeline Parallelism",
    year: "2018",
    blurb: "Training models too big to fit on a single chip.",
    chapter: "scale",
    status: "upcoming",
  },

  // Chapter 6 - The deep why
  {
    number: 21,
    slug: "variational-lossy-autoencoder",
    title: "VLAE",
    full: "Variational Lossy Autoencoder",
    year: "2016",
    blurb: "A bridge from practice to the theory of what a model should throw away.",
    chapter: "why",
    status: "upcoming",
  },
  {
    number: 22,
    slug: "keeping-nns-simple",
    title: "Keeping Networks Simple",
    full: "Keeping Neural Networks Simple by Minimizing the Description Length of the Weights",
    year: "1993",
    blurb: "Overfitting seen through the lens of information and compression.",
    chapter: "why",
    status: "upcoming",
  },
  {
    number: 23,
    slug: "mdl-tutorial",
    title: "MDL Principle",
    full: "A Tutorial Introduction to the Minimum Description Length Principle",
    year: "2004",
    blurb: "Learning as finding the shortest description of the data.",
    chapter: "why",
    status: "upcoming",
  },
  {
    number: 24,
    slug: "kolmogorov-complexity",
    title: "Kolmogorov Complexity",
    full: "Kolmogorov Complexity and Algorithmic Randomness",
    blurb: "The shortest program that could produce a given output.",
    chapter: "why",
    status: "upcoming",
  },
  {
    number: 25,
    slug: "complexodynamics",
    title: "Complexodynamics",
    full: "The First Law of Complexodynamics",
    year: "2011",
    blurb: "Why complexity tends to rise and then fall over time.",
    chapter: "why",
    status: "upcoming",
  },
  {
    number: 26,
    slug: "coffee-automaton",
    title: "Coffee Automaton",
    full: "Quantifying the Rise and Fall of Complexity in Closed Systems: The Coffee Automaton",
    year: "2014",
    blurb: "Measuring how complexity rises and falls, with a cup of coffee.",
    chapter: "why",
    status: "upcoming",
  },
  {
    number: 27,
    slug: "machine-super-intelligence",
    title: "Machine Super Intelligence",
    full: "Machine Super Intelligence",
    year: "2008",
    blurb: "A formal attempt to define intelligence itself. The capstone of the list.",
    chapter: "why",
    status: "upcoming",
  },
];
