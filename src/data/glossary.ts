// Shared glossary for inline hover-definition tooltips (<Term k="...">word</Term>).
// Keep definitions short and plain (roughly one line). The key is a stable slug;
// the displayed word comes from the component's slot, so one entry covers
// singular/plural/variant spellings.

export const glossary: Record<string, string> = {
  // Fundamentals
  model: "the finished function: a picture goes in, a guess comes out.",
  weights: "the numbers inside the model that it adjusts as it learns.",
  bias: "a constant added to a score; it shifts the result up or down.",
  classification: "sorting an input into one of a fixed list of labels.",
  feature: "a measurable property of the input the model can use.",
  hyperparameter: "a setting you choose before training, not learned from data.",
  parameter: "a value the model learns on its own, like a weight.",

  // Early classifiers
  "nearest-neighbor": "label a new image by copying its most similar training image.",
  knn: "k-nearest neighbors: take a majority vote of the k closest matches.",
  "linear-classifier": "scores an image with a weighted sum of its pixel values.",
  score: "a number the classifier gives each class; higher means more likely.",

  // Loss and training
  loss: "a single number measuring how wrong the current guesses are.",
  "loss-function": "the rule that turns the guesses and answers into the loss number.",
  softmax: "turns raw scores into probabilities that add up to 1.",
  gradient: "the slope of the loss for each weight: which way, and how steeply.",
  "gradient-descent": "repeatedly nudge the weights downhill to shrink the loss.",
  "learning-rate": "how big a step to take on each weight update.",
  derivative: "the slope of a function: how its output changes as its input changes.",
  backpropagation: "the method that computes the gradient for every weight at once.",

  // Networks
  neuron: "one unit that weights its inputs, adds them up, and applies an activation.",
  layer: "a row of neurons that all read the same inputs.",
  "activation-function": "a simple bend applied to a neuron's output, like ReLU.",
  relu: "an activation that keeps positive values and sets negatives to zero.",
  width: "how many neurons are in a layer.",
  depth: "how many layers are stacked in the network.",

  // Data prep
  normalization: "putting input values onto a consistent, smaller scale.",
  standardization: "re-center each feature to average 0 and divide by its spread.",
  "standard-deviation": "how far a typical value sits from the average.",

  // Overfitting
  overfitting: "memorizing the training data instead of learning general patterns.",
  generalization: "how well a model does on images it has never seen.",
  regularization: "techniques that hold a model back so it generalizes better.",
  "weight-decay": "a penalty on large weights (also called L2 regularization).",
  dropout: "randomly switch off neurons during training so none becomes essential.",
  "early-stopping": "stop training once validation accuracy stops improving.",
  "data-augmentation": "make modified copies of training images to add variety.",
  "training-set": "the images the model learns from.",
  "validation-set": "held-out images used to compare designs and tune settings.",
  "test-set": "held-out images used once, at the end, for the final score.",

  // Convolutional networks
  convolution: "slide a small filter across the image to find a pattern anywhere.",
  filter: "a small grid of weights that detects one pattern (also called a kernel).",
  "feature-map": "the grid of responses a filter produces as it slides over the input.",
  "receptive-field": "the patch of the input that one output value depends on.",
  padding: "a border of zeros around the image so the output keeps its size.",
  pooling: "downsample a region to one value, shrinking the feature maps.",
  "max-pooling": "keep the largest value in each small region.",
  "fully-connected-layer": "a layer where every input connects to every neuron.",
  flatten: "unroll a stack of feature maps into one long list of numbers.",

  // Interpretability and transfer
  interpretability: "making sense of what a trained network is doing and why.",
  "saliency-map": "a heatmap of which pixels the score is most sensitive to.",
  "transfer-learning": "reuse a network trained on one task as a start for another.",
  "fine-tuning": "keep training the reused layers, gently, on your own data.",
  "feature-extraction": "freeze the reused layers and train only a new final layer.",
  imagenet: "a large public image dataset networks are commonly pre-trained on.",
  ilsvrc: "the ImageNet Large Scale Visual Recognition Challenge: the annual contest on a 1,000-category slice of ImageNet.",
  "top-5-error": "the fraction of images whose correct label is not among the model's five most confident guesses.",
  gpu: "a graphics card: hardware very fast at the parallel number-crunching that training a network needs.",
  "model-parallelism": "splitting one network across several devices so each holds and computes part of it.",
  "color-jitter": "randomly shifting an image's colors and brightness so a model learns to ignore lighting changes.",
  "local-response-normalization": "an early trick that dampened each activation based on nearby ones; later found unnecessary and dropped.",
  "overlapping-pooling": "pooling where the windows overlap, moving a smaller step than their own width.",
  momentum: "letting each weight update carry some speed from the previous ones, smoothing the path downhill.",
  epoch: "one full pass through the entire training set.",
};
