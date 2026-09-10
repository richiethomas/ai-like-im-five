#!/usr/bin/env node

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const args = process.argv.slice(2);
if (args.length < 2) {
  console.log("Usage: node scripts/new-post.js <title> <category>");
  console.log("\nExample: node scripts/new-post.js 'Understanding Transformers' Transformers");
  process.exit(1);
}

const title = args[0];
const category = args[1];

// Slugify title for directory
const slug = title
  .toLowerCase()
  .replace(/[^a-z0-9]+/g, "-")
  .replace(/^-|-$/g, "");

const postDir = path.join(__dirname, "../src/content/posts", slug);

if (fs.existsSync(postDir)) {
  console.error(`✗ Post directory already exists: ${postDir}`);
  process.exit(1);
}

// Create directory
fs.mkdirSync(postDir, { recursive: true });

// Generate frontmatter with today's date
const today = new Date().toISOString().split("T")[0];
const frontmatter = `---
title: "${title}"
excerpt: ""
category: "${category}"
date: ${today}
author:
  name: "Richie Thomas"
  role: "Engineer"
featured: false
draft: true
---

`;

// Create index.mdx
fs.writeFileSync(path.join(postDir, "index.mdx"), frontmatter);

console.log(`✓ Created post: ${postDir}`);
console.log(`✓ Start editing: src/content/posts/${slug}/index.mdx`);
console.log(`✓ Set draft: false when ready to publish`);
