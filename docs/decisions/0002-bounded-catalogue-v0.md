# 0002: Bounded catalogue v0, six reformat-only formats

Date: 2026-10-31. Status: accepted.

## Decision

v0 = `shopping-list`, `todo-list`, `casual-email`, `business-email`, `blog-outline`, `calendar-json`
(`catalogue/formats.json`). The first four and the blog outline are Daniel's examples (2026-10-31). The to-do list
and calendar JSON were added to cover a checkbox list and a strict machine-readable format.

## Why bounded

A 0.6B model follows open-ended instructions poorly, but it can learn a handful of output shapes it has seen
hundreds of examples of. A fixed list also makes each format testable with a deterministic shape check.

## Selection rule

Formats only rearrange what the speaker said. Of the 206 prompts in `Text-Transformation-Prompts-300525`, the
ones that ask for new content (cover letters, op-eds, product descriptions, "infer the missing fields") are
excluded, because that is where a small model invents facts. The teacher prompts adapt the library versions
(`source_prompt` in the catalogue) and strip their inference instructions: the to-do list no longer guesses due
dates, and the calendar JSON leaves dates as spoken and leaves unstated fields null.

## Shapes are fixed

Each format has one layout (e.g. the outline is always `# title`, `## Introduction`, sections, `## Conclusion`), so
the adapter learns one target instead of averaging several. Changing a layout means re-teaching that format's pairs.
