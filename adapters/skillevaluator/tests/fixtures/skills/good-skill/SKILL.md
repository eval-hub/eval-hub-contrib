---
name: good-skill
description: Formats a markdown changelog from a list of commit subjects. Use when the user asks to generate or tidy release notes.
version: 1.0.0
metadata:
  author: EvalHub contributors
  tags: [changelog, release-notes]
---

# Changelog formatter

## Purpose

Turn a plain list of commit subjects into a grouped, readable markdown changelog.

## When to use

- The user pastes commit subjects and asks for release notes.
- The user asks to tidy an existing changelog section.

## Instructions

1. Group the subjects by conventional-commit type (`feat`, `fix`, `docs`, `chore`).
2. Write one bullet per subject, imperative mood, no trailing period.
3. Put breaking changes in their own section at the top.
4. Return only the markdown, with no extra commentary.

## Example

Input: `feat(api): add pagination`, `fix(ui): correct label`

Output:

```markdown
### Features
- add pagination (api)

### Fixes
- correct label (ui)
```

## Errors

If a subject does not follow the conventional-commit format, list it under "Other" and say so.
