---
title: Welcome to Quartz - Color Preview
---

# Quartz Color Configuration Preview

This page showcases all the color configurations available in this Quartz repository using Obsidian-formatted markdown.

## Text Colors & Highlights

Regular text appears in the default dark color (`#2b2b2b`).

==This text is highlighted using the text highlight color (`#fff23688`)==

> [!info] Information Callout
> This callout uses the secondary color (`#284b63`) for its accent. Callouts are great for drawing attention to important information.

> [!tip] Tip Callout
> This callout uses the tertiary color (`#84a59d`) for its accent. Tips provide helpful suggestions.

> [!warning] Warning Callout
> Warning callouts help highlight potential issues or important considerations.

> [!example] Example Callout
> Example callouts are useful for demonstrating concepts or providing samples.

## Links and Interactive Elements

Here's a [[non-existent page]] wiki-link that will appear in gray (`#b8b8b8`).

Regular [markdown links](https://quartz.jzhao.xyz) use the secondary color (`#284b63`).

## Code Blocks

Inline code like `const color = "#284b63"` uses the darkgray color (`#4e4e4e`) for background.

```javascript
// Code blocks showcase syntax highlighting
const theme = {
  light: "#e8e6dd",
  lightgray: "#e5e5e5",
  gray: "#b8b8b8",
  darkgray: "#4e4e4e",
  dark: "#2b2b2b",
  secondary: "#284b63",
  tertiary: "#84a59d",
  highlight: "rgba(143, 159, 169, 0.15)",
  textHighlight: "#fff23688"
};

console.log("Colors configured!");
```

```bash
chmod +x executable
```

## Typography Showcase

### Headers use Schibsted Grotesk font

Body text uses Source Sans Pro font for optimal readability.

`Code uses IBM Plex Mono` for clear distinction.

## Lists and Formatting

- **Bold text** stands out
- *Italic text* for emphasis
- ~~Strikethrough~~ for deprecated content

### Task Lists
- [ ] Uncompleted task
- [x] Completed task using tertiary color

## Tables

| Color Variable | Hex Value | Usage |
|---------------|-----------|--------|
| light | #e8e6dd | Background color |
| lightgray | #e5e5e5 | Borders and dividers |
| gray | #b8b8b8 | Muted text |
| darkgray | #4e4e4e | Code backgrounds |
| dark | #2b2b2b | Main text color |
| secondary | #284b63 | Links and accents |
| tertiary | #84a59d | Secondary accents |

## Block Quotes

> Block quotes use the lightgray color (`#e5e5e5`) for their left border.
> 
> They're great for highlighting important passages or quotes.

## Horizontal Rules

The line below uses the lightgray color:

---

## Mathematical Expressions

Inline math: $E = mc^2$ 

Block math:
$$
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
$$

## Tags and Metadata

Tags like #color-preview #theme-configuration use the tertiary color.

---

*Note: This preview page demonstrates how the color scheme defined in `quartz.config.ts` is applied throughout the site. The background uses the light color (`#e8e6dd`), while various UI elements utilize the full color palette for visual hierarchy.*
