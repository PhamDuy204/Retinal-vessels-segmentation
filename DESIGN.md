---
version: alpha
name: "SGMA-Net Vessel Viewer"
description: "A clinical image workspace for quick DRIVE vessel masks."
colors:
  ink: "#14283b"
  sea: "#146d86"
  line: "#d6e2e8"
  paper: "#f3f8fa"
  focus: "#50a9bd"
  canvas: "#e9f1f4"
typography:
  display: { fontFamily: "Georgia, serif" }
  body: { fontFamily: "system-ui, sans-serif" }
  mono: { fontFamily: "ui-monospace, monospace" }
rounded:
  md: "8px"
  lg: "12px"
spacing:
  page-max: "1200px"
  section-gap: "2.3rem"
components:
  upload: {}
  canvas: {}
  button: {}
---

# Vessel Viewer design

## Overview

### Creative North Star

A retinal imaging workbench: quiet cool surfaces, a pale image canvas, and clear distinction between original image, binary mask, and explanatory heatmap.

### Product context and register

- Audience: researchers inspecting SGMA-Net DRIVE checkpoints.
- Register: one-screen product utility in English.
- Signature: the prediction canvas changes in place to a green-on-black mask and then a green Grad-CAM overlay.
- Runtime mapping: Streamlit light theme lives in `.streamlit/config.toml`; preview CSS variables and component rules live in `demo.py`. Streamlit owns native select, upload, and button behavior.

## Colors

The light `paper` background, dark `ink` titles, and `sea` action color keep diagnostic imagery prominent. The pale `canvas` reserves a stable position for source and result previews; the downloaded 0/1 mask keeps its numerical values even though the screen displays green vessels on black. The pale red note under the result explains the first-run warmup. `focus` marks keyboard focus.

## Typography

Georgia marks the tool title and section headings. System sans is used for controls. Monospace labels distinguish image state and measurement.

## Layout

Two equal columns (source and output) on desktop; Streamlit stacks them on narrow screens. Source and result use the same 565:584 preview aspect ratio so choosing an image does not shift the controls unexpectedly. Upload sits beneath the source image.

## Elevation & Depth

Flat surfaces and a borderless upload button beneath the source image. No decorative shadows compete with the image.

## Shapes

8px image corners; 12px canvas and upload zone. Native Streamlit controls retain their accessible focus and keyboard behavior.
