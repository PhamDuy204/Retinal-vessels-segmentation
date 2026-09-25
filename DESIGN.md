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
  canvas: "#07131d"
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

A retinal imaging workbench: quiet cool surfaces, a dark output canvas, and clear distinction between original image, binary mask, and explanatory heatmap.

### Product context and register

- Audience: researchers inspecting SGMA-Net DRIVE checkpoints.
- Register: one-screen product utility in English.
- Signature: the black prediction canvas changes in place to a mask and then a Grad-CAM overlay.
- Runtime mapping: values in this file are mirrored by the CSS variables and component rules in `demo.py`; Streamlit owns native select, upload, and button behavior.

## Colors

The light `paper` background, dark `ink` titles, and `sea` action color keep diagnostic imagery prominent. The `canvas` reserves a stable position for results, while `focus` is used for keyboard focus.

## Typography

Georgia marks the tool title and section headings. System sans is used for controls. Monospace labels distinguish image state and measurement.

## Layout

Two unequal columns (source and larger output) on desktop; Streamlit stacks columns on narrow screens. The empty canvas has a minimum height to reserve space before prediction.

## Elevation & Depth

Flat surfaces and a thin border around the upload area. No decorative shadows compete with the image.

## Shapes

8px image corners; 12px canvas and upload zone. Native Streamlit controls retain their accessible focus and keyboard behavior.
