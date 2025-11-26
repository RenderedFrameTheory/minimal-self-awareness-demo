# Minimal Self-Awareness Demo

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Minimal%20Self%20Awareness-blue)](https://huggingface.co/spaces/RFTSystems/minimal_self_awareness)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17714387.svg)](https://doi.org/10.5281/zenodo.17714387)
[![GitHub](https://img.shields.io/badge/GitHub-Source%20Code-black?logo=github)](https://github.com/yourusername/minimal-self-awareness-demo)

This repository contains a live Gradio simulation of the **Rendered Frame Theory (RFT) Minimal Self agent** in a 3×3 world.  
It demonstrates the **minimum requirements for self-awareness** through embodied reinforcement learning, exploration, obstacle stress, and social mimicry.

---

## 🔬 Overview

The demo implements seven experimental stages:

1. **Baseline** (no learning)  
2. **Q-learning** (simple and complex)  
3. **Exploration & growth** (simple and complex)  
4. **Social mimicry** (simple and complex)  

Each stage shows how an agent adapts to its environment, minimizes prediction error, and interacts socially — forming the **minimal requirements for self-awareness** in a computational model.

---

## ⚙️ Features

- Adjustable parameters:
  - Steps (up to 5000)
  - Epsilon (exploration rate)
  - Learning rate
  - Seed
- Reward types: `original`, `explore_grow`, `social`
- Toggles for moving obstacle and social entity
- Visual outputs:
  - Metrics over time (predictive rate, reward, C_min, body bit strength)
  - Agent paths in the 3×3 world
  - Final toy measure of awareness: **Φ_min**
- Exportable results (`results.csv`) for reproducible research

---

## 🌍 How to Access

- **Live Demo (interactive)**:  
  👉 [Minimal Self Awareness on Hugging Face Spaces](https://huggingface.co/spaces/RFTSystems/minimal_self_awareness)

- **Archived Record (citable)**:  
  👉 [Zenodo DOI: 10.5281/zenodo.17714387](https://doi.org/10.5281/zenodo.17714387)

- **Source Code (reproducible)**:  
  👉 [GitHub Repository](https://github.com/RenderedFrameTheory/minimal-self-awareness-demo)  **

---

## 📜 Citation

If you use this demo or manuscript, please cite:

> Grinstead, L. (2025). *Minimal Self in a 3×3 World: A Minimal Active Inference Agent with Bodily Self and Integrated Information Structure*. Zenodo. https://doi.org/10.5281/zenodo.17714387

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17714387.svg)](https://doi.org/10.5281/zenodo.17714387)

---

## 🚀 Quickstart (local run)

Clone the repo and install dependencies:

```bash
git clone https://github.com/RenderedFrameTheory/minimal-self-awareness-demo.git
cd minimal-self-awareness-demo
pip install -r requirements.txt
python app.py

🧩 Proof of Minimal Self
This demo shows how an embodied agent achieves:

Prediction error minimization

Reinforcement learning adaptation

Exploration under uncertainty

Social mimicry and interaction

Together, these form the minimal requirements for self-awareness in a computational model.
