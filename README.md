# Minimal Self Awareness Demo — RFT Cognitive Core

A live demonstration of the Rendered Frame Theory (RFT) Minimal Self agent in a 3×3 world. Includes full Python implementation, Gradio interface, and seven experimental stages: baseline, Q‑learning, exploration, obstacle stress, and social mimicry. Visualizes metrics, paths, and exports results for reproducible research.

## Features
- 3×3 embodied agent with Q-learning
- Counterfactual prediction and toy coherence (`C_min`)
- Reward types: original, explore_grow, social
- Optional moving obstacle and social entity
- Live plots: metrics over time and agent paths
- CSV export of full timestep history

## Quickstart
```bash
pip install -r requirements.txt
python app.py

Seven stages
No learning baseline

Q-learning — original (simple)

Q-learning — original (complex with obstacle)

Explore & grow — simple

Explore & grow — complex with obstacle

Social — simple

Social — complex (obstacle + social)

Outputs
Metrics: predictive rate, C_min, body bit strength, reward

Final Φ_min (toy integrated binding)

Paths in 3×3 world (agent, obstacle, social)

Downloadable results.csv
