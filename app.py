import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile

from minimal_self_full import MinimalSelf, MovingObstacle, SocialEntity, run_simulation, compute_phi

plt.switch_backend("Agg")  # ensure headless plotting

def run_agent(
    steps=500,
    epsilon=0.2,
    learning_rate=0.1,
    reward_type="original",
    obstacle=False,
    social=False,
    seed=123
):
    # Configure agent
    agent = MinimalSelf(
        seed=seed,
        epsilon=epsilon,
        learning_rate=learning_rate,
        body_bit_reinforce_factor=0.1,
        body_bit_decay_rate=0.01,
        reward_type=reward_type
    )

    # Optional entities
    entity_actions = [
        np.array([0, 1]), np.array([1, 0]), np.array([0, -1]), np.array([-1, 0])
    ]
    obs = None
    soc = None
    if obstacle:
        obs = MovingObstacle(start_pos=np.array([0, 0]), actions=entity_actions, seed=43)
    if social:
        soc = SocialEntity(start_pos=np.array([2, 2]), actions=entity_actions, seed=44)

    # Run simulation
    history = run_simulation(agent, steps, obstacle_instance=obs, social_entity_instance=soc)
    df = pd.DataFrame(history)

    # Compute final phi
    final_phi = compute_phi(history)

    # Plot metrics
    fig1, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    metrics = ["predictive_rate", "C_min", "body_bit_strength", "reward"]
    colors = ["#2b8", "#06c", "#a5a", "#e67"]
    for i, m in enumerate(metrics):
        if m in df.columns:
            axes[i].plot(df["t"], df[m], label=m, color=colors[i])
            axes[i].set_ylabel(m)
            axes[i].grid(True)
            axes[i].legend()
        else:
            axes[i].text(0.5, 0.5, f"{m} not available", transform=axes[i].transAxes, ha="center")
    axes[-1].set_xlabel("Time step")
    fig1.suptitle("Metrics over time")
    fig1.tight_layout()

    # Plot path
    fig2, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Agent and environment paths")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_xticks(np.arange(0, 3))
    ax.set_yticks(np.arange(0, 3))
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)

    # Agent path
    ax.plot([p[0] for p in df["position"]], [p[1] for p in df["position"]],
            marker="o", linestyle="-", color="blue", alpha=0.7, label="Agent")
    ax.scatter(df["position"].iloc[0][0], df["position"].iloc[0][1], color="cyan", s=80, label="Start")
    ax.scatter(df["position"].iloc[-1][0], df["position"].iloc[-1][1], color="navy", s=80, label="End")

    # Obstacle path
    if obstacle and "obstacle_position" in df.columns:
        ax.plot([p[0] for p in df["obstacle_position"]], [p[1] for p in df["obstacle_position"]],
                marker="x", linestyle="--", color="red", alpha=0.6, label="Obstacle")

    # Social entity path
    if social and "social_entity_position" in df.columns:
        ax.plot([p[0] for p in df["social_entity_position"]], [p[1] for p in df["social_entity_position"]],
                marker="^", linestyle=":", color="green", alpha=0.6, label="Social entity")

    ax.legend()

    # Save CSV to a temporary file and return path
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)
    tmp.close()

    return (
        gr.Plot(fig1),
        gr.Plot(fig2),
        f"{final_phi:.2f}",
        tmp.name  # return path, not bytes
    )

with gr.Blocks(title="RFT Minimal Self: 3×3 Agent") as demo:
    gr.Markdown("# RFT Minimal Self: 3×3 Agent")
    gr.Markdown("Run the 3×3 embodied agent with Q-learning, obstacles, and social mimicry. Visualize metrics, paths, and export results.")

    with gr.Row():
        steps = gr.Slider(100, 5000, value=500, step=50, label="Steps")
        epsilon = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Epsilon (exploration)")
        learning_rate = gr.Slider(0.0, 1.0, value=0.1, step=0.05, label="Learning rate")
        seed = gr.Number(value=123, label="Seed")

    reward_type = gr.Radio(choices=["original", "explore_grow", "social"], value="original", label="Reward type")
    obstacle = gr.Checkbox(value=False, label="Enable moving obstacle")
    social = gr.Checkbox(value=False, label="Enable social entity")

    run_btn = gr.Button("Run simulation")

    metrics_plot = gr.Plot(label="Metrics over time")
    path_plot = gr.Plot(label="Paths in 3×3 world")
    final_phi = gr.Textbox(label="Final Φ_min (toy measure)", interactive=False)
    csv_out = gr.File(label="Download results.csv", file_types=[".csv"])

    run_btn.click(
        fn=run_agent,
        inputs=[steps, epsilon, learning_rate, reward_type, obstacle, social, seed],
        outputs=[metrics_plot, path_plot, final_phi, csv_out]
    )

if __name__ == "__main__":
    demo.launch()
