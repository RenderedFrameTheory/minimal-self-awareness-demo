"""
Minimal Self Agent (Rendered Frame Theory)
------------------------------------------
Implements reinforcement learning, exploration, and social mimicry in a 3×3 world.
Part of the artifact: Minimal Self in a 3×3 World
DOI: https://doi.org/10.5281/zenodo.17714387
"""

import numpy as np
import random
from typing import List, Optional
import pandas as pd
import matplotlib.pyplot as plt

# --- Classes ---

class SocialEntity:
    def __init__(self, start_pos: np.ndarray, actions: List[np.ndarray], bounds: tuple = (0, 2), seed: int = 44):
        random.seed(seed + 2)
        np.random.seed(seed + 2)
        self.pos = start_pos.astype(float)
        self.actions = actions
        self.bounds = bounds
        self.last_action = np.array([0, 0])

    def move(self):
        chosen_action = random.choice(self.actions)
        self.last_action = chosen_action.copy()
        self.pos = np.clip(self.pos + chosen_action, self.bounds[0], self.bounds[1])


class MovingObstacle:
    def __init__(self, start_pos: np.ndarray, actions: List[np.ndarray], bounds: tuple = (0, 2), seed: int = 42):
        random.seed(seed + 1)
        np.random.seed(seed + 1)
        self.pos = start_pos.astype(float)
        self.actions = actions
        self.bounds = bounds

    def move(self):
        chosen_action = random.choice(self.actions)
        self.pos = np.clip(self.pos + chosen_action, self.bounds[0], self.bounds[1])


class MinimalSelf:
    def __init__(self, seed: int = 42, error_window: int = 5, uncertainty_factor: float = 0.2,
                 initial_body_bit_strength: float = 1.0, body_bit_decay_rate: float = 0.01,
                 body_bit_reinforce_factor: float = 0.1,
                 learning_rate: float = 0.1, discount_factor: float = 0.9, epsilon: float = 0.2,
                 reward_type: str = "original"):

        random.seed(seed)
        np.random.seed(seed)

        # Embodied state
        self.pos = np.array([1, 1]).astype(float)
        self.body_bit_strength = initial_body_bit_strength
        self.body_bit_decay_rate = body_bit_decay_rate
        self.body_bit_reinforce_factor = body_bit_reinforce_factor

        # Actions
        self.actions = [
            np.array([0, 1]),  # N
            np.array([1, 0]),  # E
            np.array([0, -1]), # S
            np.array([-1, 0]), # W
        ]
        self.reverse_action_map = {i: a for i, a in enumerate(self.actions)}

        # Error tracking
        self.errors_history: List[float] = []
        self.error_window = error_window
        self.uncertainty_factor = uncertainty_factor

        # Environment
        self.env_bounds = (0, 2)
        self.obstacle = None
        self.social_entity = None

        # Q-learning
        self.q_table = np.zeros((self.env_bounds[1] + 1, self.env_bounds[1] + 1, len(self.actions)))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.prev_state = None
        self.prev_action_idx = None
        self.reward_type = reward_type

    def set_obstacle(self, obstacle: MovingObstacle):
        self.obstacle = obstacle

    def set_social_entity(self, social_entity: SocialEntity):
        self.social_entity = social_entity

    def sensory_input(self) -> np.ndarray:
        self.pos = np.clip(self.pos, self.env_bounds[0], self.env_bounds[1])
        sensation_vector = [self.pos[0], self.pos[1], self.body_bit_strength]
        if self.obstacle:
            sensation_vector.extend([self.obstacle.pos[0], self.obstacle.pos[1]])
        if self.social_entity:
            sensation_vector.extend([self.social_entity.pos[0], self.social_entity.pos[1],
                                     self.social_entity.last_action[0], self.social_entity.last_action[1]])
        return np.array(sensation_vector, dtype=float)

    def counterfactual_sensory(self, action: np.ndarray) -> np.ndarray:
        imagined_pos = np.clip(self.pos + action, self.env_bounds[0], self.env_bounds[1])
        counterfactual_vector = [imagined_pos[0], imagined_pos[1], self.body_bit_strength]
        if self.obstacle:
            counterfactual_vector.extend([self.obstacle.pos[0], self.obstacle.pos[1]])
        if self.social_entity:
            counterfactual_vector.extend([self.social_entity.pos[0], self.social_entity.pos[1],
                                          self.social_entity.last_action[0], self.social_entity.last_action[1]])
        return np.array(counterfactual_vector, dtype=float)

    def choose_action(self) -> np.ndarray:
        current_pos_int = tuple(self.pos.astype(int))
        if random.random() < self.epsilon:
            chosen_action_idx = random.randrange(len(self.actions))
        else:
            chosen_action_idx = np.argmax(self.q_table[current_pos_int])
        self.prev_state = current_pos_int
        self.prev_action_idx = chosen_action_idx
        return self.reverse_action_map[chosen_action_idx].copy()

    def step(self) -> dict:
        # Choose and apply action
        agent_chosen_action = self.choose_action()
        predicted = self.counterfactual_sensory(agent_chosen_action)
        self.pos += agent_chosen_action

        # Move environment entities
        if self.social_entity:
            self.social_entity.move()
        if self.obstacle:
            self.obstacle.move()

        # Actual sensation
        actual = self.sensory_input()

        # Prediction error and derived metrics
        prediction_error = float(np.linalg.norm(predicted[:2] - actual[:2]))
        self.errors_history.append(prediction_error)
        if len(self.errors_history) > self.error_window:
            self.errors_history.pop(0)

        mean_abs_error = float(np.mean(self.errors_history)) if self.errors_history else 0.0
        max_total_error = float(np.sqrt(8.0))  # max distance in 3x3 grid corners
        predictive_rate = 100.0 * (1.0 - (mean_abs_error / max_total_error)) if max_total_error > 0 else 100.0
        predictive_rate = float(np.clip(predictive_rate, 0.0, 100.0))
        c_min = (max_total_error - mean_abs_error) if max_total_error > 0 else 0.0

        # Body bit dynamics
        reinforcement = (predictive_rate / 100.0) * self.body_bit_reinforce_factor
        self.body_bit_strength += (reinforcement - self.body_bit_decay_rate)
        self.body_bit_strength = np.clip(self.body_bit_strength, 0.0, 2.0)

        # Reward shaping
        reward = (predictive_rate / 100.0) + (self.body_bit_strength / 2.0)

        # Optional reward variants
        if self.reward_type == "explore_grow":
            # Encourage new positions and stronger body bit
            reward += 0.2 * (self.body_bit_strength > 1.2)
        elif self.reward_type == "social" and self.social_entity is not None:
            # Reward alignment with social entity position
            align_bonus = 0.3 * (np.linalg.norm(self.pos - self.social_entity.pos) < 0.5)
            reward += align_bonus

        # Q-learning update
        if self.prev_state is not None and self.prev_action_idx is not None:
            current_pos_tuple = tuple(self.pos.astype(int))
            old_q_value = self.q_table[self.prev_state][self.prev_action_idx]
            next_max_q = np.max(self.q_table[current_pos_tuple])
            new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * next_max_q - old_q_value)
            self.q_table[self.prev_state][self.prev_action_idx] = new_q_value

        # Pack history
        record = {
            "t": None,  # filled in run_simulation
            "sensation": actual,
            "action": agent_chosen_action.copy(),
            "error": prediction_error,
            "position": self.pos.copy(),
            "predictive_rate": predictive_rate,
            "C_min": c_min,
            "body_bit_strength": self.body_bit_strength,
            "reward": reward
        }

        if self.obstacle is not None:
            record["obstacle_position"] = self.obstacle.pos.copy()
        if self.social_entity is not None:
            record["social_entity_position"] = self.social_entity.pos.copy()
            record["social_entity_last_action"] = self.social_entity.last_action.copy()

        return record

# --- Helper Functions ---

def compute_phi(history: List[dict]) -> float:
    if not history:
        return 0.0
    recent = history[-20:] if len(history) >= 20 else history
    positions = [tuple(h["sensation"][:2].astype(int)) for h in recent]
    body_bit_strengths = [h["sensation"][2] for h in recent]
    avg_body_bit_strength = np.mean(body_bit_strengths)
    unique_positions = set(positions)
    max_possible_unique_positions = min(len(recent), 9)
    position_diversity_score = len(unique_positions) / max_possible_unique_positions if max_possible_unique_positions > 0 else 0.0
    integrated_phi = avg_body_bit_strength * position_diversity_score
    return float(np.clip(integrated_phi, 0.0, 2.0))


def run_simulation(agent_instance: MinimalSelf, num_steps: int,
                   obstacle_instance: Optional[MovingObstacle] = None,
                   social_entity_instance: Optional[SocialEntity] = None) -> List[dict]:
    history: List[dict] = []
    if obstacle_instance:
        agent_instance.set_obstacle(obstacle_instance)
    if social_entity_instance:
        agent_instance.set_social_entity(social_entity_instance)

    for t in range(num_steps):
        hist = agent_instance.step()
        hist["t"] = t
        history.append(hist)

    return history


def plot_time_series(df: pd.DataFrame, title: str, metrics: List[str]):
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        if metric in df.columns:
            axes[i].plot(df['t'], df[metric], label=metric)
            axes[i].set_ylabel(metric)
            axes[i].legend()
            axes[i].grid(True)
        else:
            axes[i].set_ylabel(metric + ' (N/A)')
            axes[i].text(0.5, 0.5, f'{metric} not available', ha='center', va='center',
                         transform=axes[i].transAxes)
            axes[i].grid(True)

    axes[-1].set_xlabel("Time Step")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig


# --- Simulation Execution (optional CLI run) ---

if __name__ == "__main__":
    NUM_STEPS = 500
    all_histories = {}
    all_dataframes = {}

    entity_actions = [np.array([0, 1]), np.array([1, 0]), np.array([0, -1]), np.array([-1, 0])]

    print(f"\nRunning 'No Learning' Baseline for {NUM_STEPS} steps...")
    no_learning_agent = MinimalSelf(seed=123, initial_body_bit_strength=1.0,
                                    body_bit_decay_rate=0.0, body_bit_reinforce_factor=0.0,
                                    epsilon=0.0, learning_rate=0.0, reward_type="original")
    history_no_learning = run_simulation(no_learning_agent, NUM_STEPS)
    all_histories['no_learning'] = history_no_learning
    print("Baseline completed.")

    q_original_simple_agent = MinimalSelf(seed=123, epsilon=0.2, learning_rate=0.1,
                                          body_bit_reinforce_factor=0.1, body_bit_decay_rate=0.01,
                                          reward_type="original")
    all_histories['q_original_simple'] = run_simulation(q_original_simple_agent, NUM_STEPS)

    moving_obstacle = MovingObstacle(start_pos=np.array([0, 0]), actions=entity_actions, seed=43)
    q_original_complex_agent = MinimalSelf(seed=123, epsilon=0.2, learning_rate=0.1,
                                           body_bit_reinforce_factor=0.1, body_bit_decay_rate=0.01,
                                           reward_type="original")
    all_histories['q_original_complex'] = run_simulation(q_original_complex_agent, NUM_STEPS,
                                                         obstacle_instance=moving_obstacle)

    explore_grow_simple_agent = MinimalSelf(seed=123, epsilon=0.2, learning_rate=0.1,
                                            body_bit_reinforce_factor=0.1, body_bit_decay_rate=0.01,
                                            reward_type="explore_grow")
    all_histories['explore_grow_simple'] = run_simulation(explore_grow_simple_agent, NUM_STEPS)

    moving_obstacle2 = MovingObstacle(start_pos=np.array([0, 0]), actions=entity_actions, seed=43)
    explore_grow_complex_agent = MinimalSelf(seed=123, epsilon=0.2, learning_rate=0.1,
                                             body_bit_reinforce_factor=0.1, body_bit_decay_rate=0.01,
                                             reward_type="explore_grow")
    all_histories['explore_grow_complex'] = run_simulation(explore_grow_complex_agent, NUM_STEPS,
                                                           obstacle_instance=moving_obstacle2)

    social_entity_simple = SocialEntity(start_pos=np.array([2, 2]), actions=entity_actions, seed=44)
    q_social_simple_agent = MinimalSelf(seed=123, epsilon=0.2, learning_rate=0.1,
                                        body_bit_reinforce_factor=0.1, body_bit_decay_rate=0.01,
                                        reward_type="social")
    all_histories['q_social_simple'] = run_simulation(q_social_simple_agent, NUM_STEPS,
                                                      social_entity_instance=social_entity_simple)

    social_entity_complex = SocialEntity(start_pos=np.array([2, 2]), actions=entity_actions, seed=44)
    moving_obstacle3 = MovingObstacle(start_pos=np.array([0, 0]), actions=entity_actions, seed=43)
    q_social_complex_agent = MinimalSelf(seed=123, epsilon=0.2, learning_rate=0.1,
                                         body_bit_reinforce_factor=0.1, body_bit_decay_rate=0.01,
                                         reward_type="social")
    all_histories['q_social_complex'] = run_simulation(q_social_complex_agent, NUM_STEPS,
                                                       obstacle_instance=moving_obstacle3,
                                                       social_entity_instance=social_entity_complex)

    for name, history_list in all_histories.items():
        all_dataframes[f'df_{name}'] = pd.DataFrame(history_list)

    print("\n--- Average Metrics Comparison ---")
    metrics_for_avg = ['predictive_rate', 'C_min', 'body_bit_strength', 'reward']
    for name, df in all_dataframes.items():
        print(f"\n{name}:")
        existing_metrics = [m for m in metrics_for_avg if m in df.columns]
        print(df[existing_metrics].mean())

    print("\n--- Final Phi Values ---")
    for name, history_list in all_histories.items():
        final_phi = compute_phi(history_list)
        print(f"{name}: {final_phi:.2f}")

    metrics_for_plot = ['predictive_rate', 'C_min', 'body_bit_strength', 'reward']
    fig = plot_time_series(all_dataframes['df_q_original_simple'],
                           "Q-Learning Original Reward Simple Environment", metrics_for_plot)
    plt.show()
