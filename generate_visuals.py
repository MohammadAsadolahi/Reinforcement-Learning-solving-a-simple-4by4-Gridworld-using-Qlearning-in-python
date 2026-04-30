"""
Advanced Visualization Suite for Q-Learning GridWorld
Generates publication-quality figures for the README.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
import os

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'font.family': 'sans-serif',
    'font.size': 12,
})

OUT = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(OUT, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# 1.  GridWorld Environment  (re-implement cleanly for data capture)
# ══════════════════════════════════════════════════════════════════════


class GridWorld:
    def __init__(self):
        self.actionSpace = ('U', 'D', 'L', 'R')
        self.actions = {
            (0, 0): ('D', 'R'),      (0, 1): ('L', 'D', 'R'),  (0, 2): ('L', 'D', 'R'),  (0, 3): ('L', 'D'),
            (1, 0): ('U', 'D', 'R'),  (1, 1): ('U', 'L', 'D', 'R'), (1, 2): ('U', 'L', 'D', 'R'), (1, 3): ('U', 'L', 'D'),
            (2, 0): ('U', 'D', 'R'),  (2, 1): ('U', 'L', 'D', 'R'), (2, 2): ('U', 'L', 'D', 'R'), (2, 3): ('U', 'L', 'D'),
            (3, 0): ('U', 'R'),      (3, 1): ('U', 'L', 'R'),  (3, 2): ('U', 'L', 'R'),
        }
        self.rewards = {(3, 3): 0.5, (1, 3): -0.5, (2, 1): -0.5, (3, 1): -0.5}
        self.explored = 0
        self.exploited = 0
        self.qTable = {
            s: {a: 0.0 for a in self.actions[s]} for s in self.actions}

    def reset(self):
        return (0, 0)

    def is_terminal(self, s):
        return s not in self.actions

    def getNewState(self, state, action):
        r, c = state
        if action == 'U':
            r -= 1
        elif action == 'D':
            r += 1
        elif action == 'L':
            c -= 1
        elif action == 'R':
            c += 1
        return (r, c)

    def chooseAction(self, state, policy, exploreRate=0.01):
        if exploreRate > np.random.rand():
            self.explored += 1
            return np.random.choice(self.actions[state])
        self.exploited += 1
        return policy[state]

    def move(self, state, policy, exploreRate):
        action = self.chooseAction(state, policy, exploreRate)
        ns = self.getNewState(state, action)
        reward = self.rewards.get(ns, -0.01)
        return action, ns, reward

    def getRandomPolicy(self):
        return {s: np.random.choice(self.actions[s]) for s in self.actions}


# ── Training ──────────────────────────────────────────────────────────
np.random.seed(42)
env = GridWorld()
policy = env.getRandomPolicy()
alpha = 0.1
gamma = 0.9
epsilon = 0.01
episodes = 2001

rewards_per_ep = []
avg_rewards = []
steps_per_ep = []
policy_snapshots = {}
qtable_snapshots = {}

for i in range(1, episodes + 1):
    state = env.reset()
    step = 0
    ep_reward = 0.0
    while not env.is_terminal(state) and step < 20:
        action, ns, reward = env.move(state, policy, epsilon)
        step += 1
        ep_reward += reward
        target = reward
        if not env.is_terminal(ns):
            target = reward + gamma * \
                env.qTable[ns][max(env.qTable[ns], key=env.qTable[ns].get)]
        env.qTable[state][action] += alpha * \
            (target - env.qTable[state][action])
        state = ns
    rewards_per_ep.append(ep_reward)
    avg_rewards.append(np.mean(rewards_per_ep))
    steps_per_ep.append(step)
    for s in policy:
        policy[s] = max(env.qTable[s], key=env.qTable[s].get)
    if (i - 1) % 200 == 0:
        policy_snapshots[i - 1] = dict(policy)
        qtable_snapshots[i - 1] = {s: dict(v) for s, v in env.qTable.items()}


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1 — GridWorld Environment Diagram
# ══════════════════════════════════════════════════════════════════════
def draw_gridworld():
    fig, ax = plt.subplots(figsize=(7, 7))
    cell_colors = {
        (0, 0): '#1f6feb',  # Start – blue
        (3, 3): '#3fb950',  # Goal – green
        (1, 3): '#f85149',  # Hole
        (2, 1): '#f85149',
        (3, 1): '#f85149',
    }
    labels = {(0, 0): 'S\nStart', (3, 3): 'T\nGoal',
              (1, 3): '✕\nHole', (2, 1): '✕\nHole', (3, 1): '✕\nHole'}
    reward_labels = {(3, 3): '+0.5', (1, 3): '−0.5',
                     (2, 1): '−0.5', (3, 1): '−0.5'}

    for r in range(4):
        for c in range(4):
            color = cell_colors.get((r, c), '#21262d')
            rect = plt.Rectangle((c, 3-r), 1, 1, facecolor=color,
                                 edgecolor='#30363d', linewidth=2)
            ax.add_patch(rect)
            if (r, c) in labels:
                ax.text(c+0.5, 3-r+0.55, labels[(r, c)],
                        ha='center', va='center', fontsize=13, fontweight='bold', color='white')
            else:
                ax.text(c+0.5, 3-r+0.5, f'({r},{c})', ha='center', va='center',
                        fontsize=9, color='#8b949e')
            if (r, c) in reward_labels:
                ax.text(c+0.5, 3-r+0.15, f'r = {reward_labels[(r,c)]}',
                        ha='center', va='center', fontsize=9, color='#e6edf3',
                        fontstyle='italic')

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_title('4×4 GridWorld Environment', fontsize=18,
                 fontweight='bold', pad=15, color='#e6edf3')

    legend_elements = [
        mpatches.Patch(facecolor='#1f6feb', label='Start (S)'),
        mpatches.Patch(facecolor='#3fb950', label='Goal (T)  r = +0.5'),
        mpatches.Patch(facecolor='#f85149', label='Hole (✕)  r = −0.5'),
        mpatches.Patch(facecolor='#21262d', edgecolor='#30363d',
                       label='Normal   r = −0.01'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02),
              ncol=2, fontsize=10, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "gridworld_env.png"),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("[✓] gridworld_env.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2 — Training Curves  (2-panel: total + average reward)
# ══════════════════════════════════════════════════════════════════════
def draw_training_curves():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

    # Total rewards
    ax1.plot(rewards_per_ep, color='#58a6ff', alpha=0.35,
             linewidth=0.6, label='Per-episode')
    window = 50
    smoothed = np.convolve(
        rewards_per_ep, np.ones(window)/window, mode='valid')
    ax1.plot(range(window-1, len(rewards_per_ep)), smoothed, color='#58a6ff', linewidth=2,
             label=f'{window}-ep moving avg')
    ax1.axhline(0, color='#484f58', linestyle='--', linewidth=0.8)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards During Training', fontsize=14,
                  fontweight='bold', color='#e6edf3')
    ax1.legend(frameon=False)

    # Average reward
    ax2.plot(avg_rewards, color='#3fb950', linewidth=2)
    ax2.axhline(avg_rewards[-1], color='#3fb950', linestyle=':', alpha=0.5)
    ax2.text(len(avg_rewards)*0.7, avg_rewards[-1]*1.15,
             f'converged ≈ {avg_rewards[-1]:.3f}', fontsize=10, color='#3fb950')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Cumulative Average Reward')
    ax2.set_title('Average Reward Convergence', fontsize=14,
                  fontweight='bold', color='#e6edf3')

    fig.suptitle('Q-Learning Training Performance', fontsize=17, fontweight='bold',
                 color='#e6edf3', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "training_curves.png"),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("[✓] training_curves.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3 — Optimal Policy Arrows on Grid
# ══════════════════════════════════════════════════════════════════════
def draw_policy_grid():
    arrow_map = {'U': (0, 0.25), 'D': (0, -0.25),
                 'L': (-0.25, 0), 'R': (0.25, 0)}
    fig, ax = plt.subplots(figsize=(7, 7))
    cell_colors = {(0, 0): '#1f6feb', (3, 3): '#3fb950',
                   (1, 3): '#f85149', (2, 1): '#f85149', (3, 1): '#f85149'}
    labels = {(0, 0): 'S', (3, 3): 'T', (1, 3): '✕', (2, 1): '✕', (3, 1): '✕'}

    for r in range(4):
        for c in range(4):
            color = cell_colors.get((r, c), '#21262d')
            rect = plt.Rectangle((c, 3-r), 1, 1, facecolor=color,
                                 edgecolor='#30363d', linewidth=2)
            ax.add_patch(rect)
            if (r, c) in labels:
                ax.text(c+0.5, 3-r+0.5, labels[(r, c)],
                        ha='center', va='center', fontsize=16, fontweight='bold', color='white')

    for s, a in policy.items():
        r, c = s
        dx, dy = arrow_map[a]
        ax.annotate('', xy=(c+0.5+dx, 3-r+0.5+dy), xytext=(c+0.5, 3-r+0.5),
                    arrowprops=dict(arrowstyle='->', color='#f0f6fc', lw=2.5))

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_title('Learned Optimal Policy (π*)', fontsize=18,
                 fontweight='bold', pad=15, color='#e6edf3')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "optimal_policy.png"),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("[✓] optimal_policy.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 4 — Q-Value Heatmap (max Q per state)
# ══════════════════════════════════════════════════════════════════════
def draw_qvalue_heatmap():
    grid = np.full((4, 4), np.nan)
    for s in env.qTable:
        r, c = s
        grid[r][c] = max(env.qTable[s].values())
    grid[3][3] = env.rewards[(3, 3)]

    fig, ax = plt.subplots(figsize=(7, 6.5))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'cyber', ['#f85149', '#161b22', '#3fb950'])
    im = ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=0.5)

    for r in range(4):
        for c in range(4):
            val = grid[r][c]
            if not np.isnan(val):
                color = 'white' if abs(val) > 0.15 else '#8b949e'
                ax.text(c, r, f'{val:.3f}', ha='center', va='center',
                        fontsize=12, fontweight='bold', color=color)
            else:
                ax.text(c, r, '—', ha='center', va='center',
                        fontsize=12, color='#484f58')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
    cbar.set_label('max Q(s, a)', fontsize=12)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(['Col 0', 'Col 1', 'Col 2', 'Col 3'])
    ax.set_yticklabels(['Row 0', 'Row 1', 'Row 2', 'Row 3'])
    ax.set_title('State-Value Heatmap  V(s) = max_a Q(s,a)', fontsize=15,
                 fontweight='bold', pad=12, color='#e6edf3')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "qvalue_heatmap.png"),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("[✓] qvalue_heatmap.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 5 — Policy Evolution Timeline
# ══════════════════════════════════════════════════════════════════════
def draw_policy_evolution():
    arrow_map = {'U': (0, 0.22), 'D': (0, -0.22),
                 'L': (-0.22, 0), 'R': (0.22, 0)}
    snapkeys = sorted(policy_snapshots.keys())[:6]
    fig, axes = plt.subplots(1, len(snapkeys), figsize=(3.3*len(snapkeys), 4))
    cell_colors = {(0, 0): '#1f6feb', (3, 3): '#3fb950',
                   (1, 3): '#f85149', (2, 1): '#f85149', (3, 1): '#f85149'}
    labels = {(0, 0): 'S', (3, 3): 'T', (1, 3): '✕', (2, 1): '✕', (3, 1): '✕'}

    for idx, ep in enumerate(snapkeys):
        ax = axes[idx]
        snap = policy_snapshots[ep]
        for r in range(4):
            for c in range(4):
                color = cell_colors.get((r, c), '#21262d')
                rect = plt.Rectangle((c, 3-r), 1, 1, facecolor=color,
                                     edgecolor='#30363d', linewidth=1.2)
                ax.add_patch(rect)
                if (r, c) in labels:
                    ax.text(c+0.5, 3-r+0.5, labels[(r, c)],
                            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        for s, a in snap.items():
            r, c = s
            dx, dy = arrow_map[a]
            ax.annotate('', xy=(c+0.5+dx, 3-r+0.5+dy), xytext=(c+0.5, 3-r+0.5),
                        arrowprops=dict(arrowstyle='->', color='#f0f6fc', lw=1.5))
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_title(f'Episode {ep}', fontsize=11,
                     fontweight='bold', color='#e6edf3')

    fig.suptitle('Policy Evolution During Training', fontsize=16, fontweight='bold',
                 color='#e6edf3', y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "policy_evolution.png"),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("[✓] policy_evolution.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 6 — Exploration vs Exploitation
# ══════════════════════════════════════════════════════════════════════
def draw_explore_exploit():
    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = [env.exploited, env.explored]
    colors = ['#58a6ff', '#f0883e']
    explode = (0.03, 0.06)
    wedges, texts, autotexts = ax.pie(
        sizes, labels=['Exploitation', 'Exploration'],
        autopct='%1.1f%%', startangle=140, colors=colors,
        explode=explode, textprops={'color': '#e6edf3', 'fontsize': 12},
        wedgeprops={'edgecolor': '#0d1117', 'linewidth': 2})
    for t in autotexts:
        t.set_fontweight('bold')
    ax.set_title('Exploration vs Exploitation Ratio', fontsize=16, fontweight='bold',
                 pad=15, color='#e6edf3')
    ax.text(0, -1.35, f'Total actions: {env.exploited+env.explored:,}   |   '
            f'ε = {epsilon}', ha='center', fontsize=10, color='#8b949e')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "explore_exploit.png"),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("[✓] explore_exploit.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 7 — Steps-to-Goal Convergence
# ══════════════════════════════════════════════════════════════════════
def draw_steps_convergence():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(steps_per_ep, color='#d2a8ff', alpha=0.3, linewidth=0.6)
    window = 50
    smoothed = np.convolve(steps_per_ep, np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, len(steps_per_ep)), smoothed, color='#d2a8ff', linewidth=2,
            label=f'{window}-ep moving avg')
    ax.axhline(y=6, color='#3fb950', linestyle='--',
               alpha=0.6, label='Optimal path length (6)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps to Terminal')
    ax.set_title('Steps per Episode — Convergence to Optimal Path', fontsize=14,
                 fontweight='bold', color='#e6edf3')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "steps_convergence.png"),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("[✓] steps_convergence.png")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 8 — Q-Table Detail (per-action Q values in each cell)
# ══════════════════════════════════════════════════════════════════════
def draw_qtable_detail():
    fig, ax = plt.subplots(figsize=(9, 9))
    cell_colors = {(0, 0): '#1f6feb', (3, 3): '#3fb950',
                   (1, 3): '#f85149', (2, 1): '#f85149', (3, 1): '#f85149'}

    for r in range(4):
        for c in range(4):
            color = cell_colors.get((r, c), '#21262d')
            rect = plt.Rectangle((c*2, (3-r)*2), 2, 2, facecolor=color,
                                 edgecolor='#30363d', linewidth=2, alpha=0.4)
            ax.add_patch(rect)

            if (r, c) in env.qTable:
                q = env.qTable[(r, c)]
                # Up
                if 'U' in q:
                    ax.text(c*2+1, (3-r)*2+1.65, f'↑{q["U"]:.3f}', ha='center', va='center',
                            fontsize=7.5, color='#79c0ff', fontweight='bold')
                # Down
                if 'D' in q:
                    ax.text(c*2+1, (3-r)*2+0.35, f'↓{q["D"]:.3f}', ha='center', va='center',
                            fontsize=7.5, color='#79c0ff', fontweight='bold')
                # Left
                if 'L' in q:
                    ax.text(c*2+0.3, (3-r)*2+1, f'←\n{q["L"]:.3f}', ha='center', va='center',
                            fontsize=7, color='#79c0ff', fontweight='bold')
                # Right
                if 'R' in q:
                    ax.text(c*2+1.7, (3-r)*2+1, f'→\n{q["R"]:.3f}', ha='center', va='center',
                            fontsize=7, color='#79c0ff', fontweight='bold')
            elif (r, c) == (3, 3):
                ax.text(c*2+1, (3-r)*2+1, 'GOAL\nr=+0.5', ha='center', va='center',
                        fontsize=11, fontweight='bold', color='white')

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_title('Converged Q-Table — Per-Action Values', fontsize=16,
                 fontweight='bold', pad=15, color='#e6edf3')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "qtable_detail.png"),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("[✓] qtable_detail.png")


# ── Run all ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    draw_gridworld()
    draw_training_curves()
    draw_policy_grid()
    draw_qvalue_heatmap()
    draw_policy_evolution()
    draw_explore_exploit()
    draw_steps_convergence()
    draw_qtable_detail()
    print("\n✅  All figures saved to ./assets/")
