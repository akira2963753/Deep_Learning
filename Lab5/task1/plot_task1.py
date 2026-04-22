import re
import matplotlib.pyplot as plt

LOG_PATH = "output.log"

eval_steps = []
eval_rewards = []
total_steps = []
total_rewards = []

with open(LOG_PATH, "r") as f:
    for line in f:
        # TrueEval: 每 20 episodes 的評估分數（對應作業要求的 Eval Reward）
        m = re.search(r'\[TrueEval\].*Eval Reward: ([\d.]+).*SC: (\d+)', line)
        if m:
            eval_rewards.append(float(m.group(1)))
            eval_steps.append(int(m.group(2)))

        # Total Reward: 每個 episode 的訓練分數
        m = re.search(r'\[Eval\] Ep: \d+ Total Reward: ([\d.]+) SC: (\d+)', line)
        if m:
            total_rewards.append(float(m.group(1)))
            total_steps.append(int(m.group(2)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("CartPole Vanilla DQN - Task 1", fontsize=14)

# 左圖：Total Reward（每個 episode）
axes[0].plot(total_steps, total_rewards, alpha=0.4, color="steelblue", linewidth=0.8)
axes[0].set_xlabel("Env Step Count")
axes[0].set_ylabel("Total Reward")
axes[0].set_title("Training Reward (per episode)")
axes[0].grid(True)

# 右圖：Eval Reward（每 20 episodes）
axes[1].plot(eval_steps, eval_rewards, color="tomato", linewidth=1.5, marker="o", markersize=3)
axes[1].axhline(y=480, color="green", linestyle="--", linewidth=1, label="Target (480)")
axes[1].set_xlabel("Env Step Count")
axes[1].set_ylabel("Eval Reward")
axes[1].set_title("Eval Reward (every 20 episodes)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("task1_curve.png", dpi=150)
plt.show()
print(f"Max Eval Reward : {max(eval_rewards):.1f}")
print(f"Final Eval Reward: {eval_rewards[-1]:.1f}")
print(f"Total Env Steps  : {eval_steps[-1]:,}")
