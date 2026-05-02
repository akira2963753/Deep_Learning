# Lab 5 Demo Video — Full Script & Code Cues

> **Spec reminder (Spring 2026 Lab 5):** total length **5–6 minutes**; about **2 minutes** on **source code**, about **3 minutes** on **model performance**; **English** unless you have TA approval otherwise.  
> 下方 **Script (English)** 為建議口播逐字稿；**［螢幕］** 為你該開的檔案或畫面。可邊錄邊微調口語，不必每字相同。

---

## 0. Quick shot list

| Block | Time (approx.) | Content |
|-------|------------------|---------|
| Intro | 25–35 s | Who / what repo / three tasks |
| Code  | 1:45–2:15 | `DQN`, preprocessor, `train`, Task 3 flags |
| Demo  | 2:45–3:15 | Task 1 / 2 / 3 eval or clips + W&B |
| Outro | 15–25 s | What you learned + thanks |

---

## 1. Intro (English script)

**［螢幕］** IDE 或檔案總管顯示專案根目錄 `Lab5/`，可見 `task1/`、`task2/`、`task3/`、`report/`。

**Script:**

> Hi, this is our Lab 5 submission for NYCU Deep Learning — Value-Based Reinforcement Learning.  
> We implement DQN in PyTorch with Gymnasium: **Task 1** is CartPole with a small MLP; **Task 2 and 3** use the Arcade Learning Environment on **Pong**, with a CNN that takes **stacked 84-by-84 grayscale frames**.  
> **Task 3** adds **Double DQN**, **Prioritized Experience Replay**, and **multi-step returns** on top of that agent. Training is logged with **Weights & Biases**, and we save checkpoints at fixed environment-step milestones.  
> In the next two minutes I will walk through the code structure; then I will show the trained models.

---

## 2. Code walkthrough (~2 minutes)

### 2.1 Task 1 — MLP Q-network

**［螢幕］** 開啟 `task1/dqn_task1.py`，捲到 `class DQN` 與 `forward`。

**Script:**

> For **Task 1**, the state is a four-dimensional vector from CartPole. The Q-network is a simple MLP: two hidden layers with ReLU, and a linear output of size equal to the number of actions. The forward pass is just a forward through that `Sequential`.

**Code to show on screen (excerpt):**

```python
# task1/dqn_task1.py — DQN for CartPole
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.network(x)
```

**One-liner (optional):**

> Action selection uses **epsilon-greedy** in `select_action`, and the loss is **MSE on the TD target**, with a **replay buffer** and a **target network** updated every N optimizer steps — the standard DQN recipe from the lecture.

---

### 2.2 Task 2 & 3 — CNN + Atari preprocessing

**［螢幕］** 開啟 `task3/dqn_task3.py`（與作業繳交、測試腳本一致），顯示 `DQN` 的 Atari 分支與 `AtariPreprocessor`。

**Script:**

> For **Pong**, the observation is visual. We use the same CNN head as in the course starter: three convolution layers, then flatten, then two fully connected layers outputting one Q-value per action. In `forward`, raw pixel frames are scaled by **255** so the network sees values in roughly zero to one.

**Code to show (CNN + forward):**

```python
# task3/dqn_task3.py — DQN (Atari)
self.conv = nn.Sequential(
    nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=4, stride=2),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3, stride=1),
    nn.ReLU(),
    nn.Flatten(),
)
# conv_out_size from a dummy forward, then:
self.fc = nn.Sequential(
    nn.Linear(conv_out_size, 512),
    nn.ReLU(),
    nn.Linear(512, num_actions),
)

def forward(self, x):
    if self.is_atari:
        x = x / 255.0
        return self.fc(self.conv(x))
```

**Script (preprocessor):**

> **Preprocessing** follows the usual Atari pipeline: we take the **element-wise max** of the current and previous **raw** frame to reduce flicker, then **grayscale** and **resize** to 84 by 84, and we **stack four frames** so the network sees short-term motion.

**Code to show (preprocess core):**

```python
# task3/dqn_task3.py — AtariPreprocessor (idea)
def preprocess(self, obs):
    if self.last_raw_obs is not None:
        maxed = np.maximum(obs, self.last_raw_obs)
    else:
        maxed = obs
    self.last_raw_obs = obs
    return self._gray_resize(maxed)  # 84x84 grayscale
```

---

### 2.3 Training — TD target, Double DQN, PER

**［螢幕］** 同一檔案 `task3/dqn_task3.py`，捲到 `DQNAgent.train`。

**Script:**

> The **training step** samples a minibatch from replay. For the **bootstrap target**, vanilla DQN uses the **max** over the next state’s Q-values from the **target network**. **Double DQN** decouples selection and evaluation: the **online network** picks the greedy action, and the **target network** evaluates that action’s Q-value — that reduces overestimation.

**Code to show (DDQN + target + MSE / PER):**

```python
# task3/dqn_task3.py — train() core
q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

with torch.no_grad():
    if self.use_ddqn:
        next_actions  = self.q_net(next_states).argmax(1, keepdim=True)
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
    else:
        next_q_values = self.target_net(next_states).max(1)[0]

    gamma_n = self.gamma ** self.n_step
    target_q_values = rewards + gamma_n * next_q_values * (1 - dones)

td_errors = q_values - target_q_values

if self.use_per:
    loss = (per_weights * td_errors ** 2).mean()
    self.memory.update_priorities(per_indices, td_errors.detach().abs().cpu().numpy())
else:
    loss = nn.functional.mse_loss(q_values, target_q_values)
```

**Script:**

> The loss is **mean squared error** between the current Q and the TD target, matching the lecture formulation. With **PER**, we weight each sample by **importance-sampling weights** and update **priorities** from the absolute TD error. Every **target_update_frequency** steps we **copy** the online weights into the target network. We also **clip the gradient norm** to ten for stability.

---

### 2.4 Task 3 — flags and multi-env (short)

**［螢幕］** `task3/dqn_task3.py` 底部 `argparse`，或你實際下訓練指令的那一屏。

**Script:**

> **Task 3** is controlled by command-line flags: `--use-ddqn`, `--use-per`, and `--use-multistep` with `--num-envs` greater than one for **n-step** returns. For Pong we trained with **vectorized environments** so we collect experience faster; multi-step builds an **n-step return** before pushing transitions into the buffer when that flag is on.

**Optional one sentence if you used bonus code:**

> For the bonus section we also experimented with extra Rainbow-style components in a separate script; the main graded pipeline is the one in `task3/dqn_task3.py`.

---

### 2.5 Evaluation script (bridge to demo)

**［螢幕］** `task3/test_model_task3.py`。

**Script:**

> For grading we load the saved **state dict** into the **same** `DQN` class, set **eval** mode, run **greedy** `argmax` on Q-values, and average the episodic return over several seeds. We can optionally **record MP4** clips with `--record`.

**Code to show:**

```python
# task3/test_model_task3.py
model = DQN(input_shape, num_actions).to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
model.eval()
# ...
with torch.no_grad():
    action = model(state_tensor).argmax().item()
```

---

## 3. Model performance (~3 minutes)

**［螢幕］** 依序或可剪輯：W&B 專案頁（Task 1 / 2 / 3 曲線）、Pong 遊戲畫面錄屏、或終端機印出 average reward。

### 3.1 Task 1

**Script:**

> **Task 1:** Here is the learning curve for CartPole — reward versus environment steps. Our best checkpoint reaches a high average return over twenty evaluation episodes, above the homework threshold. The policy learns to balance the pole quickly.

**［螢幕］** W&B 圖表或終端 `eval` 數字。

---

### 3.2 Task 2 (vanilla DQN Pong)

**Script:**

> **Task 2** is **vanilla** DQN on Pong: CNN, replay, target network, epsilon decay. The eval score rises from random play toward competitive rallying; this run is only DDQN/PER/multi-step **off**, as required for the baseline comparison.

**［螢幕］** Task 2 的 W&B 或錄一段打球片段。

---

### 3.3 Task 3 (enhanced)

**Script:**

> **Task 3** enables **Double DQN**, **PER**, and **three-step** returns with multiple parallel environments. The curve reaches score **nineteen** faster than vanilla DQN — we report the **first timestep** where the moving average crosses nineteen, consistent with the report. Checkpoints at **six hundred thousand**, one million, and later steps are saved for submission.

**［螢幕］** Task 3 W&B（Eval Reward vs Env Step Count）、必要時秀出 `model_600000.pt` 或 `best_model.pt` 檔名與測試指令輸出。

**Example terminal (adjust paths to your machine):**

```bash
cd task3
python test_model_task3.py --model_path ./model/best_model.pt --episodes 20 --seed 0
```

Optional recording:

```bash
python test_model_task3.py --model_path ./model/best_model.pt --episodes 1 --record --output-dir ./eval_videos_task3
```

---

## 4. Closing (English)

**Script:**

> To summarize: we implemented **DQN** from low-dimensional control to **Atari Pong**, then added **Double DQN**, **prioritized replay**, and **multi-step** training for Task 3, with **W&B** logging and milestone checkpoints. Thank you for watching.

---

## 5. Timing cheat sheet (read while editing video)

| Timestamp | Section |
|-------------|---------|
| 0:00–0:35 | Intro |
| 0:35–1:05 | Task 1 MLP + epsilon-greedy mention |
| 1:05–1:40 | CNN + preprocessor + forward |
| 1:40–2:10 | `train`: DDQN, loss, PER, target copy |
| 2:10–2:25 | argparse / Task 3 flags + test script |
| 2:25–5:30 | W&B + game footage + terminal eval |
| 5:30–6:00 | Closing |

If you run **over six minutes**, cut duplicate W&B pans or shorten Task 2; keep Task 3 and code clarity.

---

## 6. File map (for your rehearsal)

| Topic | Primary file |
|-------|----------------|
| Task 1 | `task1/dqn_task1.py` |
| Task 2 / 3 network & train | `task3/dqn_task3.py` |
| Inference / grading script | `task3/test_model_task3.py` |
| Bonus (if you mention) | `bouns/dqn_bouns.py` |

---

*End of script. Good luck with the recording.*
