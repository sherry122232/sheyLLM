[跳转到正文](https://walkinglabs.github.io/hands-on-modern-rl/appendix_code_cheatsheet/grpo-rlvr#VPContent)

[![](https://walkinglabs.github.io/hands-on-modern-rl/readme/logo-symbol.svg)Hands on Modern RL](https://walkinglabs.github.io/hands-on-modern-rl/)

菜单

大纲

Sidebar Navigation

## 前言

[

课程导读

](https://walkinglabs.github.io/hands-on-modern-rl/preface/intro)

[

强化学习简史

](https://walkinglabs.github.io/hands-on-modern-rl/preface/brief-history/)

[

环境安装指南

](https://walkinglabs.github.io/hands-on-modern-rl/preface/env-setup)

## 基础导论

[

### 1. CartPole 倒立摆

](https://walkinglabs.github.io/hands-on-modern-rl/chapter01_cartpole/intro)

[

1.1 基本概念

](https://walkinglabs.github.io/hands-on-modern-rl/chapter01_cartpole/principles)

[

1.2 训练指标

](https://walkinglabs.github.io/hands-on-modern-rl/chapter01_cartpole/metrics)

[

### 2. DPO 偏好微调

](https://walkinglabs.github.io/hands-on-modern-rl/chapter02_dpo/intro)

[

2.1 DPO 推导

](https://walkinglabs.github.io/hands-on-modern-rl/chapter02_dpo/principles)

[

2.2 训练指标

](https://walkinglabs.github.io/hands-on-modern-rl/chapter02_dpo/metrics)

[

本篇小结

](https://walkinglabs.github.io/hands-on-modern-rl/summaries/part1-summary)

## 核心理论与方法

[

### 3. MDP 与价值函数

](https://walkinglabs.github.io/hands-on-modern-rl/chapter03_mdp/intro)

[

3.1 两台老虎机问题

](https://walkinglabs.github.io/hands-on-modern-rl/chapter03_mdp/bandit)

[

3.2 马尔可夫决策过程

](https://walkinglabs.github.io/hands-on-modern-rl/chapter03_mdp/mdp)

[

3.3 价值函数与贝尔曼方程

](https://walkinglabs.github.io/hands-on-modern-rl/chapter03_mdp/value-bellman)

[

3.4 DP、MC 与 TD

](https://walkinglabs.github.io/hands-on-modern-rl/chapter03_mdp/dp-mc-td)

[

3.5 从 Q 到 Q-Learning

](https://walkinglabs.github.io/hands-on-modern-rl/chapter03_mdp/value-q)

[

3.6 从价值到策略

](https://walkinglabs.github.io/hands-on-modern-rl/chapter03_mdp/policy-objective)

[

3.7 数据从哪里来

](https://walkinglabs.github.io/hands-on-modern-rl/chapter03_mdp/algorithm-taxonomy)

[

3.8 奖励函数设计

](https://walkinglabs.github.io/hands-on-modern-rl/chapter03_mdp/reward-design)

[

3.9 本章总结

](https://walkinglabs.github.io/hands-on-modern-rl/chapter03_mdp/panorama)

[

### 4. 深度 Q 网络

](https://walkinglabs.github.io/hands-on-modern-rl/chapter04_dqn/intro)

[

4.1 DQN 的必要性

](https://walkinglabs.github.io/hands-on-modern-rl/chapter04_dqn/from-q-to-dqn)

[

4.2 DQN 的结构

](https://walkinglabs.github.io/hands-on-modern-rl/chapter04_dqn/dqn-components)

[

4.3 动手：LunarLander 实战

](https://walkinglabs.github.io/hands-on-modern-rl/chapter04_dqn/lunar-lander)

[

4.4 DQN 改进家族

](https://walkinglabs.github.io/hands-on-modern-rl/chapter04_dqn/dqn-family)

[

4.5 动手：视觉游戏项目

](https://walkinglabs.github.io/hands-on-modern-rl/chapter04_dqn/visual-game-projects)

[

### 5. Policy-Based 方法

](https://walkinglabs.github.io/hands-on-modern-rl/chapter05_policy_gradient/intro)

[

5.1 为什么需要策略梯度

](https://walkinglabs.github.io/hands-on-modern-rl/chapter05_policy_gradient/pg-necessity)

[

5.2 策略梯度定理与 REINFORCE

](https://walkinglabs.github.io/hands-on-modern-rl/chapter05_policy_gradient/reinforce)

[

5.3 动手：策略梯度实战 CartPole

](https://walkinglabs.github.io/hands-on-modern-rl/chapter05_policy_gradient/cartpole)

[

5.4 策略梯度的方差与基线

](https://walkinglabs.github.io/hands-on-modern-rl/chapter05_policy_gradient/pg-improvements)

[

5.5 动手：带基线的策略梯度

](https://walkinglabs.github.io/hands-on-modern-rl/chapter05_policy_gradient/cartpole-baseline)

[

### 6. Actor-Critic

](https://walkinglabs.github.io/hands-on-modern-rl/chapter06_actor_critic/intro)

[

6.1 优势函数

](https://walkinglabs.github.io/hands-on-modern-rl/chapter06_actor_critic/advantage-function)

[

6.2 Critic 训练

](https://walkinglabs.github.io/hands-on-modern-rl/chapter06_actor_critic/critic-training)

[

6.3 Actor-Critic 架构

](https://walkinglabs.github.io/hands-on-modern-rl/chapter06_actor_critic/actor-critic)

[

6.4 动手：Pendulum 摆杆平衡

](https://walkinglabs.github.io/hands-on-modern-rl/chapter06_actor_critic/pendulum)

[

6.5 动手：BipedalWalker 双足行走

](https://walkinglabs.github.io/hands-on-modern-rl/chapter06_actor_critic/bipedalwalker)

[

6.6 Actor-Critic 的前沿大规模应用

](https://walkinglabs.github.io/hands-on-modern-rl/chapter06_actor_critic/ac-frontier)

[

### 7. PPO

](https://walkinglabs.github.io/hands-on-modern-rl/chapter07_ppo/intro)

[

7.1 动手：BipedalWalker 连续控制

](https://walkinglabs.github.io/hands-on-modern-rl/chapter07_ppo/ppo-bipedal-walker)

[

7.2 PPO 数学推导

](https://walkinglabs.github.io/hands-on-modern-rl/chapter07_ppo/ppo-math)

[

7.3 信任域与裁剪机制

](https://walkinglabs.github.io/hands-on-modern-rl/chapter07_ppo/trust-region-clipping)

[

7.4 GAE 与奖励模型

](https://walkinglabs.github.io/hands-on-modern-rl/chapter07_ppo/gae-reward-model)

[

7.5 PPO 游戏项目实践导论

](https://walkinglabs.github.io/hands-on-modern-rl/chapter07_ppo/ppo-game-benchmark)

[

7.6 长程任务中的 RL

](https://walkinglabs.github.io/hands-on-modern-rl/chapter07_ppo/rl-long-horizon-planning)

[

本篇小结

](https://walkinglabs.github.io/hands-on-modern-rl/summaries/part2-summary)

## 大模型 RL

[

### 8. RLHF 全流程

](https://walkinglabs.github.io/hands-on-modern-rl/chapter08_rlhf/intro)

[

8.1 Base 模型与对齐助手

](https://walkinglabs.github.io/hands-on-modern-rl/chapter08_rlhf/base-model-to-assistant)

[

8.2 RLHF 流水线

](https://walkinglabs.github.io/hands-on-modern-rl/chapter08_rlhf/standard-rlhf-pipeline)

[

8.3 SFT 指令微调

](https://walkinglabs.github.io/hands-on-modern-rl/chapter08_rlhf/imitation-learning-pipeline)

[

8.4 奖励模型

](https://walkinglabs.github.io/hands-on-modern-rl/chapter08_rlhf/reward-function-design)

[

8.5 PPO-RLHF 对齐

](https://walkinglabs.github.io/hands-on-modern-rl/chapter08_rlhf/ppo-rlhf-loop)

[

8.6 评估与奖励黑客

](https://walkinglabs.github.io/hands-on-modern-rl/chapter08_rlhf/evaluation)

[

8.7 动手：veRL PPO 训练 GSM8K

](https://walkinglabs.github.io/hands-on-modern-rl/chapter08_rlhf/verl-ppo-gsm8k)

[

8.8 扩展实战

](https://walkinglabs.github.io/hands-on-modern-rl/chapter08_rlhf/extended-practice)

[

### 9. 后训练对齐

](https://walkinglabs.github.io/hands-on-modern-rl/chapter09_alignment/intro)

[

9.1 DPO 原理、数学与选型

](https://walkinglabs.github.io/hands-on-modern-rl/chapter09_alignment/dpo-theory-and-family)

[

9.2 GRPO 训练与核心机制

](https://walkinglabs.github.io/hands-on-modern-rl/chapter09_grpo_rlvr/grpo-practice-and-mechanism)

[

9.3 R1-Zero 范式

](https://walkinglabs.github.io/hands-on-modern-rl/chapter09_grpo_rlvr/deepseek-dapo)

[

9.4 RLVR 可验证奖励

](https://walkinglabs.github.io/hands-on-modern-rl/chapter09_grpo_rlvr/rlvr)

[

9.5 动手：金融 API 工具调用 GRPO

](https://walkinglabs.github.io/hands-on-modern-rl/chapter09_grpo_rlvr/financial-tool-calling-grpo)

[

9.6 OPD 在线蒸馏

](https://walkinglabs.github.io/hands-on-modern-rl/chapter09_grpo_rlvr/on-policy-distillation)

[

9.7 动手：用 veRL 做代码生成强化学习

](https://walkinglabs.github.io/hands-on-modern-rl/chapter09_grpo_rlvr/verl-code-sandbox)

[

9.8 后训练工业实践

](https://walkinglabs.github.io/hands-on-modern-rl/chapter09_alignment/industrial-post-training)

[

### 10. Agentic RL

](https://walkinglabs.github.io/hands-on-modern-rl/chapter10_agentic_rl/intro)

[

10.1 多轮交互

](https://walkinglabs.github.io/hands-on-modern-rl/chapter10_agentic_rl/multi-turn-rl)

[

10.2 工具调用

](https://walkinglabs.github.io/hands-on-modern-rl/chapter10_agentic_rl/tool-use-and-trajectory)

[

10.3 评测与案例

](https://walkinglabs.github.io/hands-on-modern-rl/chapter10_agentic_rl/industrial-evaluation)

[

10.4 动手：Agent 数据制造

](https://walkinglabs.github.io/hands-on-modern-rl/chapter10_agentic_rl/agent-data-swe-smith)

[

10.5 动手：rLLM DeepCoder

](https://walkinglabs.github.io/hands-on-modern-rl/chapter10_agentic_rl/rllm-deepcoder-lab)

[

10.6 动手：金融问答 Agent

](https://walkinglabs.github.io/hands-on-modern-rl/chapter10_agentic_rl/rllm-finqa-lab)

[

10.7 Deep Research

](https://walkinglabs.github.io/hands-on-modern-rl/chapter10_agentic_rl/deep-research-agent)

[

10.8 Agentic RL 训练系统

](https://walkinglabs.github.io/hands-on-modern-rl/chapter10_agentic_rl/build-agentic-training-system)

[

10.9 延伸阅读

](https://walkinglabs.github.io/hands-on-modern-rl/chapter10_agentic_rl/extended-readings)

[

本篇小结

](https://walkinglabs.github.io/hands-on-modern-rl/summaries/part3-summary)

## 前沿

[

### 11. VLM 强化学习

](https://walkinglabs.github.io/hands-on-modern-rl/chapter11_vlm_rl/intro)

[

11.1 VLM 强化训练

](https://walkinglabs.github.io/hands-on-modern-rl/chapter11_vlm_rl/vlm-grpo-hands-on)

[

11.2 视觉奖励信号

](https://walkinglabs.github.io/hands-on-modern-rl/chapter11_vlm_rl/vlm-challenges)

[

11.3 VLM RL 推理框架

](https://walkinglabs.github.io/hands-on-modern-rl/chapter11_vlm_rl/vlm-frameworks)

[

11.4 视觉生成 RL

](https://walkinglabs.github.io/hands-on-modern-rl/chapter11_vlm_rl/visual-generation-rl)

[

11.5 动手：GeoQA 几何推理

](https://walkinglabs.github.io/hands-on-modern-rl/chapter11_vlm_rl/easyr1-geoqa)

[

### 12. 未来趋势

](https://walkinglabs.github.io/hands-on-modern-rl/chapter12_future_trends/intro)

[

12.1 具身智能

](https://walkinglabs.github.io/hands-on-modern-rl/chapter12_future_trends/embodied-intelligence/)

[

12.2 模型式强化学习

](https://walkinglabs.github.io/hands-on-modern-rl/chapter12_future_trends/embodied-intelligence/model-based-rl/)

[

12.3 自我博弈

](https://walkinglabs.github.io/hands-on-modern-rl/chapter12_future_trends/self-play-outlook/)

[

12.4 多智能体

](https://walkinglabs.github.io/hands-on-modern-rl/chapter12_future_trends/llm-multi-agent-rl/)

[

12.5 离线强化学习

](https://walkinglabs.github.io/hands-on-modern-rl/chapter12_future_trends/offline-rl/)

[

12.6 规模化趋势

](https://walkinglabs.github.io/hands-on-modern-rl/chapter12_future_trends/rl-scaling-outlook)

[

本篇小结

](https://walkinglabs.github.io/hands-on-modern-rl/summaries/part4-summary)

## 附录

[

A. 训练调试指南

](https://walkinglabs.github.io/hands-on-modern-rl/appendix_common_pitfalls/intro)

[

### B. RL 工程实践

](https://walkinglabs.github.io/hands-on-modern-rl/appendix_industrial_training/intro)

[

B.1 训练系统底座

](https://walkinglabs.github.io/hands-on-modern-rl/appendix_industrial_training/rl-infrastructure)

[

B.2 Agent 沙箱

](https://walkinglabs.github.io/hands-on-modern-rl/appendix_industrial_training/agentic-rl-infra)

[

B.3 评测基准

](https://walkinglabs.github.io/hands-on-modern-rl/appendix_industrial_training/evaluation-badcase)

[

B.4 训练指标词典

](https://walkinglabs.github.io/hands-on-modern-rl/appendix_industrial_training/metrics-glossary)

[

B.5 工业实战练习

](https://walkinglabs.github.io/hands-on-modern-rl/appendix_industrial_training/industrial-exercises)

[

### C. 手写代码速记

](https://walkinglabs.github.io/hands-on-modern-rl/appendix_code_cheatsheet/intro)

[

C.1 SFT 与 KL

](https://walkinglabs.github.io/hands-on-modern-rl/appendix_code_cheatsheet/sft-kl)

[

C.2 PPO 与 GAE

](https://walkinglabs.github.io/hands-on-modern-rl/appendix_code_cheatsheet/ppo-gae)

[

C.3 DPO 家族

](https://walkinglabs.github.io/hands-on-modern-rl/appendix_code_cheatsheet/dpo-family)

[

C.4 GRPO 与奖励模型

](https://walkinglabs.github.io/hands-on-modern-rl/appendix_code_cheatsheet/grpo-rlvr)

[

C.5 Softmax 与交叉熵

](https://walkinglabs.github.io/hands-on-modern-rl/appendix_code_cheatsheet/softmax-ce)

[

C.6 采样方法

](https://walkinglabs.github.io/hands-on-modern-rl/appendix_code_cheatsheet/top-k-top-p)

[

C.7 注意力机制

](https://walkinglabs.github.io/hands-on-modern-rl/appendix_code_cheatsheet/attention-mha)

[

C.8 DAPO

](https://walkinglabs.github.io/hands-on-modern-rl/appendix_code_cheatsheet/dapo)

[

D. 学习资料与复现项目推荐

](https://walkinglabs.github.io/hands-on-modern-rl/appendix_game_projects/intro)

[

### E. 强化学习的数学基础

](https://walkinglabs.github.io/hands-on-modern-rl/appendix_math/intro)

[

#### E.1 数学对象与线性代数

](https://walkinglabs.github.io/hands-on-modern-rl/appendix_math/linear-algebra)

[

#### E.2 概率、期望与随机估计

](https://walkinglabs.github.io/hands-on-modern-rl/appendix_math/probability-statistics)

[

#### E.3 微积分与优化

](https://walkinglabs.github.io/hands-on-modern-rl/appendix_math/calculus-optimization)

[

#### E.4 信息论与分布距离

](https://walkinglabs.github.io/hands-on-modern-rl/appendix_math/information-theory)

[](https://github.com/walkinglabs/hands-on-modern-rl "GitHub")

# C.4 GRPO 与 Reward Model

## GRPO Loss

### 一句话记忆

> **同 prompt 采 G 条回答，reward 在组内做 z-score 归一化当 advantage，然后套 PPO 的 clipped loss，再加 KL 惩罚。没有 Critic。**

### 伪代码

```
# 1. 同一 prompt 采样 G 条 completion
completions = [generate(prompt) for _ in range(G)]

# 2. 对每条打分
rewards = [reward_fn(c) for c in completions]   # [G]

# 3. 组内归一化 → advantage
advantages = (rewards - mean(rewards)) / (std(rewards) + eps)  # [G]

# 4. PPO clipped loss（用组内 advantage）
ratio = exp(new_logp - old_logp)
surr1 = ratio * advantages
surr2 = clip(ratio, 1-eps, 1+eps) * advantages
policy_loss = -min(surr1, surr2).mean()

# 5. KL 惩罚（相对 reference model）
kl = kl_penalty(log_probs, ref_log_probs)

# 6. 总 loss
loss = policy_loss + kl_coeff * kl
```

### 记忆方法

GRPO = **G**roup **R**elative **P**olicy **O**ptimization。和 PPO 的对比：

||PPO|GRPO|
|---|---|---|
|Advantage 来源|Critic 预测 V(s)V(s) → GAE|组内 reward 归一化|
|需要几个模型|4 个（actor, critic, ref, rm）|2~3 个（actor, ref, rm/verifier）|
|KL|可选|几乎必加|
|采样方式|单条 rollout|同 prompt 采 G 条|

口诀：**"PPO 砍掉 Critic，换成组内 z-score，其余照抄"**

### Python 实现

```
import numpy as np

def grpo_advantages(rewards):
    """
    rewards: [num_prompts, G]  每个 prompt 的 G 条回答的 reward
    """
    mean = rewards.mean(axis=1, keepdims=True)
    std = rewards.std(axis=1, keepdims=True)
    return (rewards - mean) / (std + 1e-8)

def grpo_policy_loss(new_logps, old_logps, advantages, clip_eps=0.2):
    """和 PPO clipped loss 完全相同"""
    ratio = np.exp(new_logps - old_logps)
    surr1 = ratio * advantages
    surr2 = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -np.minimum(surr1, surr2).mean()
```

### PyTorch 实现

```
import torch
import torch.nn.functional as F

def grpo_loss(log_probs, old_log_probs, ref_log_probs,
              rewards, clip_eps=0.2, kl_coeff=0.05):
    """
    log_probs:     [B, G, seq_len]  当前策略
    old_log_probs: [B, G, seq_len]  采样时策略
    ref_log_probs: [B, G, seq_len]  参考策略
    rewards:       [B, G]           组内 reward
    B = num_prompts, G = group_size
    """
    B, G = rewards.shape

    # 1. 组内归一化
    advantages = (rewards - rewards.mean(dim=1, keepdim=True)) \
                 / (rewards.std(dim=1, keepdim=True) + 1e-8)
    # [B, G] → [B, G, 1] 以广播到 seq_len 维度
    advantages = advantages.unsqueeze(-1)

    # 2. 序列级 log_prob 求和（每条 completion）
    # 假设 log_probs 已按有效 token 求和: [B, G]
    seq_logp = log_probs.sum(dim=-1)       # [B, G]
    seq_old  = old_log_probs.sum(dim=-1)
    seq_ref  = ref_log_probs.sum(dim=-1)

    # 3. Clipped policy loss
    ratio = torch.exp(seq_logp - seq_old)
    adv = advantages.squeeze(-1)            # [B, G]
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2).mean()

    # 4. KL 惩罚
    log_ratio = seq_logp - seq_ref
    kl = (torch.exp(log_ratio) - 1 - log_ratio).mean()

    return policy_loss + kl_coeff * kl
```

## Reward Model（Bradley-Terry 模型）

### 一句话记忆

> **chosen 分数比 rejected 高 ⇒⇒ sigmoid 后接近 1 ⇒⇒ -log 接近 0。就一行：`-log_sigmoid(r_chosen - r_rejected)`。**

### 伪代码

```
r_w = reward_model(chosen_input)     # chosen 的 reward 标量
r_l = reward_model(rejected_input)   # rejected 的 reward 标量

loss = -log(sigmoid(r_w - r_l))
```

### 记忆方法

Bradley-Terry 模型假设人类偏好概率为：

P(yw≻yl)=σ(r(x,yw)−r(x,yl))P(yw​≻yl​)=σ(r(x,yw​)−r(x,yl​))

训练目标就是最大化这个概率的对数，等价于最小化 `-log_sigmoid(diff)`。

口诀：**"RM 训练就是 pairwise 交叉熵"**

### Python 实现

```
def log_sigmoid(x):
    return -np.logaddexp(0, -x)

def reward_model_loss(r_chosen, r_rejected):
    """r_chosen, r_rejected: [B]"""
    return -log_sigmoid(r_chosen - r_rejected).mean()
```

### PyTorch 实现

```
def reward_model_loss(r_chosen, r_rejected):
    """
    r_chosen:  [B]  reward model 对 chosen 的打分
    r_rejected: [B]  reward model 对 rejected 的打分
    """
    return -F.logsigmoid(r_chosen - r_rejected).mean()
```

## 面试追问：DPO 和 RLHF-PPO 的关系

面试官常问"DPO 相比 PPO 的优劣"，准备这个对比表：

|维度|PPO-RLHF|DPO|
|---|---|---|
|需要 Reward Model|是|否（隐式学习）|
|需要 Critic|是|否|
|需要 Reference Model|可选|必须|
|在线/离线|在线（需要采样）|离线（只用偏好数据）|
|训练成本|高（4 个模型）|低（2 个模型）|
|奖励黑客风险|有（RM 可被钻空子）|较低（无显式 RM）|
|理论最优性|更强（可以持续探索）|受限于离线数据质量|
|适用场景|大规模在线训练|偏好数据充足的场景|

## 易错点

|易错|说明|
|---|---|
|GRPO 的 advantage 是组内归一化|不是全局归一化，是**同一个 prompt** 的 G 条回答之间比较|
|GRPO 没有 value loss|没有 Critic，所以没有 value loss，这是和 PPO 的核心区别|
|Reward Model 要 detach|训练 RM 时 chosen/rejected 的 reward 都要参与梯度，但训练 policy 时 RM 要冻结|
|GRPO 的 KL 是对每条序列的|不是 token 级别，通常是对整条 completion 的 log_prob 求和后再算 KL|
|DPO 隐式学到了 RM|DPO 的 `log_ratio_w - log_ratio_l` 本质上就是隐式 reward 差值|
|G 的大小|通常 G=4~16，太小 advantage 估计噪声大，太大采样成本高|
|RLVR 场景|reward 来自规则验证器（如代码执行、数学答案检查），不是 RM 打分|
