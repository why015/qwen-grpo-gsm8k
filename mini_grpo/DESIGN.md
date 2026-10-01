# MiniGRPO 设计文档

## 1. 三种 loss 的数学定义

### 1.1 组内 relative advantage

GRPO / DAPO:
    A_i = (r_i - mean(r)) / (std(r) + eps)

Dr.GRPO:
    A_i = r_i - mean(r)

### 1.2 importance ratio

    rho_t = exp(logp_theta - logp_old)

### 1.3 KL 估计

    KL = logp_theta - logp_ref

### 1.4 GRPO 原始 loss (sequence-level)

    surr_i = min(rho_i * A_i, clip(rho_i, 1-eps, 1+eps) * A_i)
    policy_loss = - (1/G) * sum_i mean_t(surr_i_t)
    total = policy_loss + beta * mean(KL)

### 1.5 DAPO loss (token-level + 不对称 clipping)

    surr_i_t = min(rho_i_t * A_i, clip(rho_i_t, 1-eps_low, 1+eps_high) * A_i)
    policy_loss = - sum_i_t(surr_i_t) / sum_i_t(mask_i_t)
    eps_low = 0.2, eps_high = 0.28

### 1.6 Dr.GRPO loss

结构与 GRPO 相同，仅 advantage 不除 std。

## 2. 接口约定

### 2.1 张量 shape

    logp       : (B, T)  当前 policy per-token logprob
    old_logp   : (B, T)  生成时 logprob
    ref_logp   : (B, T)  reference policy logprob
    rewards    : (B,)    每个 completion 的 reward
    mask       : (B, T)  有效 token 1/0
    advantages : (B,)    每个 completion 的 advantage

B = batch_size * group_size

### 2.2 函数签名

    def compute_advantages(rewards, group_size, scale_by_std=True, eps=1e-4)
    def grpo_loss(logp, old_logp, ref_logp, advantages, mask, beta=0.04, epsilon=0.2)
    def dapo_loss(logp, old_logp, ref_logp, advantages, mask, beta=0.04, epsilon_low=0.2, epsilon_high=0.28)
    def dr_grpo_loss(logp, old_logp, ref_logp, advantages, mask, beta=0.04, epsilon=0.2)

## 3. 关键实现点

1. Mask 处理：logp * mask 保留有效 token；mean 时用 mask.sum() 做分母
2. KL 估计：logp - ref_logp（k3 无偏）；TRL 可能用 k1
3. 数值稳定：exp(logp - old_logp) 前减 max；advantage 加 eps
4. 聚合顺序：
   - GRPO: sequence_mean -> group_mean
   - DAPO: global_sum / global_mask_sum

## 4. 测试策略

1. Toy batch 手算
2. 边界情况（zero-variance group / 全 mask）
3. 数值稳定性
4. 与 TRL 对齐

## 5. 与 TRL 1.14 的已知差异（待 B4 验证）

- KL 估计器（k1 vs k3）
- ratio 聚合时机
- advantage 归一化时机
- clipping 与 KL 顺序
