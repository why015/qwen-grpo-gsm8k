# Evaluation Protocol (v2)

## 数据集角色

| 数据集 | 文件 | 条数 | 用途 |
|---|---|---|---|
| Training | train_clean.jsonl | 6973 | 训练 |
| Development | development500.jsonl | 500 | 所有实验、消融、失败分析 |
| Final Test | official_test.jsonl | 1319 | 配置冻结后只跑一次 |

## 纪律

1. 所有消融实验（Reward / beta / group_size / temperature）都在 development500 上做；
2. official_test 在所有设计冻结之前不允许运行；
3. 若使用 official_test，只允许对最终 1~2 个冻结配置各跑一次；
4. 所有评测统一 greedy decoding + 相同 verifier。

## 实验记录

- 2026-09-30: 原 dev200 + heldout300 合并为 development500。此前观察到
  GRPO v3 在 dev200 上 +2.5pp，在 heldout300 上 −2.0pp，未复现。
  为诚实起见，不再将 heldout300 视为 held-out，全部作为 development set。
