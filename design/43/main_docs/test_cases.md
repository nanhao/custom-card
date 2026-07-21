# 设计 #43 — 测试用例集

## T1. _AtteV1Body.forward_pre_gru — 输出正确性

| ID | 场景 | 预期 |
|----|------|------|
| T1.1 | batch=1，正常手牌输入 | gru_input shape (1,80)，intermediates 包含全部 11 个 key |
| T1.2 | 空手牌（hand_ids=[]） | gru_input shape (1,80)，不抛异常 |
| T1.3 | 空状态（status_ids=[]） | gru_input shape (1,80)，不抛异常 |
| T1.4 | last_action=-1 | 正确映射为 PAD_ACTION_ID(24) |
| T1.5 | gru_input 数值与旧 forward() 中 GRU 输入一致 | 逐元素相等 |

## T2. _AtteV1Body.forward_post_gru — 输出正确性

| ID | 场景 | 预期 |
|----|------|------|
| T2.1 | 正常输入 | out shape (1,64) |
| T2.2 | 空手牌 | 不抛异常 |
| T2.3 | 空状态 | 不抛异常 |
| T2.4 | out 数值与旧 forward() 的 body 输出一致 | 逐元素相等 |

## T3. forward() 向后兼容

| ID | 场景 | 预期 |
|----|------|------|
| T3.1 | AtteV1PolicyModel.forward()，正常输入 | 返回 (probs, log_probs, new_hidden)，与原实现一致 |
| T3.2 | AtteV1ValueModel.forward()，正常输入 | 返回 (value, new_hidden)，与原实现一致 |
| T3.3 | forward 含 mask_fn / temperature | 行为不变 |
| T3.4 | forward 连续调用 3 步（hidden_state 传递） | 输出序列与原实现一致 |

## T4. AtteV1TrainingRunner._compute_all_pre_gru

| ID | 场景 | 预期 |
|----|------|------|
| T4.1 | perspective 含 5 个 step | 返回 list 长度 5 |
| T4.2 | 空 perspective（steps=[]） | 返回空 list |
| T4.3 | step 的 input 为 None | 跳过，对应位置返回 (zero_tensor, empty_dict) |

## T5. AtteV1TrainingRunner._forward_value_sequence

| ID | 场景 | 预期 |
|----|------|------|
| T5.1 | 5 个 step 的 pre_gru_results | 返回 (5,) value tensor |
| T5.2 | value 数值与原 _process_value_perspective 逐步计算一致 | 逐元素相等（允许浮点误差） |
| T5.3 | 含空 step | 对应位置 value 为 0.0 |

## T6. AtteV1TrainingRunner._forward_policy_sequence

| ID | 场景 | 预期 |
|----|------|------|
| T6.1 | 5 个 step 的 pre_gru_results | 返回 (5, 24) log_probs tensor |
| T6.2 | log_probs 逐行 softmax 和为 1 | ≈ 1.0 |
| T6.3 | log_probs 数值与原 _process_policy_perspective 逐步计算一致 | 逐元素相等（允许浮点误差） |

## T7. _process_value_perspective（重写后）

| ID | 场景 | 预期 |
|----|------|------|
| T7.1 | 正常 perspective，batch_size=1 | 返回 value_loss（非 NaN） |
| T7.2 | 空 perspective | 返回 None |
| T7.3 | loss 数值与原实现一致 | 相等（允许浮点误差） |

## T8. _recompute_advantages（重写后）

| ID | 场景 | 预期 |
|----|------|------|
| T8.1 | 正常 perspective | perspective["new_advantages"] 和 ["new_value_targets"] 写入 |
| T8.2 | advantages 数值与原实现一致 | 相等（允许浮点误差） |

## T9. _process_policy_perspective（重写后）

| ID | 场景 | 预期 |
|----|------|------|
| T9.1 | 正常 perspective | 返回 (policy_loss, entropy) |
| T9.2 | policy_loss 数值与原实现一致 | 相等（允许浮点误差） |
| T9.3 | entropy 数值与原实现一致 | 相等（允许浮点误差） |

## T10. 端到端训练

| ID | 场景 | 预期 |
|----|------|------|
| T10.1 | run() 完整三阶段训练 | 不抛异常，输出 policy.pth / value.pth / 损失曲线 |
| T10.2 | 使用真实样本文件训练 1 epoch | 完成，value_loss 和 policy_loss 非 NaN |
| T10.3 | 优化前后训练结果一致 | 相同随机种子下，输出权重文件逐字节相同 |

## T11. 性能回归

| ID | 场景 | 预期 |
|----|------|------|
| T11.1 | 单条 48 step perspective 的 _process_value_perspective 耗时 | < 300ms（优化前 ~1008ms） |
| T11.2 | 单条 48 step perspective 的 _process_policy_perspective 耗时 | < 300ms |
| T11.3 | 完整 run() 耗时（500 局样本，value_epoch=1，policy_epoch=1） | < 60s |
