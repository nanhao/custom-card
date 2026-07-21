# 设计 #43 — 接口设计

## 1. _AtteV1Body — 新增方法

### 1.1 forward_pre_gru

```python
class _AtteV1Body(nn.Module):
    def forward_pre_gru(
        self,
        own_hand_ids: torch.Tensor,          # (batch, hand_len)
        own_discard_ids: torch.Tensor,        # (batch, discard_len)
        own_draw_ids: torch.Tensor,           # (batch, draw_len)
        own_last_action: torch.Tensor,        # (batch, 1) long
        own_status_ids: torch.Tensor,         # (batch, status_len)
        own_status_durations: torch.Tensor,   # (batch, status_len)
        own_opp_hand_ids: torch.Tensor,       # (batch, opp_hand_len)
        own_opp_discard_ids: torch.Tensor,    # (batch, opp_discard_len)
        own_opp_draw_ids: torch.Tensor,       # (batch, opp_draw_len)
        opp_hand_ids: torch.Tensor,
        opp_discard_ids: torch.Tensor,
        opp_draw_ids: torch.Tensor,
        opp_last_action: torch.Tensor,
        opp_status_ids: torch.Tensor,
        opp_status_durations: torch.Tensor,
        own_hp: torch.Tensor,                 # (batch, 1)
        own_energy: torch.Tensor,
        opp_hp: torch.Tensor,
        opp_energy: torch.Tensor,
        round_in_cycle: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """计算 GRU 之前的所有特征。

        Returns:
            gru_input: (batch, 80) — 供 GRU 消费
            intermediates: dict，包含 forward_post_gru 所需的中间结果：
                own_hand_emb2:   (batch, hand_len, 34)
                opp_hand_emb2:   (batch, opp_hand_len, 34)
                z3:              (batch, 34)
                own_la_emb:      (batch, 32)
                opp_la_emb:      (batch, 32)
                own_hp:          (batch, 1)
                own_energy:      (batch, 1)
                opp_hp:          (batch, 1)
                opp_energy:      (batch, 1)
                round_in_cycle:  (batch, 1)
        """
```

### 1.2 forward_post_gru

```python
class _AtteV1Body(nn.Module):
    def forward_post_gru(
        self,
        h_out: torch.Tensor,          # (batch, 32) — GRU 输出
        intermediates: dict,           # forward_pre_gru 返回的中间结果
    ) -> torch.Tensor:
        """计算 GRU 之后的所有特征 → 拼接 → FC → out。

        Returns:
            out: (batch, 64)
        """
```

### 1.3 forward 重构

`forward()` 保持签名不变，内部改为委托调用：

```
forward_pre_gru(...) → gru_input, intermediates
gru(gru_input.unsqueeze(0), hidden) → gru_out, new_h
forward_post_gru(gru_out.squeeze(0), intermediates) → out
```

## 2. AtteV1PolicyModel / AtteV1ValueModel — 不变

`forward()` 签名、返回值、行为完全不变。内部委托给 body 的新方法，外部调用方无需修改。

## 3. AtteV1TrainingRunner — 修改

### 3.1 新增方法

```python
class AtteV1TrainingRunner(BaseTrainingRunner):
    def _compute_all_pre_gru(
        self,
        perspective: dict,
        device: torch.device,
    ) -> list[tuple[torch.Tensor, dict]]:
        """对一条 perspective 的全部 step 调用 forward_pre_gru()。

        Returns:
            list of (gru_input, intermediates)，长度 = len(steps)
        """

    def _forward_value_sequence(
        self,
        pre_gru_results: list[tuple[torch.Tensor, dict]],
        value_model: nn.Module,
        device: torch.device,
    ) -> torch.Tensor:
        """批量 value 前向。

        1. 拼接全部 gru_input → (T, 1, 80)
        2. 调用 value_model.body.gru() 一次
        3. 循环调用 forward_post_gru() + value head → (T,) value tensor
        """

    def _forward_policy_sequence(
        self,
        pre_gru_results: list[tuple[torch.Tensor, dict]],
        policy_model: nn.Module,
        device: torch.device,
    ) -> torch.Tensor:
        """批量 policy 前向。

        1. 拼接全部 gru_input → (T, 1, 80)
        2. 调用 policy_model.body.gru() 一次
        3. 循环调用 forward_post_gru() + policy head → (T, 24) log_probs
        """
```

### 3.2 重写方法

```python
    def _process_value_perspective(
        self, perspective: dict, value_model: nn.Module,
        config: TrainingConfig, device: torch.device,
    ) -> torch.Tensor | None:
        """改为：_compute_all_pre_gru → _forward_value_sequence → MSE loss。"""

    def _recompute_advantages(
        self, perspectives: list[dict], value_model: nn.Module,
        config: TrainingConfig, device: torch.device,
    ) -> None:
        """改为：批量 GRU 前向。"""

    def _process_policy_perspective(
        self, perspective: dict, policy_model: nn.Module,
        config: TrainingConfig, device: torch.device,
    ) -> tuple | None:
        """改为：_compute_all_pre_gru → _forward_policy_sequence → PPO loss。"""
```
