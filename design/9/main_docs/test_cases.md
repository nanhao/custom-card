# Test Cases — Split Policy/Value Networks

Test cases use text and symbols only. No code. Organized by target behavior.

---

## 1. Model Architecture

### TC-MOD1: GRUPolicyModel has no value output
- **Given**: `GRUPolicyModel()` instantiated
- **When**: `model.forward(x, h)` called with valid tensors
- **Then**: returns 3-tuple (probs, log_probs, new_hidden); no value tensor

### TC-MOD2: GRUValueModel has no policy output
- **Given**: `GRUValueModel()` instantiated
- **When**: `model.forward(x, h)` called with valid tensors
- **Then**: returns 2-tuple (value, new_hidden); no probs/log_probs

### TC-MOD3: GRUValueModel has no tanh on output
- **Given**: `GRUValueModel()` instantiated, input that saturates tanh (>3)
- **When**: `model.forward(x, h)` called
- **Then**: returned value is unbounded (can be >1 or <-1)

### TC-MOD4: both models share same GRU architecture
- **Given**: `GRUPolicyModel()` and `GRUValueModel()` instantiated
- **When**: comparing GRU layer parameter names and shapes
- **Then**: both have `gru.weight_ih_l0` and `gru.weight_hh_l0`; both have `INITIAL_HIDDEN_STATE = [0.0] * 8`

### TC-MOD5: GRUPolicyModel accepts mask_fn and temperature
- **Given**: `GRUPolicyModel()` with valid mask_fn and temperature=0.5
- **When**: `model.forward(x, h, mask_fn=mask, temperature=0.5)` called
- **Then**: probs are masked for illegal actions; log_probs reflect temperature scaling

### TC-MOD6: GRUValueModel does not accept mask_fn or temperature
- **Given**: `GRUValueModel()` instantiated
- **When**: checking `forward()` signature
- **Then**: signature is `forward(x, hidden)` only; no mask_fn, no temperature parameters

---

## 2. Registry

### TC-REG1: split-format entry resolves correctly
- **Given**: registry contains `rnn_v2.1` with `"policy"` and `"value"` sub-dicts
- **When**: `get_ai_info("rnn_v2.1")` called
- **Then**: returns dict with `"policy"` and `"value"` keys; each with `"weight_path"`, `"module"`, `"class"`

### TC-REG2: unknown name raises ValueError
- **Given**: registry has no entry named `"nonexistent"`
- **When**: `get_ai_info("nonexistent")` called
- **Then**: ValueError raised

### TC-REG3: list_ais returns all entries
- **Given**: registry has `rnn_v2.1`, `rnn_v2.2`
- **When**: `list_ais()` called
- **Then**: returns 2 entries, each with `"name"` and `"description"` only

---

## 3. TrainingConfig

### TC-CFG1: valid config passes validation
- **Given**: all fields valid, model exists in registry
- **When**: `TrainingConfig(...)` constructed
- **Then**: no exception raised

### TC-CFG2: `policy_lr <= 0` raises ValueError
- **Given**: `policy_lr = 0.0`
- **When**: `TrainingConfig(...)` constructed
- **Then**: ValueError — "policy_lr must be > 0"

### TC-CFG3: `value_lr <= 0` raises ValueError
- **Given**: `value_lr = -0.1`
- **When**: `TrainingConfig(...)` constructed
- **Then**: ValueError — "value_lr must be > 0"

### TC-CFG4: gamma <= 0 raises ValueError
- **Given**: `gamma = 0.0`
- **When**: `TrainingConfig(...)` constructed
- **Then**: ValueError — "gamma must be in (0, 1]"

### TC-CFG5: gamma > 1 raises ValueError
- **Given**: `gamma = 1.1`
- **When**: `TrainingConfig(...)` constructed
- **Then**: ValueError — "gamma must be in (0, 1]"

### TC-CFG6: gae_lambda outside [0,1] raises ValueError
- **Given**: `gae_lambda = -0.1` or `gae_lambda = 1.1`
- **When**: `TrainingConfig(...)` constructed
- **Then**: ValueError raised

### TC-CFG7: clip_eps <= 0 raises ValueError
- **Given**: `clip_eps = 0.0`
- **When**: `TrainingConfig(...)` constructed
- **Then**: ValueError — "clip_eps must be > 0"

### TC-CFG8: value_coef / entropy_coef < 0 raises ValueError
- **Given**: `value_coef = -0.1`
- **When**: `TrainingConfig(...)` constructed
- **Then**: ValueError — "value_coef must be >= 0"

### TC-CFG9: max_grad_norm <= 0 raises ValueError
- **Given**: `max_grad_norm = 0.0`
- **When**: `TrainingConfig(...)` constructed
- **Then**: ValueError — "max_grad_norm must be > 0"

### TC-CFG10: sample_path nonexistent raises ValueError
- **Given**: `sample_path` does not exist
- **When**: `TrainingConfig.__post_init__()` validates
- **Then**: ValueError — "sample_path does not exist"

### TC-CFG11: value_epochs < 1 raises ValueError
- **Given**: `value_epochs = 0`
- **When**: `TrainingConfig(...)` constructed
- **Then**: ValueError — "value_epochs must be >= 1"

### TC-CFG12: weight_decay < 0 raises ValueError
- **Given**: `weight_decay = -0.01`
- **When**: `TrainingConfig(...)` constructed
- **Then**: ValueError — "weight_decay must be >= 0"

---

## 4. Training — Three-Phase Execution

### TC-PHS1: Phase 1 runs before Phase 3
- **Given**: value_epochs=3, epoch=5
- **When**: `TrainingRunner.run(config)` called
- **Then**: value training (Phase 1) completes all 3 epochs; then advantage re-computation (Phase 2); then policy training (Phase 3) starts

### TC-PHS2: Phase 2 uses trained value network (no_grad)
- **Given**: Phase 1 completed, value network weights updated
- **When**: Phase 2 re-computes GAE
- **Then**: `torch.no_grad()` context used; value network forward produces different V_new than old_values from sample

### TC-PHS3: Phase 3 advantages differ from Phase 1 advantages
- **Given**: value network trained (Phase 1 + Phase 2 completed)
- **When**: comparing advantages from Phase 1 (sample old_values GAE) vs Phase 2 (trained V_new GAE)
- **Then**: advantages are numerically different (trained value produces different estimates)

### TC-PHS4: value network frozen during Phase 3
- **Given**: Phase 3 policy training in progress
- **When**: checking value model parameters after a policy optimizer step
- **Then**: value network parameters unchanged (only policy_optimizer steps)

### TC-PHS5: policy network frozen during Phase 1
- **Given**: Phase 1 value training in progress
- **When**: checking policy model parameters after a value optimizer step
- **Then**: policy network parameters unchanged (only value_optimizer steps)

---

## 5. Training — Gradient Isolation

### TC-GRAD1: policy backward does not affect value model
- **Given**: one training step, policy_loss.backward() called
- **When**: checking value model parameter gradients
- **Then**: all value model parameter gradients are None or zero

### TC-GRAD2: value backward does not affect policy model
- **Given**: one training step, value_loss.backward() called
- **When**: checking policy model parameter gradients
- **Then**: all policy model parameter gradients are None or zero

### TC-GRAD3: both models update independently
- **Given**: one full training group processed
- **When**: comparing model parameters before and after optimizer steps
- **Then**: policy params and value params both changed; no shared gradient path

---

## 5. Training — Loss Computation

### TC-LOSS1: policy_loss has no value MSE component
- **Given**: a perspective with known advantages
- **When**: policy loss computed
- **Then**: loss is PPO clipped objective only

### TC-LOSS2: value_loss is pure MSE
- **Given**: a perspective with known V_targets
- **When**: value loss computed
- **Then**: loss = MSE(V_target, V_new); no policy term

### TC-LOSS3: entropy computed from policy probs only
- **Given**: policy forward pass produces probs tensor
- **When**: entropy bonus computed
- **Then**: entropy = -sum(p * log(p)) / T; subtracted from policy loss

---

## 6. GAE Advantage Computation

### TC-GAE1: winning perspective — advantages positive
- **Given**: a perspective where reward=1.0 (win), old_values near 0
- **When**: GAE advantages computed
- **Then**: advantages are positive for all steps

### TC-GAE2: losing perspective — advantages negative
- **Given**: a perspective where reward=-1.0 (loss), old_values near 0
- **When**: GAE advantages computed
- **Then**: advantages are negative for all steps

### TC-GAE3: advantage at last step = TD error
- **Given**: a perspective with 3 steps
- **When**: GAE computed
- **Then**: advantage at last step = r_t - old_V_t

### TC-GAE4: advantage length matches step count
- **Given**: a perspective with N steps
- **When**: GAE advantages computed
- **Then**: advantage list has exactly N elements

---

## 7. Weight Saving

### TC-SAVE1: two weight files produced
- **Given**: training completes
- **When**: run() finishes
- **Then**: `policy_output_path` and `value_output_path` both exist

### TC-SAVE2: saved policy weights load into GRUPolicyModel
- **Given**: policy weights saved
- **When**: `GRUPolicyModel().load_state_dict(torch.load(path))` called
- **Then**: loads without error; keys match

### TC-SAVE3: saved value weights load into GRUValueModel
- **Given**: value weights saved
- **When**: `GRUValueModel().load_state_dict(torch.load(path))` called
- **Then**: loads without error; keys match

### TC-SAVE4: trained weights produce valid inference
- **Given**: trained weights loaded into AppManager via registry
- **When**: `ai_select_action()` called
- **Then**: returns valid card string and hidden state

---

## 8. Inference — AppManager

### TC-APP1: ai_select_action uses only policy network
- **Given**: AI name = `"rnn_v2.1"`, policy weights loaded
- **When**: `ai_select_action(state, h, "rnn_v2.1")` called
- **Then**: loads GRUPolicyModel from cache; no value model accessed; returns (card_str, new_hidden)

### TC-APP2: ai_sample_action uses both networks
- **Given**: AI name = `"rnn_v2.1"`, both weights loaded
- **When**: `ai_sample_action(state, h_pol, h_val, "rnn_v2.1", t=1.0)` called
- **Then**: loads both models; returns (card_str, new_h_pol, log_prob, value, new_h_val); h_pol != h_val after training

### TC-APP3: model caching works for both caches
- **Given**: first call to `_get_policy_model("rnn_v2.1")` loads from disk
- **When**: second call
- **Then**: returns cached instance; no disk I/O

---

## 9. Self-Play

### TC-SP1: dual hidden states maintained independently
- **Given**: self-play with `rnn_v2.1` as both players
- **When**: playing one round
- **Then**: h_pol and h_val diverge after training (different GRU weights)

### TC-SP2: sample value comes from value network
- **Given**: self-play with sample output enabled
- **When**: sample JSON written
- **Then**: `value` field in each step = GRUValueModel output at that step

### TC-SP3: sample format unchanged
- **Given**: self-play with sample output
- **When**: sample JSON written
- **Then**: structure: game → perspective → steps; each step has `state`, `action`, `reward`, `value`, `log_prob`

---

## 10. Training Artifacts

### TC-ART1: loss.csv has phase column
- **Given**: training completes
- **When**: reading `loss/loss.csv`
- **Then**: header = `phase,epoch,policy_loss,value_loss,entropy`; Phase 1 rows have `phase="value"`, `policy_loss`/`entropy` as NaN; Phase 3 rows have `phase="policy"`, `value_loss` as NaN

### TC-ART2: grad.csv has phase column
- **Given**: training completes
- **When**: reading `loss/grad.csv`
- **Then**: header = `phase,epoch,policy_grad_norm,value_grad_norm`; Phase 1 rows have `policy_grad_norm` as NaN; Phase 3 rows have `value_grad_norm` as NaN

### TC-ART3: value_loss does not monotonically increase
- **Given**: split models, epoch >= 5
- **When**: reading `loss/loss.csv`
- **Then**: value_loss at epoch 5 <= value_loss at epoch 1 (gradient interference eliminated)

## 12. Optimizer

### TC-OPT1: AdamW optimizer type
- **Given**: both models loaded, training loop initialized
- **When**: checking optimizer instances
- **Then**: `policy_optimizer` is `torch.optim.AdamW`; `value_optimizer` is `torch.optim.AdamW`

### TC-OPT2: weight_decay forwarded to AdamW
- **Given**: `config.weight_decay = 0.01`
- **When**: optimizers created
- **Then**: `policy_optimizer.param_groups[0]["weight_decay"] == 0.01`; same for value_optimizer

---

## 11. End-to-End

### TC-E2E1: full split training pipeline
- **Given**: sample file of 10 games from `rnn_v2.1` self-play, epoch=5
- **When**: `TrainingRunner.run(config)` called
- **Then**: training completes; policy.pth + value.pth exist; loss/* artifacts exist

### TC-E2E2: trained split model outperforms random
- **Given**: `rnn_v2.2` trained on biased samples
- **When**: `rnn_v2.2` vs random baseline, 100 games
- **Then**: `rnn_v2.2` wins >50%

### TC-E2E3: full cycle — self-play → train → inference
- **Given**: `rnn_v2.1` random weights; 100 self-play → samples; train epoch=5 → `rnn_v2.2`
- **When**: `rnn_v2.2` vs `rnn_v2.1` 100 games
- **Then**: `rnn_v2.2` win rate > `rnn_v2.1`
