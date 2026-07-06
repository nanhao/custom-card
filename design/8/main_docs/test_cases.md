# Test Cases — PPO Training Script

Test cases use text and symbols only. No code. Tests are organized by target behavior.

---

## 1. Sample Loading

### TC-L1: load a single sample file
- **Given**: a valid sample file at `sample-path` with 2 games, each with 2 perspectives
- **When**: `TrainingRunner.run(config)` starts
- **Then**: 4 perspectives are loaded (2 games × 2 perspectives); no error raised

### TC-L2: load all JSON files from a directory
- **Given**: `sample-path` is a directory containing 3 `.json` files: `a.json` (1 game), `b.json` (2 games), `c.json` (1 game); no other file types
- **When**: `TrainingRunner.run(config)` starts
- **Then**: 8 perspectives are loaded (4 games × 2); only `.json` files are read

### TC-L3: directory with no JSON files raises RuntimeError
- **Given**: `sample-path` is a directory containing only `readme.txt`
- **When**: `TrainingRunner.run(config)` starts
- **Then**: RuntimeError raised — "no valid training samples found"

### TC-L4: nonexistent path raises FileNotFoundError
- **Given**: `sample-path` is a path that does not exist
- **When**: `TrainingConfig.__post_init__()` validates
- **Then**: ValueError raised — "sample_path does not exist"

### TC-L5: file with empty array `[]` raises RuntimeError
- **Given**: sample file containing `[]`
- **When**: `TrainingRunner.run(config)` starts
- **Then**: RuntimeError raised — "no valid training samples found"

### TC-L6: duplicate game_ids across files are accepted
- **Given**: two sample files, each containing game_id=1
- **When**: `TrainingRunner.run(config)` starts
- **Then**: all perspectives from both files are loaded (perspectives are the training unit, not games)

### TC-L7: corrupted JSON file raises RuntimeError
- **Given**: sample file with invalid JSON syntax
- **When**: `TrainingRunner.run(config)` starts
- **Then**: RuntimeError raised — "Failed to parse sample file 'x.json'" with descriptive message

### TC-L8: non-JSON files in directory silently skipped
- **Given**: directory with `a.json` (valid), `b.txt`, `c.json.bak`, `d.json` (valid)
- **When**: loading samples
- **Then**: only `a.json` and `d.json` loaded; `b.txt` and `c.json.bak` skipped

---

## 2. TrainingConfig Validation

### TC-C1: valid config passes validation
- **Given**: all fields set to valid values (epoch=10, batch_size=64, gamma=0.99, gae_lambda=0.95, clip_eps=0.2, lr=0.001, value_coef=0.5, entropy_coef=0.01)
- **When**: `TrainingConfig(...)` constructed
- **Then**: no exception raised

### TC-C2: epoch < 1 raises ValueError
- **Given**: epoch=0
- **When**: `TrainingConfig(...)` constructed
- **Then**: ValueError raised — "epoch must be >= 1"

### TC-C3: batch_size < 1 raises ValueError
- **Given**: batch_size=0
- **When**: `TrainingConfig(...)` constructed
- **Then**: ValueError raised — "batch_size must be >= 1"

### TC-C4: gamma <= 0 raises ValueError
- **Given**: gamma=0.0
- **When**: `TrainingConfig(...)` constructed
- **Then**: ValueError raised — "gamma must be in (0, 1]"

### TC-C5: gamma > 1 raises ValueError
- **Given**: gamma=1.1
- **When**: `TrainingConfig(...)` constructed
- **Then**: ValueError raised — "gamma must be in (0, 1]"

### TC-C6: gae_lambda < 0 raises ValueError
- **Given**: gae_lambda=-0.1
- **When**: `TrainingConfig(...)` constructed
- **Then**: ValueError raised — "gae_lambda must be in [0, 1]"

### TC-C7: gae_lambda > 1 raises ValueError
- **Given**: gae_lambda=1.1
- **When**: `TrainingConfig(...)` constructed
- **Then**: ValueError raised — "gae_lambda must be in [0, 1]"

### TC-C8: clip_eps <= 0 raises ValueError
- **Given**: clip_eps=0.0
- **When**: `TrainingConfig(...)` constructed
- **Then**: ValueError raised — "clip_eps must be > 0"

### TC-C9: lr <= 0 raises ValueError
- **Given**: lr=0.0
- **When**: `TrainingConfig(...)` constructed
- **Then**: ValueError raised — "lr must be > 0"

### TC-C10: value_coef < 0 raises ValueError
- **Given**: value_coef=-0.1
- **When**: `TrainingConfig(...)` constructed
- **Then**: ValueError raised — "value_coef must be >= 0"

### TC-C11: entropy_coef < 0 raises ValueError
- **Given**: entropy_coef=-0.01
- **When**: `TrainingConfig(...)` constructed
- **Then**: ValueError raised — "entropy_coef must be >= 0"

### TC-C12: max_grad_norm <= 0 raises ValueError
- **Given**: max_grad_norm=0.0
- **When**: `TrainingConfig(...)` constructed
- **Then**: ValueError raised — "max_grad_norm must be > 0"

---

## 3. Model Loading

### TC-M1: valid registry name loads model successfully
- **Given**: `config.model = "rnn_v1.1"` (exists in registry)
- **When**: `TrainingRunner.run(config)` starts
- **Then**: GRUCardModel instantiated, weights loaded from registry path, model moved to device

### TC-M2: unknown registry name raises ValueError
- **Given**: `config.model = "nonexistent_ai"`
- **When**: `TrainingRunner.run(config)` starts
- **Then**: ValueError raised — "Unknown AI: 'nonexistent_ai'"

### TC-M3: model loaded in training mode
- **Given**: model loaded successfully
- **When**: training loop starts
- **Then**: `model.training == True`

---

## 4. GRU Hidden State Management

### TC-H1: hidden state resets at perspective boundary
- **Given**: a group of 2 perspectives, each with 3 steps
- **When**: processing perspectives sequentially with gradient accumulation
- **Then**: `h` starts at `INITIAL_HIDDEN_STATE` (all zeros) for each perspective; `h` is carried forward within a perspective; gradients are accumulated across perspectives

### TC-H2: initial hidden state shape matches model expectation
- **Given**: model loaded, INITIAL_HIDDEN_STATE = [0.0] * 8
- **When**: first forward pass of a perspective
- **Then**: input hidden tensor has shape (1, 1, 8); all values are 0.0

### TC-H3: hidden state changes after forward pass
- **Given**: any step with non-zero input
- **When**: `model.forward(x, h)` called
- **Then**: returned `new_hidden` is not equal to input `h` (tensor not identical)

---

## 4b. Gradient Accumulation

### TC-GA1: optimizer step only after accumulating batch_size perspectives
- **Given**: batch_size=3, 3 perspectives
- **When**: processing a group
- **Then**: `optimizer.zero_grad()` called once; 3 calls to `loss.backward()`; `optimizer.step()` called once

### TC-GA2: loss scaled by batch_size during accumulation
- **Given**: batch_size=3, 3 perspectives processed
- **When**: computing loss for each perspective
- **Then**: each perspective's loss is divided by 3 before backward(); accumulated gradients match full-batch equivalent

### TC-GA3: variable-length perspectives work without padding
- **Given**: a group with perspectives of lengths 3, 5, and 7 rounds
- **When**: processing the group with gradient accumulation
- **Then**: each perspective is processed independently with its own h trajectory; no tensors padded; loss computed correctly for each

### TC-GA5: batch_size=1 works correctly
- **Given**: batch_size=1, 5 perspectives
- **When**: processing each group (one perspective per group)
- **Then**: each perspective triggers independent optimizer step; training completes successfully

---

## 4d. NaN Handling

### TC-NAN1: NaN loss triggers skip and warning
- **Given**: a perspective causing NaN loss (e.g., all-zero input, extreme weights)
- **When**: loss computed and NaN detected
- **Then**: warning message printed; optimizer step skipped for this group; training continues with next group

### TC-NAN2: NaN gradient triggers skip and warning
- **Given**: gradient norm is NaN after backward()
- **When**: gradient norm computed
- **Then**: warning message printed; optimizer step skipped; training continues

---

## 4c. Temperature Behavior

### TC-TEMP1: forward called with temperature=1.0 during training
- **Given**: model in training mode, any sample
- **When**: computing new policy logits during training
- **Then**: `model.forward()` called with `temperature=1.0` (standard softmax, no sampling)

### TC-TEMP2: log_probs represent full distribution (not sampled)
- **Given**: temperature=1.0 in forward()
- **When**: examining returned log_probs tensor
- **Then**: log_probs has shape (1, 4); all 4 values are finite (not just one 0 and rest -inf)

---

## 5. GAE Advantage Computation

### TC-A1: winning perspective — terminal advantage is positive
- **Given**: a perspective where all steps have reward=1.0 (win), gamma=0.99, gae_lambda=0.95, old_values are near 0
- **When**: GAE advantages computed
- **Then**: advantages are positive for all steps

### TC-A2: losing perspective — terminal advantage is negative
- **Given**: a perspective where all steps have reward=-1.0 (loss), gamma=0.99, gae_lambda=0.95, old_values are near 0
- **When**: GAE advantages computed
- **Then**: advantages are negative for all steps

### TC-A3: draw perspective — terminal advantage near zero
- **Given**: a perspective where all steps have reward=0.0 (draw), gamma=0.99, gae_lambda=0.95, old_values are near 0
- **When**: GAE advantages computed
- **Then**: advantages are close to 0.0 (within ±0.5)

### TC-A4: advantage at last step equals TD error
- **Given**: a perspective with 3 steps, gamma=0.99
- **When**: GAE computed
- **Then**: advantage at last step t = r_t - old_V_t (no bootstrap from future)

### TC-A5: advantage length matches step count
- **Given**: a perspective with N steps
- **When**: GAE advantages computed
- **Then**: advantage list has exactly N elements

---

## 6. PPO Loss Computation

### TC-PPO1: loss is a scalar tensor with grad
- **Given**: a single perspective after forward pass
- **When**: PPO loss computed via TrainingRunner
- **Then**: loss is a scalar torch.Tensor; `loss.requires_grad == True`

### TC-PPO2: policy loss is non-negative
- **Given**: any valid perspective
- **When**: policy loss computed (= `L_clip` component)
- **Then**: policy loss >= 0

### TC-PPO3: value loss is non-negative
- **Given**: any valid perspective
- **When**: value loss computed (= MSE between V_target and V_new)
- **Then**: value loss >= 0

### TC-PPO4: log_prob ratio within clip range after update
- **Given**: clip_eps=0.2, one optimizer step on an accumulation group
- **When**: computing ratio = exp(new_log_prob - old_log_prob) on the same group
- **Then**: loss is bounded (not NaN or inf)

---

## 7. Weight Update

### TC-W1: weights change after one epoch
- **Given**: model with loaded weights, 1 epoch of training on valid samples
- **When**: comparing state_dict before and after training
- **Then**: at least one parameter's values changed (not bitwise identical)

### TC-W2: gradient norm is finite
- **Given**: any training step after backward()
- **When**: gradient norm computed via `torch.nn.utils.clip_grad_norm_` or manual computation
- **Then**: grad_norm is finite (not NaN or inf)

### TC-W3: optimizer step reduces loss
- **Given**: an accumulation group with consistent advantage signals (all positive or all negative)
- **When**: one optimizer step executed after gradient accumulation
- **Then**: loss after step < loss before step (for that same group)

---

## 8. Weight Saving

### TC-S1: output file created at output_path
- **Given**: training completes successfully
- **When**: run() finishes
- **Then**: file exists at `config.output_path`; can be loaded with `torch.load(path, weights_only=True)`

### TC-S2: saved weights match model state
- **Given**: training completes, weights saved
- **When**: a new GRUCardModel loads the saved weights
- **Then**: all parameter values match the trained model's state_dict (within float tolerance)

### TC-S3: parent directories created if needed
- **Given**: `output_path = "nonexistent/dir/model.pth"`, parent dirs do not exist
- **When**: training completes
- **Then**: parent directories are created; weights file exists at output_path

### TC-S4: saved weights usable for inference
- **Given**: weights saved to output_path
- **When**: `AppManager._get_model()` loads a registry entry pointing to output_path
- **Then**: model produces valid probs/log_probs/value via forward()

---

## 9. Loss/Gradient Artifacts

### TC-G1: loss/loss.png created
- **Given**: training completes with epoch >= 2
- **When**: run() finishes
- **Then**: `loss/loss.png` exists; is a valid PNG image

### TC-G2: loss/loss.csv created
- **Given**: training completes with epoch >= 2
- **When**: run() finishes
- **Then**: `loss/loss.csv` exists; has header row with column names; has epoch rows of valid float values

### TC-G3: loss/grad.png created
- **Given**: training completes with epoch >= 2
- **When**: run() finishes
- **Then**: `loss/grad.png` exists; is a valid PNG image

### TC-G4: loss/grad.csv created
- **Given**: training completes with epoch >= 2
- **When**: run() finishes
- **Then**: `loss/grad.csv` exists; has header row with column names; has epoch rows of valid float values

### TC-G5: CSV epoch count matches config.epoch
- **Given**: config.epoch=5, training completes
- **When**: reading `loss/loss.csv`
- **Then**: data rows count = 5 (one per epoch)

### TC-G6: loss values are decreasing trend (not strictly monotonic)
- **Given**: training on consistent samples, epoch >= 5
- **When**: reading `loss/loss.csv`
- **Then**: loss at epoch 5 is lower than loss at epoch 1 (general downward trend)

### TC-G7: loss/ directory auto-created if missing
- **Given**: `loss/` directory does not exist before training
- **When**: run() finishes
- **Then**: `loss/` directory is created; artifact files exist inside

---

## 10. Device Management

### TC-D1: CUDA used when available
- **Given**: `torch.cuda.is_available()` returns True
- **When**: `TrainingRunner.run()` starts
- **Then**: model parameters are on `cuda` device; input tensors are on `cuda` device

### TC-D2: CPU fallback when CUDA unavailable
- **Given**: `torch.cuda.is_available()` returns False
- **When**: `TrainingRunner.run()` starts
- **Then**: warning printed to stdout; model parameters are on `cpu` device; training proceeds without error

---

## 11. CLI Argument Parsing

### TC-CLI1: all required arguments provided
- **Given**: `--sample-path`, `--output-path`, `--model`, `--epoch` all specified
- **When**: script parses arguments
- **Then**: parsed values match input; no error

### TC-CLI2: missing required argument raises SystemExit
- **Given**: `--epoch` omitted
- **When**: script parses arguments
- **Then**: argparse raises SystemExit (or error message printed)

### TC-CLI3: epoch is int type
- **Given**: `--epoch 10`
- **When**: script parses arguments
- **Then**: `args.epoch == 10` (type int, not str)

---

## 12. End-to-End

### TC-E2E1: full training run completes without error
- **Given**: a sample file with 10 games, model="rnn_v1.1", epoch=2, valid backend.json
- **When**: `TrainingRunner.run(config)` called
- **Then**: training completes without exception; output.pth exists; loss/*.png and loss/*.csv exist

### TC-E2E2: trained model produces different actions than untrained
- **Given**: model trained for 10 epochs on biased samples (all wins by player 1 using "attack")
- **When**: comparing action distribution before vs after training on identical input states
- **Then**: action probabilities differ measurably (the policy has learned)

### TC-E2E3: training is deterministic with fixed seed
- **Given**: same samples, same config, `torch.manual_seed(42)` set before run()
- **When**: two separate training runs executed
- **Then**: final loss values match within float tolerance (training is reproducible)
