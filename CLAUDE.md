# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 1. Think Before Coding
**Don't assume. Don't hide confusion. Surface tradeoffs.**
Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.
## 2. Simplicity First
**Minimum code that solves the problem. Nothing speculative.**
- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.
## 3. Surgical Changes
**Touch only what you must. Clean up only your own mess.**
When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.
When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.
The test: Every changed line should trace directly to the user's request.
## 4. Goal-Driven Execution
**Define success criteria. Loop until verified.**
Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"
For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

## Project Overview

**RouteFinder** (v0.4.0, AI4CO, TMLR 2025) trains neural foundation models that solve ~48 Vehicle Routing Problem (VRP) variants with a single model. Variants are defined by combining 5 constraint features: Capacity (C), Open routes (O), Backhauls (B/M), Duration Limit (L), Time Windows (TW). E.g. CVRP, VRPTW, OVRPBLTW, and all combinations.

Built on:
- **`rl4co`** (external lib, pinned to git `main`) — environment base classes, attention model policy, POMO RL algorithm, `RL4COTrainer`. Understanding behavior often requires reading rl4co source.
- **PyTorch Lightning** — training loop via `RL4COTrainer`.
- **TorchRL / TensorDict** — `tensordict.TensorDict` is the universal data container for state, instances, actions, and batches.
- **Hydra** — all training/eval is launched through `run.py` with YAML configs under `configs/`.

## Commands
**Install (recommended):**
```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync --all-extras
```

**Install (pip):**
```bash
pip install -e ".[dev,solver]"
```
**Train:**
```bash
python run.py experiment=main/rf/rf-transformer-100
# Override any Hydra config key inline, e.g.:
python run.py experiment=main/rf/rf-transformer-100 +trainer.devices="[0]"
```
Experiment configs live under `configs/experiment/` (e.g. `main/rf/`, `main/mtpomo/`, `main/mvmoe/`, `ablations/`).

## Architecture

### TensorDict as universal currency
Every env method, embedding, and model step takes/returns a `tensordict.TensorDict` keyed by named tensors (`locs`, `demand_linehaul`, `time_windows`, `action_mask`, `current_node`, …). A "VRP variant" is just which keys hold non-default values — absent constraints are stored as `inf`/`0` defaults and never affect behavior.

### Environment: `routefinder/envs/mtvrp/`
`MTVRPEnv(RL4COEnvBase)` is a single environment that represents all variants:
- **Constraints are enforced by action masking**, not reward penalties — all VRP rules live in `get_action_mask`.
- Absent constraints are filled to defaults in `_reset` (e.g. `time_windows → [0, ∞]`, `distance_limit → ∞`), then zeroed via `nan_to_num` in the embeddings. This lets one network handle all variants.
- `MTVRPGenerator.subsample_problems` randomly drops constraint features per instance to produce mixed-variant batches.

### Model: `routefinder/models/`
| File | Class | Role |
|------|-------|------|
| `model.py` | `RouteFinderBase(POMO)` | Lightning module; REINFORCE + POMO multistart baseline; per-variant reward normalization |
| `model.py` | `RouteFinderSingleVariantSampling` | Baseline training strategy: one variant per batch |
| `policy.py` | `RouteFinderPolicy(AttentionModelPolicy)` | Wires encoder + decoder + embeddings |
| `encoder.py` | `RouteFinderEncoder` | Init embedding → stack of `TransformerBlock`s |
| `env_embeddings/mtvrp/init.py` | `MTVRPInitEmbeddingRouteFinder` | Raw TensorDict features → node vectors; global features broadcast through depot |
| `env_embeddings/mtvrp/context.py` | `MTVRPContextEmbeddingRouteFinder` | Decoder query context (available load, current time, remaining distance) |
| `reward_normalization.py` | `BaseValues` + normalizers | Per-variant reward normalization for MBT |

**Two training strategies:**
1. **Mixed-Batch Training (MBT, default):** `RouteFinderBase` + `subsample_problems` — one batch contains many variants simultaneously; per-variant reward normalization balances gradients.
2. **Single-variant sampling (baselines):** `RouteFinderSingleVariantSampling` removes one chosen feature set for the whole batch per step.

**RL algorithm:** REINFORCE with POMO shared baseline — N start nodes per instance, best tour is the solution, per-instance mean reward is the baseline.

### Training orchestration
`run.py` is a Hydra `@hydra.main` app over `configs/main.yaml`. It instantiates (via `_target_`): env → model → callbacks → loggers → `RL4COTrainer`, then calls `trainer.fit` / `trainer.test`. Config subdirs: `configs/{env,model,trainer,callbacks,logger,experiment,paths}/`.

### Data flow
```
MTVRPGenerator._generate       → TensorDict (locs, demands, time_windows, …)
  ↓ subsample_problems         → randomly drop constraints (MBT)
MTVRPEnv._reset                → fill defaults, init state, compute first action_mask
  ↓
RouteFinderEncoder             → init_embedding → TransformerBlocks → node embeddings
  ↓
AttentionModel decoder loop    → context embedding + masked pointer attention
  ↓ each step calls MTVRPEnv._step → update state, recompute action_mask
MTVRPEnv._get_reward           → negative tour length (open routes skip depot return)
  ↓
RouteFinderBase.calculate_loss → REINFORCE with POMO baseline + reward normalization
```