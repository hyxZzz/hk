# Multi-Agent DDQN Workflow Overview

This document summarises the multi-agent training and evaluation pipeline that the
project now follows after the recent refactor.  It is meant to help you understand
how checkpoints are created, grouped, and validated so that the reported intercept
success rate truly reflects the behaviour of every cooperating aircraft.

## Training loop (`utils/train.py`)

1. **Environment bootstrap** – the training script initialises a multi-plane
   environment through `Environment.init_env.init_env`, configuring the number of
   cooperative aircraft, incoming missiles, interceptors, and simulation steps.
2. **Per-agent construction** – each plane receives its own `MyDQNAgent` instance
   with an independent `Double_DQN` model.  All agents share the same replay
   buffer (`utils.ERbuffer.MyMemoryBuffer`) so they learn from joint experiences.
3. **Experience collection** – during an episode the script gathers the complete
   state dictionary, the actions chosen by every agent, and the reward mapping.
   These tuples are stored in the replay buffer for subsequent sampling.
4. **Independent updates** – whenever the buffer is warm and the learning
   frequency is met, each agent draws the relevant parts of the sampled batches
   (state, action, reward, next state, and terminal flag) before running its own
   `learn` step.
5. **Periodic evaluation** – every 50 episodes the training loop temporarily
   switches all agents to evaluation mode (`evaluate_agents`) to estimate mean
   rewards per aircraft and the cooperative success rate.  Metrics are logged to
   TensorBoard for monitoring.
6. **Multi-agent checkpoints** – every 100 episodes the script persists the model
   parameters of *each* agent to
   `models/DQNmodels/DDQNmodels3_23/DDQN_agent{ID}_episode{N}.pth`.  The full set
   of paths is passed to `utils.validate.evaluate_checkpoint` so that validation
   always runs with a coherent group of policies.  The resulting intercept success
   rate is written to both TensorBoard and a CSV file inside `runs/val/`.

## Validation utilities (`utils/validate.py`)

1. **Checkpoint discovery** – `collect_checkpoints` walks the model directory,
   matches filenames via the pattern `DDQN_agent(\d+)_episode(\d+)`, and groups the
   results by episode.  This produces ready-to-evaluate bundles where every agent
   checkpoint shares the same episode index.
2. **Configuration capsule** – the `EvaluationConfig` dataclass encapsulates the
   environment and agent hyper-parameters required to faithfully reproduce the
   training setup at validation time.
3. **Robust loading** – `load_checkpoint` now attempts to read checkpoints using
   `torch.load(..., weights_only=True)` when supported, preventing the security
   warning that PyTorch emits for the legacy pickle-based mode.  If the installed
   version does not offer this flag the function automatically falls back to the
   original behaviour.
4. **Complete-agent enforcement** – `_validate_checkpoint_mapping` ensures that a
   checkpoint bundle contains weights for every expected agent.  Incomplete
   bundles are skipped with an explanatory message instead of silently producing
   misleading metrics.
5. **Vectorised evaluation** – `evaluate_checkpoint` instantiates a `MyDQNAgent`
   per aircraft, loads its corresponding checkpoint, and deploys the agents
   together in the simulation.  Actions are selected greedily (without exploration)
   so the evaluation measures the exact policy contained in the saved models.
6. **Results logging** – the script writes success rates to TensorBoard and emits
   a CSV (`runs/val/<timestamp>/intercept_success_rates.csv`) that enumerates the
   exact checkpoints contributing to each measurement.

## Running validation manually

You can trigger an evaluation sweep over all detected checkpoints with:

```bash
python -m utils.validate --model-root models --episodes 200 --num-planes 2
```

Adjust `--episodes`, `--num-planes`, and other CLI flags if your training setup
changes.  The output lists each episode's success rate alongside the checkpoint
paths that were used, making it easy to trace surprising results back to the
underlying models.
