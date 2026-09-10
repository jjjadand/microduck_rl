# Model Artifacts

## Official ONNX policies

The official Pollen Robotics inference policies are stored in:

```text
pretrained/pollen-robotics/
```

These ONNX files are inference artifacts and do not contain the PPO optimizer,
critic, or training state required to resume training.

## Jetson training checkpoints

The `.pt` files under:

```text
models/checkpoints/rsl_rl/velocity/
```

were produced by the Jetson training runs copied from
`logs/rsl_rl/velocity/`. They are local training checkpoints rather than
official Pollen Robotics releases. `model_3000.pt` is the latest checkpoint in
the longest included walking run.

The original TensorBoard event files, videos, generated logs, and the `.venv`
environment are intentionally excluded from this repository.

## Front-back split

The completed Jetson front-back split artifacts are stored in:

```text
models/checkpoints/rsl_rl/front_back_split/2026-09-09_18-04-10_front_back_split_left_forward/model_999.pt
models/exports/front_back_split/front_back_split_model_999.onnx
```

The checkpoint contains the PPO training state. The ONNX export contains the
actor policy with its observation normalizer embedded and is loaded by
`scripts/infer_policy.py --front-back-split`. Press `O` to run one six-second
phase cycle and return to the walking or standing policy.
