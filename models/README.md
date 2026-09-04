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
