# Technology adoption declaration

| Technique | Decision | Code location | Matched evidence |
|---|---|---|---|
| Low-rank context-conditioned generator | ADOPT | `jka_model/adaptive/models.py` | nominal, static mean update, shuffled context |
| Batched exact matrix exponential | ADOPT | adaptive model and rollout | zero-update exact-equivalence test |
| Controlled smooth/abrupt inlet schedules | ADOPT | `data/cylinder_wake_2d.py` | separate schedule metrics |
| Known-condition oracle | ADOPT | operator condition embedding | compared with latent-inferred |
| Symmetric-abscissa proxy | OPTIONAL diagnostic/regularizer | adaptive model/trainer | real rollout remains acceptance owner |
| Teacher-free rollout curriculum | ADOPT | `adaptive/objectives.py` | one-step-only baseline |
| Bounded eta and learned trust gate | ADOPT | `adaptive/models.py` | gate/burden diagnostics |
| Relative nominal propagator growth | ADOPT | `adaptive/objectives.py` | H32/H80 locked rollout |
| Frozen-decoder raw-unit physics anchor | ADOPT | `adaptive/physics.py` | nominal physical non-inferiority |
| Mixture-of-Koopman | DEFER | not implemented | only if continuous low-rank fails for discrete switching |
| Persistent residual state | DEFER TO V1.1 | not implemented | remaining residual first reassessed |
| Joint fine-tuning | DEFER | not implemented | frozen modules verified |

采用项不会增加新的 governing physics。入口变化是受控边界参数变化；operator regularizers 约束学习模型，不修改 CFD。
