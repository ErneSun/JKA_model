# Latest technology review

Review date: 2026-08-16. Primary papers and official implementations were used.

The mature transferable mechanism is the momentum target: I-JEPA/V-JEPA create a
same-architecture target, stop its gradients, save it in checkpoints, and update it
after optimizer steps. V-JEPA 2 continues this family, but its ViT masking, dense/deep
self-supervision and action components solve different problems and are outside V0.6.

Recent PDE JEPA work introduces action-conditioned control, ViTs or new latent
decompositions; those would change the scientific variable and are deferred.
LeWorldModel's Gaussian regularizer also removes EMA and changes collapse control, so
it is an ablation candidate. PI-JEPA is withdrawn and is not technical evidence.

The direct JEPA–Koopman result motivates invariant and near-identity diagnostics, but
does not provide a drop-in continuous-time, variable-`dt`, physics-decoded method.
V0.6 therefore uses a minimal reviewed construction and preserves V0.5.
