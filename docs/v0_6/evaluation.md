# Evaluation and scientific gates

Formal evaluation uses the online encoder, Koopman core and decoder only. It reports
short/medium/long rollout, persistence baseline, mass drift, exact operator penalty,
spectrum, frequency/decay, near-identity, latent statistics, covariance condition,
online-target latent distance and parameter distance.

The required experiment is JEPA versus `lambda_J=0`, with the same V0.5 checkpoint,
seed, split, data, normalizer, trainable capacity, optimizer, epochs, physics and other
losses. Seeds 47, 53 and 59 are required for stabilization claims. The conservative
automated gate requires every seed to satisfy long-rollout non-inferiority (5%),
mass/operator non-inferiority (10%), no-collapse, and target-free rollout. Mean and
sample standard deviation are retained.

Software success does not imply scientific acceptance. Failure must not be repaired by
adding attention, residual state, or capacity.
