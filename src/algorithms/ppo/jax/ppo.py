"""From-scratch JAX PPO implementation.

Clipped surrogate objective with GAE, value clipping, and entropy bonus.
Uses nested jax.lax.scan for fully JIT-compiled training.
"""
