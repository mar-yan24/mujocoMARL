"""From-scratch JAX TRPO implementation.

Natural gradient with KL constraint via conjugate gradient solver.
Uses jax.jvp/jax.vjp for Fisher-vector products.
"""
