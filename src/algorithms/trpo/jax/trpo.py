"""From-scratch JAX TRPO implementation.

Natural gradient with KL constraint via conjugate gradient solver.
Uses jax.jvp/jax.vjp for Fisher-vector products.
"""

from __future__ import annotations
from typing import Any

import jax
import jax.numpy as jnp
import optax
import flax.struct

from src.algorithms.on_policy_base import OnPolicyBase, OnPolicyState
from src.buffers.rollout_buffer import RolloutBatch, flatten_batch


# ---- pytree math utils ----

def _tree_dot(a, b):
    leaves = jax.tree.leaves(jax.tree.map(lambda x, y: jnp.sum(x * y), a, b))
    return sum(leaves)


def _tree_add(a, b):
    return jax.tree.map(lambda x, y: x + y, a, b)


def _tree_scale(a, s):
    return jax.tree.map(lambda x: x * s, a)


class TRPO(OnPolicyBase):
    '''
    trust region policy optimization.
    overrides the full update() because TRPO uses conjugate gradient
    instead of minibatch SGD for the actor.
    critic is still updated with SGD.
    '''

    def update(self, state: OnPolicyState, batch: RolloutBatch):
        '''
        TRPO update:
          1. flatten batch (no minibatching for TRPO policy update)
          2. compute policy gradient g
          3. solve H @ x = g via conjugate gradient (H = Fisher info matrix)
          4. compute step size from max KL constraint
          5. line search with KL backtracking
          6. update critic with normal SGD
        '''
        trpo_cfg = self.config.trpo
        rng = state.rng

        # normalize advantages
        adv = batch.advantage
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        batch = batch.replace(advantage=adv)
        flat = flatten_batch(batch)

        rng, pg_rng, cg_rng, critic_rng = jax.random.split(rng, 4)

        # -- policy update via natural gradient --

        def surrogate_loss(params):
            '''policy gradient surrogate (vanilla, no clipping)'''
            _, new_log_prob, _, _ = self.actor_model.apply(
                params, flat.obs, rngs={"sample": pg_rng},
            )
            ratio = jnp.exp(new_log_prob - flat.log_prob)
            return -(ratio * flat.advantage).mean()

        def kl_divergence(params):
            '''mean KL(pi_old || pi_new) using gaussian KL'''
            _, _, new_mean, new_log_std = self.actor_model.apply(
                params, flat.obs, rngs={"sample": pg_rng},
            )
            _, _, old_mean, old_log_std = self.actor_model.apply(
                state.actor_params, flat.obs, rngs={"sample": pg_rng},
            )
            # analytic KL for diagonal gaussians
            old_std = jnp.exp(old_log_std)
            new_std = jnp.exp(new_log_std)
            kl = (
                new_log_std - old_log_std
                + (old_std ** 2 + (old_mean - new_mean) ** 2) / (2 * new_std ** 2 + 1e-8)
                - 0.5
            )
            return kl.sum(-1).mean()

        # policy gradient
        g = jax.grad(surrogate_loss)(state.actor_params)

        # fisher-vector product: Hv = d/dθ (dKL/dθ · v)
        def fvp(v):
            def kl_grad_dot_v(params):
                kl_grad = jax.grad(kl_divergence)(params)
                return _tree_dot(kl_grad, v)
            hvp = jax.grad(kl_grad_dot_v)(state.actor_params)
            return _tree_add(hvp, _tree_scale(v, trpo_cfg.cg_damping))

        # conjugate gradient: solve Hx = g
        def _cg_step(carry, _):
            x, r, p, rdotr = carry
            Ap = fvp(p)
            alpha = rdotr / (_tree_dot(p, Ap) + 1e-8)
            x = _tree_add(x, _tree_scale(p, alpha))
            r = jax.tree.map(lambda ri, api: ri - alpha * api, r, Ap)
            new_rdotr = _tree_dot(r, r)
            beta = new_rdotr / (rdotr + 1e-8)
            p = _tree_add(r, _tree_scale(p, beta))
            return (x, r, p, new_rdotr), None

        x0 = jax.tree.map(jnp.zeros_like, g)
        init_carry = (x0, g, g, _tree_dot(g, g))
        (step_dir, _, _, _), _ = jax.lax.scan(
            _cg_step, init_carry, None, length=trpo_cfg.cg_iters,
        )

        # step size: sqrt(2 * max_kl / (x^T H x))
        shs = _tree_dot(step_dir, fvp(step_dir))
        step_size = jnp.sqrt(2.0 * trpo_cfg.max_kl / (shs + 1e-8))
        full_step = _tree_scale(step_dir, -step_size)  # negative because we minimize

        # line search with backtracking
        def _line_search_step(carry, _):
            params, coeff, found = carry
            new_params = _tree_add(state.actor_params, _tree_scale(full_step, coeff))
            new_loss = surrogate_loss(new_params)
            new_kl = kl_divergence(new_params)
            old_loss = surrogate_loss(state.actor_params)
            improved = (new_loss < old_loss) & (new_kl < trpo_cfg.max_kl)
            # keep the best found so far
            params = jax.lax.cond(
                improved & ~found,
                lambda: new_params,
                lambda: params,
            )
            found = found | improved
            coeff = coeff * 0.5
            return (params, coeff, found), None

        init_ls = (state.actor_params, jnp.float32(1.0), jnp.bool_(False))
        (new_actor_params, _, _), _ = jax.lax.scan(
            _line_search_step, init_ls, None, length=trpo_cfg.line_search_steps,
        )

        # -- critic update via SGD (a few epochs) --
        def _critic_step(carry, _):
            critic_params, critic_opt = carry
            (loss, metrics), grads = jax.value_and_grad(
                self._critic_loss, has_aux=True
            )(critic_params, batch)
            updates, critic_opt = self.critic_optimizer.update(grads, critic_opt, critic_params)
            critic_params = optax.apply_updates(critic_params, updates)
            return (critic_params, critic_opt), metrics

        (new_critic_params, new_critic_opt), critic_metrics = jax.lax.scan(
            _critic_step,
            (state.critic_params, state.critic_opt_state),
            None,
            length=5,  # a few critic SGD steps
        )
        critic_metrics = jax.tree.map(lambda x: x.mean(), critic_metrics)

        # compute final metrics
        final_kl = kl_divergence(new_actor_params)
        metrics = {
            "kl": final_kl,
            **critic_metrics,
        }

        new_state = OnPolicyState(
            actor_params=new_actor_params,
            critic_params=new_critic_params,
            actor_opt_state=state.actor_opt_state,  # not used for actor (natural gradient)
            critic_opt_state=new_critic_opt,
            rng=rng,
            step=state.step + 1,
        )
        return new_state, metrics

    # these are only used for the critic in TRPO
    def _actor_loss(self, actor_params, batch, rng):
        raise NotImplementedError("TRPO uses natural gradient, not SGD for actor")

    def _critic_loss(self, critic_params, batch: RolloutBatch):
        '''simple MSE value loss'''
        trpo_cfg = self.config.trpo
        value = self.critic_model.apply(critic_params, batch.obs)
        value_loss = 0.5 * ((value - batch.returns) ** 2).mean()
        total_loss = trpo_cfg.value_coeff * value_loss
        metrics = {
            "value_loss": value_loss,
            "value_mean": value.mean(),
        }
        return total_loss, metrics
