import math

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from gentlest_ascent_dynamics_on_manifolds.spherical_sampler \
    import SphericalSampler, equilibria, potential


num_samples: int = 50000


def test_spherical_sampler():
    spherical_sampler = SphericalSampler(potential, temperature=1, delta=1,
                                         initial_point=equilibria[-1, :])
    samples = spherical_sampler.draw_samples(num_samples)
    energies = jax.vmap(potential)(samples)

    mean_energy = jnp.mean(energies)
    assert -0.2 <= mean_energy <= 0.2
    assert jnp.allclose(mean_energy, 0.0, atol=1e-1)

    assert jnp.allclose(jnp.linalg.norm(samples, axis=1),
                        jnp.ones(num_samples))
    assert jnp.allclose(jnp.mean(samples, axis=0), jnp.zeros(3),
                        atol=1e-1)
