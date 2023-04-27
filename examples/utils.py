"""Miscellaneous utilities."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import diffrax

from gentlest_ascent_dynamics_on_manifolds import (
    DiffusionMapCoordinates,
    IdealizedSaddleDynamics,
)


def map_vectors(
    phi: DiffusionMapCoordinates,
    points: Float[Array, 'N n'],
    vectors: Float[Array, 'N n'],
):
    """Map vectors from the ambient space to the chart."""
    return jnp.einsum(
        '...ij,...j', jax.vmap(jax.jacobian(phi))(points), vectors
    )


def integrate(
    phi, psi, mapped_vector_field, *, dt, tmax, initial_mapped_point
):
    """Integrate GAD/ISD vector field."""
    idealized_saddle_dynamics = IdealizedSaddleDynamics(
        phi, psi, mapped_vector_field
    )

    term = diffrax.ODETerm(lambda t, u, args: idealized_saddle_dynamics(u))
    solver = diffrax.Euler()
    solution = diffrax.diffeqsolve(
        term,
        solver,
        dt0=dt,
        t0=0.0,
        t1=tmax,
        y0=initial_mapped_point,
        max_steps=8192,
        saveat=diffrax.SaveAt(steps=True),
    )
    steps = solution.stats["num_accepted_steps"]
    mapped_path = solution.ys[:steps]

    path = jax.vmap(psi)(mapped_path)

    return mapped_path, path
