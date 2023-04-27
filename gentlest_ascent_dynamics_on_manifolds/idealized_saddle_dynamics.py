"""Idealized saddle dynamics."""

__all__ = ['IdealizedSaddleDynamics']

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp


class IdealizedSaddleDynamics:
    """Idealized saddle dynamics."""

    def __init__(
        self,
        phi: Callable[[jnp.ndarray], jnp.ndarray],
        psi: Callable[[jnp.ndarray], jnp.ndarray],
        vector_field: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> None:
        self.phi = phi  # System of coordinates
        self.Dphi = jax.jacobian(phi)

        self.psi = psi  # Parameterization (psi = phi⁻¹)
        self.Dpsi = jax.jacobian(psi)

        self.Y = vector_field
        self.DY = jax.jacobian(vector_field)

    @partial(jax.jit, static_argnums=0)
    def metric(self, u: jnp.ndarray) -> jnp.ndarray:
        """Return the first fundamental form (Riemannian metric)."""
        Dpsi = self.Dpsi(u)
        return Dpsi.T @ Dpsi

    @partial(jax.jit, static_argnums=0)
    def christoffel_symbols(self, u: jnp.ndarray) -> jnp.ndarray:
        """Return the Christoffel symbols of the Levi-Civita connection."""
        g = self.metric(u)
        g_inv = jnp.linalg.inv(g)
        Dg = jax.jacobian(self.metric)(u)
        return (
            -jnp.einsum('ijl,lk', Dg, g_inv)
            + jnp.einsum('lij,lk', Dg, g_inv)
            + jnp.einsum('jli,lk', Dg, g_inv)
        ) / 2

    @partial(jax.jit, static_argnums=0)
    def hessian(self, u: jnp.ndarray) -> jnp.ndarray:
        """Return the Hessian matrix of the gradient field."""
        Y = self.Y(u)
        DY = self.DY(u)
        Gamma = self.christoffel_symbols(u)
        return -(DY.T + jnp.einsum('ijk,i', Gamma, Y))

    @partial(jax.jit, static_argnums=0)
    def __call__(self, u: jnp.ndarray) -> jnp.ndarray:
        """Returns the ascent direction."""
        hessian = self.hessian(u)
        _, eigenvectors = jnp.linalg.eigh(hessian)
        v = eigenvectors[:, 0]

        g = self.metric(u)
        ascent_direction = v / jnp.sqrt(jnp.dot(v, g @ v))

        grad_U = -self.Y(u)

        isd_vector = (
            2 * jnp.dot(ascent_direction, g @ grad_U) * ascent_direction
            - grad_U
        )
        isd_vector /= jnp.sqrt(jnp.dot(isd_vector, g @ isd_vector))

        return isd_vector
