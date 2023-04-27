"""Simple potential on the 2-sphere."""

__all__ = ['potential', 'force', 'fixed_points']

from math import sqrt
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def potential(x: Float[Array, '3']) -> float:
    """Simple potential."""
    return jnp.prod(x)


def force(x: Float[Array, '3']) -> Float[Array, '3']:
    """Vector field corresponding to simple potential on the sphere."""
    F = -jax.grad(potential)(x)
    return F - x * jnp.dot(x, F)


fixed_points = jnp.array(
    [
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
        (
            (-1 - sqrt(3)) / (3 + sqrt(3)),
            (-1 - sqrt(3)) / (3 + sqrt(3)),
            (1 + sqrt(3)) / (3 + sqrt(3)),
        ),
        (
            (-1 - sqrt(3)) / (3 + sqrt(3)),
            (1 + sqrt(3)) / (3 + sqrt(3)),
            (1 + sqrt(3)) / (3 + sqrt(3)),
        ),
        (
            (1 - sqrt(3)) / (-3 + sqrt(3)),
            (1 - sqrt(3)) / (-3 + sqrt(3)),
            (sqrt(3) - 1) / (-3 + sqrt(3)),
        ),
        (
            (1 - sqrt(3)) / (-3 + sqrt(3)),
            (sqrt(3) - 1) / (-3 + sqrt(3)),
            (sqrt(3) - 1) / (-3 + sqrt(3)),
        ),
        (
            (1 + sqrt(3)) / (3 + sqrt(3)),
            (-1 - sqrt(3)) / (3 + sqrt(3)),
            (1 + sqrt(3)) / (3 + sqrt(3)),
        ),
        (
            (1 + sqrt(3)) / (3 + sqrt(3)),
            (1 + sqrt(3)) / (3 + sqrt(3)),
            (1 + sqrt(3)) / (3 + sqrt(3)),
        ),
        (
            (sqrt(3) - 1) / (-3 + sqrt(3)),
            (1 - sqrt(3)) / (-3 + sqrt(3)),
            (sqrt(3) - 1) / (-3 + sqrt(3)),
        ),
        (
            (sqrt(3) - 1) / (-3 + sqrt(3)),
            (sqrt(3) - 1) / (-3 + sqrt(3)),
            (sqrt(3) - 1) / (-3 + sqrt(3)),
        ),
    ]
)


def classify_fixed_points():
    def phi(x: jnp.ndarray) -> jnp.ndarray:
        """System of coordinates (from Cartesian to stereographic projection)."""
        return jnp.array([x[0], x[1]]) / (1 - x[2])

    def hessian(u: jnp.ndarray) -> jnp.ndarray:
        """Riemannian Hessian of the potential energy."""
        return jnp.array(
            [
                [
                    4
                    * u[1]
                    * u[0]
                    * (u[0] ** 4 - u[1] ** 4 - 11 * u[0] ** 2 - u[1] ** 2 + 6)
                    / (u[0] ** 2 + u[1] ** 2 + 1) ** 3,
                    (
                        -u[0] ** 6
                        + (5 * u[1] ** 2 + 5) * u[0] ** 4
                        + (5 * u[1] ** 4 - 30 * u[1] ** 2 + 5) * u[0] ** 2
                        - u[1] ** 6
                        + 5 * u[1] ** 4
                        + 5 * u[1] ** 2
                        - 1
                    )
                    / (u[0] ** 2 + u[1] ** 2 + 1) ** 3,
                ],
                [
                    (
                        -u[0] ** 6
                        + (5 * u[1] ** 2 + 5) * u[0] ** 4
                        + (5 * u[1] ** 4 - 30 * u[1] ** 2 + 5) * u[0] ** 2
                        - u[1] ** 6
                        + 5 * u[1] ** 4
                        + 5 * u[1] ** 2
                        - 1
                    )
                    / (u[0] ** 2 + u[1] ** 2 + 1) ** 3,
                    -4
                    * u[1]
                    * u[0]
                    * (u[0] ** 4 - u[1] ** 4 + u[0] ** 2 + 11 * u[1] ** 2 - 6)
                    / (u[0] ** 2 + u[1] ** 2 + 1) ** 3,
                ],
            ]
        )

    fixed_points3d = fixed_points[
        [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], ...
    ]
    fixed_points = jax.vmap(phi)(fixed_points3d)
    hessians = jax.vmap(hessian)(fixed_points)
    eigenvalues, _ = jax.vmap(jnp.linalg.eigh)(hessians)
    sums = jnp.sum(jnp.sign(eigenvalues), axis=1, dtype=jnp.int64)
    saddle_points = fixed_points3d[sums == 0]
    sink_points = fixed_points3d[sums == -2]
    source_points = fixed_points3d[sums == 2]

    return source_points, sink_points, saddle_points
