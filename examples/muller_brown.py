"""Müller-Brown potential."""

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp


@jax.jit
def potential(x):
    """Müller-Brown potential."""
    return (
        -200.0 * jnp.exp(-((x[0] - 1.0) ** 2) - 10.0 * x[1] ** 2)
        - 100.0 * jnp.exp(-x[0] ** 2 - 10.0 * (x[1] - 1 / 2) ** 2)
        - 170.0
        * jnp.exp(
            -(13 / 2) * (x[0] + 1 / 2) ** 2
            + 11.0 * (x[0] + 1 / 2) * (x[1] - 3 / 2)
            - (13 / 2) * (x[1] - 3 / 2) ** 2
        )
        + 15.0
        * jnp.exp(
            (7 / 10) * (x[0] + 1) ** 2
            + (3 / 5) * (x[0] + 1.0) * (x[1] - 1.0)
            + (7 / 10) * (x[1] - 1.0) ** 2
        )
    )


fixed_points = jnp.array(
    [
        [0.623499404930877, 0.0280377585286857],
        [0.212486582000662, 0.292988325107368],
        [-0.0500108229982061, 0.466694104871972],
        [-0.822001558732732, 0.624312802814871],
        [-0.558223634633024, 1.44172584180467],
    ]
)


def plot_potential(ax=None, n=50):
    """Plot Muller-Brown potential in 2D."""
    if ax is None:
        ax = plt.gca()

    x1a, x1b = -1.85, 5 / 4
    x2a, x2b = -0.5, 9 / 4
    aspect_ratio = (x1b - x1a) / (x2b - x2a)

    x1, x2 = jnp.meshgrid(
        jnp.linspace(x1a, x1b, int(aspect_ratio * n)),
        jnp.linspace(x2a, x2b, n),
    )
    X = jnp.stack((x1.flatten(), x2.flatten())).T
    z = jax.vmap(potential)(X).reshape(*x1.shape)

    ax.pcolormesh(x1, x2, z, vmin=-150, vmax=20, cmap='Blues_r')
    ax.set_aspect(aspect_ratio)
    ax.set_xlabel('$x^1$')
    ax.set_ylabel('$x^2$')
