from IPython import get_ipython
get_ipython().run_line_magic('reset', '-fs')
get_ipython().run_line_magic('matplotlib', 'qt')

from typing import Callable

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm


def phi(x: jnp.ndarray) -> jnp.ndarray:
    """System of coordinates (from Cartesian to stereographic projection)."""
    return jnp.array([x[0], x[1]]) / (1 - x[2])


def psi(u: jnp.ndarray) -> jnp.ndarray:
    """Parameterization (from stereographic projection to Cartesian)."""
    return (jnp.array([2 * u[0], 2 * u[1], u[0]**2 + u[1]**2 - 1])
            / (u[0]**2 + u[1]**2 + 1))


def g(u: jnp.ndarray) -> jnp.ndarray:
    """Riemannian metric on the stereographic projection."""
    return jnp.eye(2) * 4 / (u[0]**2 + u[1]**2 + 1)**2


def U(u: jnp.ndarray) -> float:
    """Potential energy."""
    return ((4 * u[0] * u[1] * (u[0]**2 + u[1]**2 - 1))
            / (u[0]**2 + u[1]**2 + 1)**3)


def grad_U(u: jnp.ndarray) -> jnp.ndarray:
    """Gradient of potential energy (accounting for Riemannian metric)."""
    return (jnp.array([-u[1] * (3 * u[0]**4 + 2 * u[0]**2 * u[1]**2 - u[1]**4 - 8 * u[0]**2 + 1),
                       u[0] * (-3 * u[1]**4 + (-2 * u[0]**2 + 8) * u[1]**2 + u[0]**4 - 1)])
                      / (u[0]**2 + u[1]**2 + 1)**2)


def plot_energy(a, b, n):
    u1, u2 = jnp.meshgrid(jnp.linspace(a, b, n),
                          jnp.linspace(a, b, n))
    points = jnp.stack((u1.flatten(), u2.flatten())).T
    energies = jax.vmap(U)(points)
    plt.pcolormesh(u1, u2, energies.reshape(*u1.shape), cmap='Blues',
                   rasterized=True)
    plt.colorbar(label='Energy')
    plt.gca().set_aspect('equal')


def plot_force(a, b, n):
    u1, u2 = jnp.meshgrid(jnp.linspace(a, b, n), jnp.linspace(a, b, n))
    points = jnp.stack((u1.flatten(), u2.flatten())).T
    # forces = -jax.vmap(jax.grad(U))(points)
    forces = -jax.vmap(grad_U)(points)
    plt.quiver(points[:, 0], points[:, 1], forces[:, 0], forces[:, 1], scale=20)


def hessian(u: jnp.ndarray) -> jnp.ndarray:
    """Riemannian Hessian of the potential energy."""
    return jnp.array([[4 * u[1] * u[0] * (u[0]**4 - u[1]**4 - 11 * u[0]**2 - u[1]**2 + 6) / (u[0]**2 + u[1]**2 + 1)**3,
      (-u[0]**6 + (5 * u[1]**2 + 5) * u[0]**4 + (5 * u[1]**4 - 30 * u[1]**2 + 5) * u[0]**2 - u[1]**6 + 5 * u[1]**4 + 5 * u[1]**2 - 1) / (u[0]**2 + u[1]**2 + 1)**3],
     [(-u[0]**6 + (5 * u[1]**2 + 5) * u[0]**4 + (5 * u[1]**4 - 30 * u[1]**2 + 5) * u[0]**2 - u[1]**6 + 5 * u[1]**4 + 5 * u[1]**2 - 1) / (u[0]**2 + u[1]**2 + 1)**3,
      -4 * u[1] * u[0] * (u[0]**4 - u[1]**4 + u[0]**2 + 11 * u[1]**2 - 6) / (u[0]**2 + u[1]**2 + 1)**3]])


a, b, n = -2, 2, 40
u1, u2 = jnp.meshgrid(jnp.linspace(a, b, n), jnp.linspace(a, b, n))
points = jnp.stack((u1.flatten(), u2.flatten())).T

g_points = jax.vmap(g)(points)

hessian_points = jax.vmap(hessian)(points)
eigenvalues, eigenvectors = jax.vmap(jnp.linalg.eigh)(hessian_points)
# Collect all the directions of gentlest ascent.
eigenvalue_permutation = jnp.argsort(eigenvalues, axis=1)
v = jnp.array([eigenvectors[i, :, eigenvalue_permutation[i, 0]]
               for i in range(eigenvectors.shape[0])])
norm_v = jnp.sqrt(jnp.einsum('...i,...ij,...j', v, g_points, v))
ascent_directions = v / norm_v[..., None]

grad_U_points = jax.vmap(grad_U)(points)

isd_vectors = (2 * (jnp.einsum('...i,...ij,...j',
                               ascent_directions, g_points, grad_U_points)[..., None]
                    * ascent_directions) - grad_U_points)
isd_vectors /= jnp.einsum('...i,...ij,...j', isd_vectors, g_points, isd_vectors)[..., None]
isd_vectors /= jnp.linalg.norm(isd_vectors, axis=1)[..., None]

from gentlest_ascent_dynamics_on_manifolds.spherical_sampler import classify_equilibria

source_points, sink_points, saddle_points = classify_equilibria()

fig = plt.figure(figsize=(5, 5))
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "cm",
})
ax = fig.gca()
plot_energy(-2, 2, 100)
plt.quiver(points[:, 0], points[:, 1], isd_vectors[:, 0], isd_vectors[:, 1])
plt.scatter(saddle_points[:, 0], saddle_points[:, 1], marker='s', color='black', s=150)
plt.scatter(sink_points[:, 0], sink_points[:, 1], marker='o', color='black', s=150)
plt.scatter(source_points[:, 0], source_points[:, 1], marker='^', color='black', s=150)
plt.xticks(np.linspace(-2, 2, 5))
plt.yticks(np.linspace(-2, 2, 5))
plt.xlabel('$u^1$')
plt.ylabel('$u^2$')
plt.title('Idealized saddle-point dynamics')
plt.show()
