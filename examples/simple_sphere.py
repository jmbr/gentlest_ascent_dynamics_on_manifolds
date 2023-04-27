"""Example of a simple potential on a sphere."""

from itertools import count

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import jax
import jax.numpy as jnp

from simple_potential import potential, force, fixed_points
from utils import map_vectors, integrate

from gentlest_ascent_dynamics_on_manifolds import (
    DiffusionMapCoordinates,
    Metropolis,
    make_gaussian_process,
)


jax.config.update('jax_enable_x64', True)


# Parameters for sampler.
TEMPERATURE: float = 5e-2
DELTA: float = 5e-2
NUM_BATCHES: int = 100
NUM_SAMPLES_PER_BATCH: int = 10
SAMPLER_RANDOM_SEED: int = 23

# Parameters for integration of GAD/ISD vector field.
TIME_STEP_LENGTH: float = 1e-4
TIME_HORIZON: float = 1e-1

# Convergence criterion.
THRESHOLD: float = 1e-4

np.random.seed(SAMPLER_RANDOM_SEED)


class SphericalSampler(Metropolis):
    def _generate_candidate(self) -> jnp.ndarray:
        point = super()._generate_candidate()
        return point / jnp.linalg.norm(point)


def plot_sphere(ax, n=20):
    u1, u2 = jnp.meshgrid(
        jnp.linspace(0, 2 * jnp.pi, 2 * n),
        jnp.linspace(-jnp.pi / 2, jnp.pi / 2, n),
    )
    r = 0.95
    x1 = r * jnp.cos(u1) * jnp.cos(u2)
    x2 = r * jnp.sin(u1) * jnp.cos(u2)
    x3 = r * jnp.sin(u2)
    ax.plot_wireframe(x1, x2, x3)


plt.close('all')
sns.set_style('ticks')

fig = plt.figure()
gs = GridSpec(1, 2, figure=fig)
ax1 = fig.add_subplot(gs[0], projection='3d')
ax2 = fig.add_subplot(gs[1])

point = fixed_points[-1, :]
prev_point = None

sampler = SphericalSampler(
    potential,
    temperature=TEMPERATURE,
    delta=DELTA,
    initial_point=point,
)

for i in count():
    if prev_point is not None and jnp.allclose(
        prev_point, point, atol=THRESHOLD
    ):
        break

    prev_point = point.copy()

    points = sampler.sample(
        point,
        num_batches=NUM_BATCHES,
        num_samples_per_batch=NUM_SAMPLES_PER_BATCH,
    )
    energies = jax.vmap(potential)(points)

    vectors = jax.vmap(force)(points)

    phi = DiffusionMapCoordinates(3, 2)
    mapped_points = phi.learn(points)

    psi = make_gaussian_process(mapped_points, points)

    mapped_vectors = map_vectors(phi, points, vectors)
    mapped_vector_field = make_gaussian_process(mapped_points, mapped_vectors)

    mapped_path, path = integrate(
        phi,
        psi,
        mapped_vector_field,
        dt=TIME_STEP_LENGTH,
        tmax=TIME_HORIZON,
        initial_mapped_point=mapped_points[0, ...],
    )

    point = path[-1, :]

    ax1.clear()
    ax1.set_aspect('equal')
    ax1.set_xlim(-1.25, 1.25)
    ax1.set_ylim(-1.25, 1.25)
    ax1.set_zlim(-1.25, 1.25)
    ax1.set_box_aspect([1, 1, 1])
    ax1.set_xlabel('$x^1$')
    ax1.set_ylabel('$x^2$')
    ax1.set_zlabel('$x^3$')
    plot_sphere(ax1)
    ax1.scatter3D(
        fixed_points[:, 0],
        fixed_points[:, 1],
        fixed_points[:, 2],
        s=100,
        color='k',
    )
    ax1.scatter3D(
        path[:, 0],
        path[:, 1],
        path[:, 2],
        c=np.arange(path.shape[0]),
        cmap='Greens',
    )

    ax2.clear()
    ax2.set_xlabel('$\phi^1$')
    ax2.set_ylabel('$\phi^2$')
    ax2.quiver(
        mapped_points[:, 0],
        mapped_points[:, 1],
        mapped_vectors[:, 0],
        mapped_vectors[:, 1],
        color='k',
    )
    ax2.scatter(
        mapped_path[:, 0],
        mapped_path[:, 1],
        c=np.arange(mapped_path.shape[0]),
        cmap='Greens',
        alpha=0.75,
    )
    sns.despine()

    plt.pause(0.1)

plt.show()
