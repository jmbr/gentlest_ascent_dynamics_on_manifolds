"""Example of Müller-Brown potential mapped onto a random regular surface."""

from itertools import count

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import jax
import jax.numpy as jnp

from muller_brown import fixed_points, plot_potential
from regular_surface import Surface, SurfaceSampler, VectorField
from utils import map_vectors, integrate

from gentlest_ascent_dynamics_on_manifolds import (
    DiffusionMapCoordinates,
    make_gaussian_process,
)


jax.config.update('jax_enable_x64', True)


# Parameters for sampler.
SPRING_CONSTANT: float = 1e4
DELTA: float = 2.5e-2
NUM_BATCHES: int = 500
NUM_SAMPLES_PER_BATCH: int = 10
SAMPLER_RANDOM_SEED: int = 42

# Parameters for GP regression.
EPSILON: float = 1e-4
SIGMA: float = 1e-2

# Parameters for integration of GAD/ISD vector field.
TIME_STEP_LENGTH: float = 1e-4
TIME_HORIZON: float = 5e-2

# Convergence criterion.
THRESHOLD: float = 1e-4

# Random seeds for regular surface.
SURFACE_RANDOM_SEED: int = 555


def make_initial_point(surface, point):
    """Create initial condition on the regular surface."""
    return np.array([point[0], point[1], float(surface(point))])


plt.close('all')
sns.set_style('ticks')

np.random.seed(SURFACE_RANDOM_SEED)

surface = Surface(4)
vector_field = VectorField(surface)
point = make_initial_point(
    surface, fixed_points[0, ...] - jnp.array([0.1, 0.0])
)

np.random.seed(SAMPLER_RANDOM_SEED)
sampler = SurfaceSampler(
    spring_constant=SPRING_CONSTANT,
    surface=surface,
    temperature=1.0,
    delta=DELTA,
    initial_point=point,
)

path = jnp.array([point])

fig = plt.figure()
gs = GridSpec(1, 3, figure=fig)
ax1 = fig.add_subplot(gs[0], projection='3d')
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

for i in count():
    if jnp.allclose(path[-1, :2], fixed_points[1, ...], atol=THRESHOLD):
        break

    points = sampler.sample(
        point,
        num_batches=NUM_BATCHES,
        num_samples_per_batch=NUM_SAMPLES_PER_BATCH,
    )

    vectors = jax.vmap(vector_field)(points)

    phi = DiffusionMapCoordinates(3, 2)
    mapped_points = phi.learn(points)

    psi = make_gaussian_process(
        mapped_points, points, epsilon=EPSILON, sigma=SIGMA
    )

    mapped_vectors = map_vectors(phi, points, vectors)

    mapped_vector_field = make_gaussian_process(
        mapped_points, mapped_vectors, epsilon=EPSILON, sigma=SIGMA
    )

    mapped_path, path = integrate(
        phi,
        psi,
        mapped_vector_field,
        dt=TIME_STEP_LENGTH,
        tmax=TIME_HORIZON,
        initial_mapped_point=mapped_points[0, ...],
    )

    point = path[-1, ...]

    ax1.clear()
    surface.plot(ax=ax1)
    ax1.scatter3D(
        path[:, 0],
        path[:, 1],
        path[:, 2],
        c=np.arange(path.shape[0]),
        cmap='Greens',
        s=200,
    )
    ax1.set_title('Regular surface')

    ax2.clear()
    plot_potential(ax=ax2, n=100)
    ax2.scatter(
        path[:, 0],
        path[:, 1],
        c=np.arange(path.shape[0]),
        cmap='Greens',
        alpha=0.75,
    )
    sns.despine()
    ax2.set_title('Müller-Brown potential')

    ax3.clear()
    ax3.quiver(
        mapped_points[:, 0],
        mapped_points[:, 1],
        mapped_vectors[:, 0],
        mapped_vectors[:, 1],
    )
    ax3.scatter(
        mapped_path[:, 0],
        mapped_path[:, 1],
        c=np.arange(path.shape[0]),
        cmap='Greens',
        alpha=0.75,
    )
    ax3.set_xlabel('$\phi^1$')
    ax3.set_ylabel('$\phi^2$')
    ax3.set_title('Diffusion map coordinates')
    ax3.set_aspect('equal')
    sns.despine()

    plt.pause(0.1)

plt.show()
