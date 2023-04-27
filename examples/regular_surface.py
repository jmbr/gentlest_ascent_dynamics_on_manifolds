"""Random regular surface."""

__all__ = ['Surface', 'SurfaceSampler', 'VectorField']

from functools import partial
from itertools import product

import numpy as np

import jax
import jax.numpy as jnp

import muller_brown

from gentlest_ascent_dynamics_on_manifolds import Metropolis


class Surface:
    """Random regular surface."""

    K: int
    seed: int
    amplitudes: np.array
    phases: np.array

    def __init__(self, K, amplitudes=None, phases=None):
        if amplitudes is not None or phases is not None:
            assert len(amplitudes) == len(phases)
            assert len(amplitudes) == np.sqrt(len(amplitudes))
            self.seed = 0
            self.amplitudes = amplitudes
            self.phases = phases
        else:
            self.generate(K)

    def generate(self, K):
        """Generate the coefficients of a random regular surface."""
        self.K = K
        m = K**2
        amplitudes = 2 * np.random.rand(m) - 1
        coins = np.random.rand(m)
        amplitudes[coins < 0.25] = 0.0
        self.amplitudes = amplitudes
        self.phases = np.random.rand(m)

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, x):
        K = self.K
        A = self.amplitudes
        B = self.phases
        value = 0.0
        for a, b, *k in zip(A, B, product(range(K), range(K))):
            k = jnp.array(k)
            value += a * jnp.cos((jnp.dot(k, x) + b))
        return value / K**2

    def plot(self, ax=None, colors=None):
        """Plot surface."""
        if ax is None:
            ax = plt.subplot(projection='3d')
        x1, x2 = np.meshgrid(
            np.linspace(-1.85, 5 / 4, 56), np.linspace(-0.5, 9 / 4, 50)
        )
        X = np.stack((x1.flatten(), x2.flatten())).T
        z = jax.vmap(self)(X).reshape(*x1.shape)
        ax.plot_surface(x1, x2, z, alpha=0.5)
        ax.set_xlabel('$x^1$')
        ax.set_ylabel('$x^2$')
        ax.set_zlabel('$x^3$')


class VectorField:
    """Vector field of Müller-Brown potential on a regular surface."""

    surface: Surface

    def __init__(self, surface: Surface) -> None:
        self.surface = surface

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, x):
        w = -jax.grad(muller_brown.potential)(x[:2])
        v = jax.jacobian(self.surface)(x[:2]) @ w
        return jnp.hstack((w, v))


class Potential:
    """Potential used to sample from a regular surface."""

    kappa: float
    surface: Surface

    def __init__(self, spring_constant, surface):
        self.kappa = spring_constant
        self.surface = surface

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, x):
        return self.kappa / 2.0 * (x[2] - self.surface(x[:2])) ** 2


class SurfaceSampler(Metropolis):
    """Metropolis sampler on a regular surface."""

    def __init__(
        self,
        *,
        surface: Surface,
        spring_constant: float,
        temperature: float,
        delta: float,
        initial_point: np.ndarray,
    ) -> None:
        self.potential = Potential(spring_constant, surface)
        super().__init__(self.potential, temperature, delta, initial_point)
