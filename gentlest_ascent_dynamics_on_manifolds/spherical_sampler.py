from math import sqrt

import jax.numpy as jnp

from .metropolis import Metropolis


class SphericalSampler(Metropolis):
    def _generate_candidate(self) -> jnp.ndarray:
        point = super()._generate_candidate()
        return point / jnp.linalg.norm(point)

    def sample_batches(self, num_batches: int,
                       num_samples_per_batch: int) -> jnp.ndarray:
            X = None

            for k in range(num_batches):
                self.reset()

                samples = self.draw_samples(num_samples_per_batch)

                if X is None:
                    X = samples.copy()
                else:
                    X = jnp.vstack((X, samples))

            return X


def potential(x: jnp.ndarray) -> float:
    return jnp.prod(x)


equilibria = jnp.array([
    (-1, 0, 0),
    (0, -1, 0),
    (0, 0, -1),
    (0, 0, 1),
    (0, 1, 0),
    (1, 0, 0),
    ((-1 - sqrt(3)) / (3 + sqrt(3)), (-1 - sqrt(3)) / (3 + sqrt(3)), (1 + sqrt(3)) / (3 + sqrt(3))),
    ((-1 - sqrt(3)) / (3 + sqrt(3)), (1 + sqrt(3)) / (3 + sqrt(3)), (1 + sqrt(3)) / (3 + sqrt(3))),
    ((1 - sqrt(3)) / (-3 + sqrt(3)), (1 - sqrt(3)) / (-3 + sqrt(3)), (sqrt(3) - 1) / (-3 + sqrt(3))),
    ((1 - sqrt(3)) / (-3 + sqrt(3)), (sqrt(3) - 1) / (-3 + sqrt(3)), (sqrt(3) - 1) / (-3 + sqrt(3))),
    ((1 + sqrt(3)) / (3 + sqrt(3)), (-1 - sqrt(3)) / (3 + sqrt(3)), (1 + sqrt(3)) / (3 + sqrt(3))),
    ((1 + sqrt(3)) / (3 + sqrt(3)), (1 + sqrt(3)) / (3 + sqrt(3)), (1 + sqrt(3)) / (3 + sqrt(3))),
    ((sqrt(3) - 1) / (-3 + sqrt(3)), (1 - sqrt(3)) / (-3 + sqrt(3)), (sqrt(3) - 1) / (-3 + sqrt(3))),
    ((sqrt(3) - 1) / (-3 + sqrt(3)), (sqrt(3) - 1) / (-3 + sqrt(3)), (sqrt(3) - 1) / (-3 + sqrt(3)))])


def classify_equilibria():
    import jax
    import jax.numpy as jnp

    equilibria = jnp.array([
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
        ((-1 - sqrt(3)) / (3 + sqrt(3)), (-1 - sqrt(3)) / (3 + sqrt(3)), (1 + sqrt(3)) / (3 + sqrt(3))),
        ((-1 - sqrt(3)) / (3 + sqrt(3)), (1 + sqrt(3)) / (3 + sqrt(3)), (1 + sqrt(3)) / (3 + sqrt(3))),
        ((1 - sqrt(3)) / (-3 + sqrt(3)), (1 - sqrt(3)) / (-3 + sqrt(3)), (sqrt(3) - 1) / (-3 + sqrt(3))),
        ((1 - sqrt(3)) / (-3 + sqrt(3)), (sqrt(3) - 1) / (-3 + sqrt(3)), (sqrt(3) - 1) / (-3 + sqrt(3))),
        ((1 + sqrt(3)) / (3 + sqrt(3)), (-1 - sqrt(3)) / (3 + sqrt(3)), (1 + sqrt(3)) / (3 + sqrt(3))),
        ((1 + sqrt(3)) / (3 + sqrt(3)), (1 + sqrt(3)) / (3 + sqrt(3)), (1 + sqrt(3)) / (3 + sqrt(3))),
        ((sqrt(3) - 1) / (-3 + sqrt(3)), (1 - sqrt(3)) / (-3 + sqrt(3)), (sqrt(3) - 1) / (-3 + sqrt(3))),
        ((sqrt(3) - 1) / (-3 + sqrt(3)), (sqrt(3) - 1) / (-3 + sqrt(3)), (sqrt(3) - 1) / (-3 + sqrt(3)))])

    def phi(x: jnp.ndarray) -> jnp.ndarray:
        """System of coordinates (from Cartesian to stereographic projection)."""
        return jnp.array([x[0], x[1]]) / (1 - x[2])

    def hessian(u: jnp.ndarray) -> jnp.ndarray:
        """Riemannian Hessian of the potential energy."""
        return jnp.array([[4 * u[1] * u[0] * (u[0]**4 - u[1]**4 - 11 * u[0]**2 - u[1]**2 + 6) / (u[0]**2 + u[1]**2 + 1)**3,
          (-u[0]**6 + (5 * u[1]**2 + 5) * u[0]**4 + (5 * u[1]**4 - 30 * u[1]**2 + 5) * u[0]**2 - u[1]**6 + 5 * u[1]**4 + 5 * u[1]**2 - 1) / (u[0]**2 + u[1]**2 + 1)**3],
         [(-u[0]**6 + (5 * u[1]**2 + 5) * u[0]**4 + (5 * u[1]**4 - 30 * u[1]**2 + 5) * u[0]**2 - u[1]**6 + 5 * u[1]**4 + 5 * u[1]**2 - 1) / (u[0]**2 + u[1]**2 + 1)**3,
          -4 * u[1] * u[0] * (u[0]**4 - u[1]**4 + u[0]**2 + 11 * u[1]**2 - 6) / (u[0]**2 + u[1]**2 + 1)**3]])

    equilibria3d = equilibria[[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], :]
    equilibria = jax.vmap(phi)(equilibria3d)
    hessians = jax.vmap(hessian)(equilibria)
    eigenvalues, eigenvectors = jax.vmap(jnp.linalg.eigh)(hessians)
    sums = jnp.sum(jnp.sign(eigenvalues), axis=1, dtype=jnp.int64)
    saddle_points = equilibria[sums == 0]
    sink_points = equilibria[sums == -2]
    source_points = equilibria[sums == 2]
    return source_points, sink_points, saddle_points
