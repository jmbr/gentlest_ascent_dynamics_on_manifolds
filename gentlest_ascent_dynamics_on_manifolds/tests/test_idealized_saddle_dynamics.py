import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from gentlest_ascent_dynamics_on_manifolds import IdealizedSaddleDynamics as ISD


def phi(x: jnp.ndarray) -> jnp.ndarray:
    """System of coordinates (from Cartesian to stereographic projection)."""
    return jnp.array([x[0], x[1]]) / (1 - x[2])


def psi(u: jnp.ndarray) -> jnp.ndarray:
    """Parameterization (from stereographic projection to Cartesian)."""
    return (jnp.array([2 * u[0], 2 * u[1], u[0]**2 + u[1]**2 - 1])
            / (u[0]**2 + u[1]**2 + 1))


def grad_U(u: jnp.ndarray) -> jnp.ndarray:
    """Gradient of potential energy (accounting for Riemannian metric)."""
    return (jnp.array([-u[1] * (3 * u[0]**4 + 2 * u[0]**2 * u[1]**2 - u[1]**4 - 8 * u[0]**2 + 1),
                       u[0] * (-3 * u[1]**4 + (-2 * u[0]**2 + 8) * u[1]**2 + u[0]**4 - 1)])
                      / (u[0]**2 + u[1]**2 + 1)**2)


def X(u: jnp.ndarray) -> jnp.ndarray:
    """Force (negated gradient of the potential energy."""
    return -grad_U(u)


def metric(u: jnp.ndarray) -> jnp.ndarray:
    """Riemannian metric on the stereographic projection."""
    return jnp.eye(2) * 4 / (u[0]**2 + u[1]**2 + 1)**2


def christoffel_symbols(u: jnp.ndarray) -> jnp.ndarray:
    denom = u[0]**2 + u[1]**2 + 1

    gamma = np.zeros((2, 2, 2))

    gamma[0, 0, 0] = -2 * u[0] / denom
    gamma[0, 1, 0] = -2 * u[1] / denom
    gamma[1, 0, 0] = gamma[0, 1, 0]
    gamma[1, 1, 0] =  2 * u[0] / denom

    gamma[0, 0, 1] =  2 * u[1] / denom
    gamma[0, 1, 1] = -2 * u[0] / denom
    gamma[1, 0, 1] = gamma[0, 1, 1]
    gamma[1, 1, 1] = -2 * u[1] / denom

    return gamma


def hessian(u: jnp.ndarray) -> jnp.ndarray:
    """Riemannian Hessian of the potential energy."""
    return jnp.array([[4 * u[1] * u[0] * (u[0]**4 - u[1]**4 - 11 * u[0]**2 - u[1]**2 + 6) / (u[0]**2 + u[1]**2 + 1)**3,
      (-u[0]**6 + (5 * u[1]**2 + 5) * u[0]**4 + (5 * u[1]**4 - 30 * u[1]**2 + 5) * u[0]**2 - u[1]**6 + 5 * u[1]**4 + 5 * u[1]**2 - 1) / (u[0]**2 + u[1]**2 + 1)**3],
     [(-u[0]**6 + (5 * u[1]**2 + 5) * u[0]**4 + (5 * u[1]**4 - 30 * u[1]**2 + 5) * u[0]**2 - u[1]**6 + 5 * u[1]**4 + 5 * u[1]**2 - 1) / (u[0]**2 + u[1]**2 + 1)**3,
      -4 * u[1] * u[0] * (u[0]**4 - u[1]**4 + u[0]**2 + 11 * u[1]**2 - 6) / (u[0]**2 + u[1]**2 + 1)**3]])


def make_points(a, b, n):
    """Return equally n space points in the square [a, b] × [a, b]."""
    u1, u2 = jnp.meshgrid(jnp.linspace(a, b, n), jnp.linspace(a, b, n))
    return jnp.stack((u1.flatten(), u2.flatten())).T


def test_isd_metric_christoffel_symbols_and_hessian():
    U = make_points(-2, 2, 10)
    isd = ISD(phi, psi, X)
    for i in range(U.shape[0]):
        u = U[i, :]
        assert jnp.allclose(isd.metric(u), metric(u))
        assert jnp.allclose(isd.christoffel_symbols(u), christoffel_symbols(u))
        assert jnp.allclose(isd.hessian(u), hessian(u))


def test_isd_vectors():
    points = make_points(-2, 2, 30)

    g_points = jax.vmap(metric)(points)

    hessian_points = jax.vmap(hessian)(points)
    eigenvalues, eigenvectors = jax.vmap(jnp.linalg.eigh)(hessian_points)
    # Collect all the directions of gentlest ascent.
    eigenvalue_permutation = jnp.argsort(eigenvalues, axis=1)
    v = jnp.array([eigenvectors[i, :, eigenvalue_permutation[i, 0]]
                   for i in range(eigenvectors.shape[0])])
    norm_v = jnp.sqrt(jnp.einsum('...i,...ij,...j', v, g_points, v))
    ascent_directions = v / norm_v[..., None]

    grad_U_points = jax.vmap(grad_U)(points)

    isd_vectors1 = (2 * (jnp.einsum('...i,...ij,...j',
                                    ascent_directions, g_points, grad_U_points)[..., None]
                         * ascent_directions) - grad_U_points)
    isd_vectors1 /= jnp.sqrt(jnp.einsum('...i,...ij,...j', isd_vectors1, g_points, isd_vectors1)[..., None])
    # isd_vectors1 /= jnp.linalg.norm(isd_vectors1, axis=1)[..., None]

    isd = ISD(phi, psi, X)
    isd_vectors2 = jax.vmap(isd)(points)
    # isd_vectors2 /= jnp.linalg.norm(isd_vectors2, axis=1)[..., None]

    assert jnp.allclose(isd_vectors1, isd_vectors2)
    
    # import matplotlib.pyplot as plt
    # plt.quiver(points[:, 0], points[:, 1], isd_vectors1[:, 0], isd_vectors1[:, 1], color='b')
    # plt.quiver(points[:, 0], points[:, 1], isd_vectors2[:, 0], isd_vectors2[:, 1], color='r', alpha=0.75)
