import numpy as np
import matplotlib.pyplot as plt

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import diffrax

from gentlest_ascent_dynamics_on_manifolds \
    import (DiffusionMapCoordinates, IdealizedSaddleDynamics,
            SphericalSampler, equilibria, make_gaussian_process,
            potential)


temperature: float = 5e-2
delta: float = 5e-2
random_seed: int = 23
num_batches: int = 100
num_samples_per_batch: int = 10


np.random.seed(random_seed)


def make_vector_field(coordinates, points):
    forces = -jax.vmap(jax.jacobian(potential))(points)
    mapped_points, mapped_forces = jax.vmap(
        lambda x, v: jax.jvp(coordinates, (x,), (v,)))(points, forces)
    return make_gaussian_process(mapped_points, mapped_forces)


initial_point = equilibria[-1, :]

plt.close()
plt.figure(figsize=(7.5, 5), dpi=200)

for i in range(10):
    print(f'Iteration #{i}', flush=True)
    sampler = SphericalSampler(potential, temperature=temperature,
                               delta=delta, initial_point=initial_point)
    X = sampler.sample_batches(num_batches=num_batches,
                               num_samples_per_batch=num_samples_per_batch)
    energies = jax.vmap(potential)(X)

    phi = DiffusionMapCoordinates(3, 2)
    U = phi.learn(X)

    psi = make_gaussian_process(U, X)

    vector_field = make_vector_field(phi, X)
    V = jax.vmap(vector_field)(U)

    idealized_saddle_dynamics = IdealizedSaddleDynamics(phi, psi, vector_field)
    W = jax.vmap(idealized_saddle_dynamics)(U)

    term = diffrax.ODETerm(lambda t, u, args: idealized_saddle_dynamics(u))
    solver = diffrax.Euler()
    solution = diffrax.diffeqsolve(term, solver, t0=0.0, t1=0.1,
                                   dt0=1e-4, y0=U[0, :], max_steps=8192,
                                   saveat=diffrax.SaveAt(steps=True))
    steps = solution.stats['num_accepted_steps']
    Z = solution.ys[:steps]
    Z0 = jax.vmap(psi)(Z)

    initial_point = Z0[-1, :]

    plt.clf()

    plt.title(f'Iteration #{i:2d}')

    ax = plt.gcf().add_subplot(1, 2, 1, projection='3d')
    ax.set_aspect('equal')
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_zlim(-1.25, 1.25)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('$x^1$')
    ax.set_ylabel('$x^2$')
    ax.set_zlabel('$x^3$')
    ax.scatter3D(equilibria[:, 0], equilibria[:, 1], equilibria[:, 2],
                 s=100, color='k')
    ax.plot(Z0[:, 0], Z0[:, 1], Z0[:, 2], c='b')

    ax = plt.gcf().add_subplot(1, 2, 2)
    ax.quiver(U[:, 0], U[:, 1], V[:, 0], V[:, 1], color='g', alpha=0.25)
    ax.quiver(U[:, 0], U[:, 1], W[:, 0], W[:, 1], color='k', alpha=0.75)

    ax.plot(Z[:, 0], Z[:, 1], lw=4, alpha=0.9)
    plt.tight_layout()
    plt.pause(1)
