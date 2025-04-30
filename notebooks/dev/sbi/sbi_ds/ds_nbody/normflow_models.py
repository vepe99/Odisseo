import jax
import jax.numpy as jnp
from jax import vmap, jit, pmap
from jax import random
import optax
import numpyro.distributions as dist

from functools import partial
import haiku as hk
from tensorflow_probability.substrates import jax as tfp
from jaxopt import Bisection
from jaxopt.linear_solve import solve_normal_cg

# tfp = tfp.substrates.jax
tfb = tfp.bijectors
tfd = tfp.distributions


# This module is to store our implicit inverse functions
@partial(jax.custom_vjp, nondiff_argnums=(0,))
def root_bisection(f, params):
    """
    f: optimality fn with input arg (params, x)
    """
    bisec = Bisection(
        optimality_fun=f,
        lower=0.0,
        upper=1.0,
        check_bracket=False,
        maxiter=100,
        tol=1e-06,
    )
    return bisec.run(None, params).params


def root_bisection_fwd(f, params):
    z_star = root_bisection(f, params)
    return z_star, (params, z_star)


def root_bwd(f, res, z_star_bar):
    params, z_star = res
    _, vjp_a = jax.vjp(lambda p: f(z_star, p), params)
    _, vjp_z = jax.vjp(lambda z: f(z, params), z_star)
    return vjp_a(solve_normal_cg(lambda u: vjp_z(u)[0], -z_star_bar))


root_bisection.defvjp(root_bisection_fwd, root_bwd)


def make_inverse_fn(f):
    """Defines the inverse of the input function, and provides implicit gradients
    of the inverse.

    Args:
      f: callable of input shape (params, x)
    Retuns:
      inv_f: callable of with args (params, y)
    """

    def inv_fn(params, y):
        def optimality_fn(x, params):
            p, y = params
            return f(p, x) - y

        return root_bisection(optimality_fn, [params, y])

    return inv_fn

# Bijiector functions
class MixtureAffineSigmoidBijector(tfp.bijectors.Bijector):
    """
    Bijector based on a ramp function, and implemented using an implicit
    layer.
    This implementation is based on the Smooth Normalizing Flows described
    in: https://arxiv.org/abs/2110.00351
    """

    def __init__(self, a, b, c, p, name="MixtureAffineSigmoidBijector"):
        """
        Args:
          rho: function of x that defines a ramp function between 0 and 1
          a,b,c: scalar parameters of the coupling layer.
        """
        super(self.__class__, self).__init__(forward_min_event_ndims=0, name=name)
        self.a = a
        self.b = b
        self.c = c
        self.p = p

        def sigmoid(x, a, b, c):
            z = (jax.scipy.special.logit(x) + b) * a
            y = jax.nn.sigmoid(z) * (1 - c) + c * x
            return y

        # Rescaled bijection
        def f(params, x):
            a, b, c, p = params
            a_in, b_in = [0.0 - 1e-1, 1.0 + 1e-1]

            x = (x - a_in) / (b_in - a_in)
            x0 = (jnp.zeros_like(x) - a_in) / (b_in - a_in)
            x1 = (jnp.ones_like(x) - a_in) / (b_in - a_in)

            y = sigmoid(x, a, b, c)
            y0 = sigmoid(x0, a, b, c)
            y1 = sigmoid(x1, a, b, c)

            y = (y - y0) / (y1 - y0)
            return jnp.sum(p * (y * (1 - c) + c * x), axis=0)

        self.f = f

        # Inverse bijector
        self.inv_f = make_inverse_fn(f)

    def _forward(self, x):
        return jax.vmap(jax.vmap(self.f))([self.a, self.b, self.c, self.p], x)

    def _inverse(self, y):
        return jax.vmap(jax.vmap(self.inv_f))([self.a, self.b, self.c, self.p], y)

    def _forward_log_det_jacobian(self, x):
        def logdet_fn(x, a, b, c, p):
            g = jax.grad(self.f, argnums=1)([a, b, c, p], x)
            s, logdet = jnp.linalg.slogdet(jnp.atleast_2d(g))
            return s * logdet

        return jax.vmap(jax.vmap(logdet_fn))(x, self.a, self.b, self.c, self.p)


#Coupling layers
class AffineCoupling(hk.Module):
    def __init__(
        self, y, *args, layers=[128, 128], activation=jax.nn.leaky_relu, **kwargs
    ):
        """
        Args:
        y, conditioning variable
        layers, list of hidden layers
        activation, activation function for hidden layers
        """
        self.y = y
        self.layers = layers
        self.activation = activation
        super().__init__(*args, **kwargs)

    def __call__(self, x, output_units, **condition_kwargs):
        net = jnp.concatenate([x, self.y], axis=-1)
        for i, layer_size in enumerate(self.layers):
            net = self.activation(hk.Linear(layer_size, name="layer%d" % i)(net))

        shifter = tfb.Shift(hk.Linear(output_units)(net))
        scaler = tfb.Scale(jnp.clip(jnp.exp(hk.Linear(output_units)(net)), 1e-2, 1e2))
        return tfb.Chain([shifter, scaler])


class AffineSigmoidCoupling(hk.Module):
    """This is the coupling layer used in the Flow."""

    def __init__(
        self,
        y,
        *args,
        layers=[128, 128],
        n_components=32,
        activation=jax.nn.silu,
        **kwargs
    ):
        """
        Args:
        y, conditioning variable
        layers, list of hidden layers
        n_components, number of mixture components
        activation, activation function for hidden layers
        """
        self.y = y
        self.layers = layers
        self.n_components = n_components
        self.activation = activation
        super().__init__(*args, **kwargs)

    def __call__(self, x, output_units, **condition_kwargs):
        net = jnp.concatenate([x, self.y], axis=-1)
        for i, layer_size in enumerate(self.layers):
            net = self.activation(hk.Linear(layer_size, name="layer%d" % i)(net))

        log_a_bound = 4
        min_density_lower_bound = 1e-4
        n_components = self.n_components

        log_a = (
            jax.nn.tanh(hk.Linear(output_units * n_components, name="l3")(net))
            * log_a_bound
        )
        b = hk.Linear(output_units * n_components, name="l4")(net)
        c = min_density_lower_bound + jax.nn.sigmoid(
            hk.Linear(output_units * n_components, name="l5")(net)
        ) * (1 - min_density_lower_bound)
        p = hk.Linear(output_units * n_components, name="l6")(net)

        log_a = log_a.reshape(-1, output_units, n_components)
        b = b.reshape(-1, output_units, n_components)
        c = c.reshape(-1, output_units, n_components)
        p = p.reshape(-1, output_units, n_components)
        p = jax.nn.softmax(p)

        return MixtureAffineSigmoidBijector(jnp.exp(log_a), b, c, p)

# Normalizing flow model 
class ConditionalRealNVP(hk.Module):
    """A normalizing flow based on RealNVP using specified bijector functions."""

    def __init__(
        self, d, *args, n_layers=3, bijector_fn=AffineSigmoidCoupling, **kwargs
    ):
        """
        Args:
        d, dimensionality of the input
        n_layers, number of layers
        coupling_layer, list of coupling layers
        """
        self.d = d
        self.n_layer = n_layers
        self.bijector_fn = bijector_fn
        super().__init__(*args, **kwargs)

    def __call__(self, y):
        chain = tfb.Chain(
            [
                tfb.Permute(jnp.arange(self.d)[::-1])(
                    tfb.RealNVP(
                        self.d // 2, bijector_fn=self.bijector_fn(y, name="b%d" % i)
                    )
                )
                for i in range(self.n_layer)
            ]
        )

        nvp = tfd.TransformedDistribution(
            tfd.MultivariateNormalDiag(0.5 * jnp.ones(self.d), 0.05 * jnp.ones(self.d)),
            bijector=chain,
        )

        return nvp

