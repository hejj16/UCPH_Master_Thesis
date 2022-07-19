import jax
import jax.numpy as jnp
import os
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import torch
import torch.utils.dlpack
import jax
import jax.dlpack
from pyro.distributions import Delta
from helper import load_blosum


# Written by Sergey Ovchinnikov and Sam Petti
# Spring 2021

def nw(unroll=2, batch=True, gap=0, temp=1):
    # rotate matrix for vectorized dynamic-programming
    def rotate(x, lengths, gap, temp):
        def _ini_global(L):
            return gap * jnp.arange(L)

        a, b = x.shape
        real_a, real_b = lengths
        mask = (jnp.arange(a) < real_a)[:, None] * (jnp.arange(b) < real_b)[None, :]

        real_L = lengths
        mask = jnp.pad(mask, [[1, 0], [1, 0]])
        x = jnp.pad(x, [[1, 0], [1, 0]])

        # solution from jake vanderplas (thanks!)
        a, b = x.shape
        ar, br = jnp.arange(a)[::-1, None], jnp.arange(b)[None, :]
        i, j = (br - ar) + (a - 1), (ar + br) // 2
        n, m = (a + b - 1), (a + b) // 2
        zero = jnp.zeros((n, m))
        output = {"x": zero.at[i, j].set(x),
                  "mask": zero.at[i, j].set(mask),
                  "o": (jnp.arange(n) + a % 2) % 2}

        ini_a, ini_b = _ini_global(a), _ini_global(b)
        ini = jnp.zeros((a, b)).at[:, 0].set(ini_a).at[0, :].set(ini_b)
        output["ini"] = zero.at[i, j].set(ini)

        return {"x": output,
                "prev": (jnp.zeros(m), jnp.zeros(m)),
                "idx": (i, j),
                "mask": mask,
                "L": real_L}

    # fill the scoring matrix
    def sco(x, lengths, gap=0.0, temp=1.0):

        def _logsumexp(x, axis=None, mask=None):
            if mask is None:
                return jax.nn.logsumexp(x, axis=axis)
            else:
                return x.max(axis) + jnp.log(jnp.sum(mask * jnp.exp(x - x.max(axis, keepdims=True)), axis=axis))

        def _soft_maximum(x, axis=None, mask=None):
            return temp * _logsumexp(x / temp, axis, mask)

        def _cond(cond, true, false):
            return cond * true + (1 - cond) * false

        def _step(prev, sm):
            h2, h1 = prev  # previous two rows of scoring (hij) mtx

            Align = h2 + sm["x"]
            Turn = _cond(sm["o"], jnp.pad(h1[:-1], [1, 0]), jnp.pad(h1[1:], [0, 1]))
            h0 = [Align, h1 + gap, Turn + gap]
            h0 = jnp.stack(h0)
            h0 = sm["mask"] * _soft_maximum(h0, 0)
            h0 += sm["ini"]
            return (h1, h0), h0

        a, b = x.shape
        sm = rotate(x, lengths=lengths, gap=gap, temp=temp)
        hij = jax.lax.scan(_step, sm["prev"], sm["x"], unroll=unroll)[-1][sm["idx"]]

        return hij[sm["L"][0], sm["L"][1]]

    # traceback to get alignment (aka. get marginals)
    traceback = jax.grad(sco)

    # add batch dimension
    if batch:
        return jax.vmap(lambda x, y: traceback(x, y, gap, temp), (0, 0)), \
               jax.vmap(lambda x, y: sco(x, y, gap, temp), (0, 0))
    else:
        return lambda x, y: traceback(x, y, gap, temp), \
               lambda x, y: sco(x, y, gap, temp)


# A generic mechanism for turning a JAX function into a PyTorch function.
# code by Matthew Johnson
# see: https://gist.github.com/mattjj/e8b51074fed081d765d2f3ff90edf0e9

def j2t(x_jax):
  x_torch = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x_jax))
  return x_torch


def t2j(x_torch):
  x_torch = x_torch.contiguous()  # https://github.com/google/jax/issues/8082
  x_jax = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x_torch))
  return x_jax


def jax2torch(fun):

  class JaxFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
      y_, ctx.fun_vjp = jax.vjp(fun, t2j(x))
      return j2t(y_)

    @staticmethod
    def backward(ctx, grad_y):
      grad_x_, = ctx.fun_vjp(t2j(grad_y))
      return j2t(grad_x_),

  return JaxFun.apply


def snw(similar_tensor, S_lengths, nw_fn):
    lens = jnp.array([jnp.array([similar_tensor.shape[1], i]) for i in S_lengths])
    nw_fn_torch = jax2torch(lambda x: nw_fn(x, lens))
    path_torch = nw_fn_torch(similar_tensor)
    return path_torch




