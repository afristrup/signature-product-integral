extern crate candle_core;
use candle_core::{Device, Error, Shape, Tensor};

// def signature(
//     path: Float[Array, "path_len dim"] | Float[Array, "batch path_len dim"],
//     depth: int,
//     stream: bool = False,
//     flatten: bool = True,
//     num_chunks: int = 1,
// ) -> list[Array] | Array:
//     """
//     Compute the signature of a path. Automatically dispatches to vmap or not based on the shape of `path`.
//
//     Args:
//         path: size (length, dim) or (batch, length, dim)
//         depth: signature is truncated at this depth
//         stream: whether to handle `path` as a stream. Default is False
//         flatten: whether to flatten the output. Default is False
//         num_chunks: number of chunks to use. Default is 1. If > 1, path will be divided into
//         chunks to compute signatures. Then, obtained signatures are combined (using Chen's identity).
//
//     Returns:
//         If `stream` is `True`, this will return a list of `Array` in a form
//             [(path_len - 1, dim), (path_len - 1, dim, dim), (path_len - 1, dim, dim, dim), ...]
//         If `stream` is `False`, this will return a list of `Array` in a form
//             [(dim, ), (dim, dim), (dim, dim, dim), ...]
//         If `flatten` is `True`, this will return a flattened array of shape
//             (dim + dim**2 + ... + dim**depth, )
//         If your path is of shape (batch, path_len, dim), all of the above will have an extra
//         dimension of size `batch` as the first dimension.
//
//     """
//     if num_chunks > 1:
//         sig_fun: Callable[[Array], Array | list[Array]] = partial(
//             _signature_chunked,
//             num_chunks=num_chunks,
//             depth=depth,
//             stream=stream,
//             flatten=flatten,
//         )
//     else:
//         sig_fun = partial(_signature, depth=depth, stream=stream, flatten=flatten)
//     # this is just to handle shape errors
//     if path.ndim == 2:
//         return sig_fun(path)  # regular case
//     if path.ndim == 3:  # batch case (mimics signatory)
//         return jax.vmap(sig_fun)(path)
//     msg = f"Path must be of shape (path_length, path_dim) or (batch, path_length, path_dim), got {path.shape}"
//     raise ValueError(msg)

// @partial(jax.jit, static_argnames=["depth", "stream", "flatten"])
// def _signature(
//     path: Float[Array, "path_len dim"],
//     depth: int,
//     stream: bool = False,
//     flatten: bool = False,
// ) -> list[Array] | Array:
//     """
//     Compute the signature of a path. Optionally, divide the path into chunks to compute signatures
//     and combine them using Chen's identity (useful for long paths).
//
//     Args:
//         path: size (length, dim)
//         depth: signature is truncated at this depth
//         stream: whether to handle `path` as a stream. Default is False
//         flatten: whether to flatten the output. Default is False
//
//         If `stream` is `True`, this will return a list of `Array` in a form
//             [(path_len - 1, dim), (path_len - 1, dim, dim), (path_len - 1, dim, dim, dim), ...]
//         If `stream` is `False`, this will return a list of `Array` in a form
//             [(dim, ), (dim, dim), (dim, dim, dim), ...]
//     """
//
//     path_increments = jnp.diff(path, axis=0)
//     exp_term = restricted_exp(path_increments[0], depth=depth)
//
//     def _body(carry, path_inc):
//         ret = mult_fused_restricted_exp(path_inc, carry)
//         return ret, ret
//
//     carry, stacked = jax.lax.scan(f=_body, init=exp_term, xs=path_increments[1:])
//     if stream:
//         res = [
//             jnp.concatenate([first[None, ...], rest], axis=0)
//             for first, rest in zip(exp_term, stacked)
//         ]
//         # here `res` has shape [(path_len - 1, dim), (path_len - 1, dim, dim), ...]
//         if flatten:
//             res = jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0])(res)
//             # now `res` has shape (path_len -1, dim + dim * dim + ...)
//     else:
//         res = carry
//         # `res` has shape [(dim,), (dim, dim), ...]
//         if flatten:
//             res = flatten_util.ravel_pytree(res)[0]
//             # `res` has shape (dim + dim * dim + ..., )
//     return res

// fn unchunked_signature(
//
// )
