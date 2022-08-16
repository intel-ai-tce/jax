# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

import jax
from jax import lax
import jax._src.test_util as jtu
from jax.config import config
from jax.experimental import checkify
from jax.experimental import pjit
from jax.experimental import maps
from jax.experimental.sharding import MeshPspecSharding
from jax.experimental import array
from jax._src.checkify import CheckEffect
import jax.numpy as jnp

config.parse_flags_with_absl()


class CheckifyTransformTests(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_jit={jit}", "jit": jit}
      for jit in [False, True]))
  @jtu.skip_on_devices("tpu")
  def test_jit_nan(self, jit):
    def f(x1, x2):
      y1 = jnp.sin(x1)
      y2 = jnp.sin(x2)
      return y1 + y2

    f = jax.jit(f) if jit else f
    checked_f = checkify.checkify(f, errors=checkify.float_checks)

    err, _ = checked_f(3., 4.)
    self.assertIs(err.get(), None)

    err, _ = checked_f(3., jnp.inf)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "nan generated by primitive sin")

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_jit={jit}", "jit": jit}
      for jit in [False, True]))
  def test_jit_oob(self, jit):
    def f(x, i):
      y = jnp.sin(x)
      z = y[i]
      w = jnp.cos(z)
      return w

    f = jax.jit(f) if jit else f
    checked_f = checkify.checkify(f, errors=checkify.index_checks)

    err, _ = checked_f(jnp.arange(3), 2)
    self.assertIs(err.get(), None)

    err, _ = checked_f(jnp.arange(3), 5)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "out-of-bounds indexing")

  @parameterized.named_parameters(
      {"testcase_name": f"_updatefn={update_fn}", "update_fn": update_fn}
      for update_fn in ["set", "add", "multiply", "divide", "power", "min",
                        "max", "get"])
  def test_jit_oob_update(self, update_fn):
    def f(x, i):
      return getattr(x.at[i], update_fn)(1)

    f = jax.jit(f)
    checked_f = checkify.checkify(f, errors=checkify.index_checks)

    err, _ = checked_f(jnp.arange(3), 2)
    self.assertIs(err.get(), None)

    err, _ = checked_f(jnp.arange(3), 3)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "out-of-bounds indexing")

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_jit={jit}", "jit": jit}
      for jit in [False, True]))
  def test_jit_div_errors(self, jit):
    def f(x, y):
      return x / y

    f = jax.jit(f) if jit else f
    checked_f = checkify.checkify(f, errors=checkify.float_checks)

    err, _ = checked_f(jnp.ones((3,)), jnp.ones((3,)))
    self.assertIs(err.get(), None)

    err, _ = checked_f(jnp.ones((3,)), jnp.array([1., 0., 1.]))
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "divided by zero")

    err, _ = checked_f(jnp.array([1, jnp.inf, 1]), jnp.array([1, jnp.inf, 1]))
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "nan generated by primitive div")

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_jit={jit}", "jit": jit}
      for jit in [False, True]))
  @jtu.skip_on_devices("tpu")
  def test_jit_multi(self, jit):
    def f(x, i):
      y = x[i]
      z = jnp.cos(y)
      return z

    f = jax.jit(f) if jit else f
    checked_f = checkify.checkify(f, errors=checkify.automatic_checks)

    # no error
    err, _ = checked_f(jnp.array([0., jnp.inf, 2.]), 2)
    self.assertIs(err.get(), None)

    # oob error
    err, _ = checked_f(jnp.array([0., 1., 2.]), 5)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "out-of-bounds indexing")

    # nan error
    err, _ = checked_f(jnp.array([0., 1., jnp.inf]), 2)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "nan generated by primitive cos")

  def test_numpy_indexing_oobs(self):
    def raises_oob(fn, idx, *expected_strs):
      err, _ = checkify.checkify(fn, errors=checkify.index_checks)(x, idx)
      error_txt = err.get()
      self.assertIsNotNone(error_txt)
      self.assertStartsWith(error_txt, "out-of-bounds indexing")
      for s in expected_strs:
        self.assertIn(s, error_txt)

    x = jnp.ones((2, 3, 7))
    axis0_msg = "axis 0 with size 2"
    axis1_msg = "axis 1 with size 3"
    axis2_msg = "axis 2 with size 7"

    single_idx = lambda x, i: x[i]
    raises_oob(single_idx, 5, "index 5", axis0_msg)
    raises_oob(single_idx, -5, "index -3", axis0_msg)
    raises_oob(single_idx, (0, 100), "index 100", axis1_msg)
    raises_oob(single_idx, (0, 5, 100), "index 5", axis1_msg)
    raises_oob(single_idx, (0, 0, 100), "index 100", axis2_msg)
    raises_oob(single_idx, ((1, 20), (1, 4)), "index 20", axis0_msg)
    raises_oob(single_idx, ((1, 20), (3, 4)), "index 3", axis1_msg)
    raises_oob(single_idx, (((1, 1), (1, 20)), 3), "index 3", axis1_msg)
    raises_oob(single_idx, (((1, 1), (1, 20)), 0), "index 20", axis0_msg)

    multi_idx = lambda x, i: x[i[0], :, i[1]]
    raises_oob(multi_idx, (0, 9), "index 9", axis2_msg)
    # TODO(lenamartens): numpy reports index -5 here, need to normalize?
    raises_oob(multi_idx, (-5, 9), "index -3", axis0_msg)
    raises_oob(multi_idx, (5, -9), "index 5", axis0_msg)
    raises_oob(multi_idx, ((0, 9), 0), "index 9", axis0_msg)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": f"_jit={jit}", "jit": jit}
      for jit in [False, True]))
  def test_jit_ordering(self, jit):
    def f(x, i):
      y = x[i]
      z = jnp.sin(x)
      return y * z

    f = jax.jit(f) if jit else f
    checked_f = checkify.checkify(f, errors=checkify.automatic_checks)

    # both oob and nan error, but oob happens first
    err, _ = checked_f(jnp.array([0., 1., jnp.inf]), 5)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "out-of-bounds indexing")

  @jtu.skip_on_devices("tpu")
  def test_pmap_basic(self):
    if len(jax.devices()) < 2:
      raise unittest.SkipTest("requires at least 2 devices")

    @jax.pmap
    def f(x1, x2):
      y1 = jnp.sin(x1)
      y2 = jnp.sin(x2)
      return y1 + y2
    checked_f = checkify.checkify(f, errors=checkify.float_checks)

    xs = jnp.array([0., 2.])
    err, _ = checked_f(xs, xs)
    self.assertIs(err.get(), None)

    ys = jnp.array([3., jnp.inf])
    err, _ = checked_f(xs, ys)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "nan generated by primitive sin")

  @jtu.skip_on_devices("tpu")
  def test_cond_basic(self):
    @jax.jit
    def f(x):
      return lax.cond(x > 0,
                      lambda: jnp.sin(x),
                      lambda: x)

    checked_f = checkify.checkify(f, errors=checkify.float_checks)

    err, _ = checked_f(3.)
    self.assertIs(err.get(), None)

    err, _ = checked_f(jnp.inf)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "nan generated by primitive sin")

    err, _ = checked_f(-jnp.inf)
    self.assertIs(err.get(), None)

  @jtu.skip_on_devices("tpu")
  def test_scan_map(self):
    def scan_body(_, x):
      return None, jnp.sin(x)

    @jax.jit
    def f(xs):
      return lax.scan(scan_body, None, xs)

    checked_f = checkify.checkify(f, errors=checkify.float_checks)

    xs = jnp.array([0., 2.])
    err, (_, ch_outs) = checked_f(xs)
    _, outs = f(xs)
    self.assertIs(err.get(), None)
    self.assertArraysEqual(ch_outs, outs)

    xs = jnp.array([3., jnp.inf])
    err, (_, ch_outs) = checked_f(xs)
    _, outs = f(xs)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "nan generated by primitive sin")
    self.assertArraysEqual(ch_outs, outs)

  @jtu.skip_on_devices("tpu")
  def test_scan_carry(self):
    def scan_body(carry, x):
      carry = carry-1.
      possible_nan = jnp.sin(1./carry)
      return carry, x+possible_nan

    @jax.jit
    def f(carry, xs):
      return lax.scan(scan_body, carry, xs)

    checked_f = checkify.checkify(f, errors=checkify.float_checks)

    carry, xs = 3., jnp.ones((2,))
    err, (ch_out_carry, ch_outs) = checked_f(carry, xs)
    out_carry, outs = f(carry, xs)
    self.assertIs(err.get(), None)
    self.assertArraysEqual(ch_outs, outs)
    self.assertArraysEqual(ch_out_carry, out_carry)

    # error happens on first iteration
    carry, xs = 1., jnp.ones((2,))
    err, (ch_out_carry, ch_outs) = checked_f(carry, xs)
    out_carry, outs = f(carry, xs)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "divided by zero")
    self.assertArraysEqual(ch_outs, outs)
    self.assertArraysEqual(ch_out_carry, out_carry)

    # error happens on second iteration
    carry, xs = 2., jnp.ones((4,))
    err, (ch_out_carry, ch_outs) = checked_f(carry, xs)
    out_carry, outs = f(carry, xs)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "divided by zero")
    self.assertArraysEqual(ch_outs, outs)
    self.assertArraysEqual(ch_out_carry, out_carry)

  @jtu.skip_on_devices("tpu")
  def test_while_loop_body_error(self):
    def while_cond(val):
      i, _ = val
      return i < 2

    def while_body(val):
      i, x = val
      possible_nan = jnp.sin(1./i)
      return i+1., x+possible_nan

    @jax.jit
    def f(init_val):
      return lax.while_loop(while_cond, while_body, (init_val, 0.))

    checked_f = checkify.checkify(f, errors=checkify.float_checks)

    init_val = 1.
    err, ch_out = checked_f(init_val)
    out = f(init_val)
    self.assertIs(err.get(), None)
    self.assertArraysEqual(ch_out, out)

    init_val = 0.
    err, ch_out = checked_f(init_val)
    out = f(init_val)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "divided by zero")
    self.assertArraysEqual(ch_out, out)

  @jtu.skip_on_devices("tpu")
  def test_while_loop_cond_error(self):
    def while_cond(val):
      _ = jnp.sin(1./val)
      return val < 2.

    def while_body(val):
      return val+1.

    @jax.jit
    def f(init_val):
      return lax.while_loop(while_cond, while_body, init_val)

    checked_f = checkify.checkify(f, errors=checkify.float_checks)

    init_val = 1.
    err, ch_out = checked_f(init_val)
    out = f(init_val)
    self.assertIs(err.get(), None)
    self.assertArraysEqual(ch_out, out)

    init_val = 0.
    err, ch_out = checked_f(init_val)
    out = f(init_val)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "divided by zero")
    self.assertArraysEqual(ch_out, out)

  @jtu.skip_on_devices("tpu")
  def test_while_loop_cond_error_and_false(self):
    # Tests if an error is generated when cond returns False.
    def while_cond(val):
      possible_nan = jnp.sin(1./val)
      return jnp.logical_not(jnp.isnan(possible_nan))

    @jax.jit
    def f(init_val):
      return lax.while_loop(while_cond, lambda val: val-1, init_val)

    checked_f = checkify.checkify(f, errors=checkify.float_checks)

    # error on first cond
    init_val = 0.
    err, _ = checked_f(init_val)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "divided by zero")

    # error on second cond
    init_val = 1.
    err, _ = checked_f(init_val)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "divided by zero")

  @jtu.skip_on_devices("tpu")
  def test_while_loop_body_and_cond_error(self):
    def while_cond(val):
      i, cond_val, _ = val
      _ = jnp.sin(cond_val)
      return i < 2

    def while_body(val):
      i, cond_val, body_val = val
      possible_nan = jnp.cos(body_val)
      return i+1., cond_val, possible_nan

    @jax.jit
    def f(cond_val, body_val):
      return lax.while_loop(while_cond, while_body, (0., cond_val, body_val))

    checked_f = checkify.checkify(f, errors=checkify.float_checks)

    cond_val = jnp.inf
    body_val = 1.
    err, _ = checked_f(cond_val, body_val)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "nan generated by primitive sin")

    cond_val = 1.
    body_val = jnp.inf
    err, _ = checked_f(cond_val, body_val)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "nan generated by primitive cos")

    cond_val = jnp.inf
    body_val = jnp.inf
    err, _ = checked_f(cond_val, body_val)
    self.assertIsNotNone(err.get())
    # first error which occurs is in cond
    self.assertStartsWith(err.get(), "nan generated by primitive sin")

  def test_pjit(self):
    def f(x):
      # unary func
      return x / x

    def g(x, y):
      # binary func
      return x / y

    mesh = maps.Mesh(np.array(jax.devices()), ["dev"])
    if config.jax_array:
      ps = MeshPspecSharding(mesh, pjit.PartitionSpec("dev"))
      inp = np.arange(8)
      x = array.make_array_from_callback(inp.shape, ps, lambda idx: inp[idx])
    else:
      ps = pjit.PartitionSpec("dev")
      x = jnp.arange(8)

    f = pjit.pjit(f, in_axis_resources=ps, out_axis_resources=ps)
    f = checkify.checkify(f, errors=checkify.float_checks)
    g = pjit.pjit(g, in_axis_resources=ps, out_axis_resources=ps)
    g = checkify.checkify(g, errors=checkify.float_checks)
    with mesh:
      u_err, _ = f(x)
      b_err, _ = g(x, x)

    self.assertIsNotNone(u_err.get())
    self.assertStartsWith(u_err.get(), "divided by zero")
    self.assertIsNotNone(b_err.get())
    self.assertStartsWith(b_err.get(), "divided by zero")

  def test_empty_enabled_errors(self):
    def multi_errors(x):
      x = x/0         # DIV
      x = jnp.sin(x)  # NAN
      x = x[500]      # OOB
      checkify.check(x < 0, "must be negative!")  # ASSERT
      return x

    x = jnp.ones((2,))
    err, _ = checkify.checkify(multi_errors, errors=set())(x)
    self.assertIsNone(err.get())

  @parameterized.named_parameters(
      ("assert", checkify.user_checks, "must be negative!"),
      ("div", {checkify.ErrorCategory.DIV}, "divided by zero"),
      ("nan", {checkify.ErrorCategory.NAN}, "nan generated"),
      ("oob", checkify.index_checks, "out-of-bounds indexing"),
      ("automatic_checks", checkify.automatic_checks, "divided by zero"),
    )
  @jtu.skip_on_devices("tpu")
  def test_enabled_errors(self, error_set, expected_error):
    def multi_errors(x):
      checkify.check(jnp.all(x < 0), "must be negative!")  # ASSERT
      x = x/0         # DIV
      x = jnp.sin(x)  # NAN
      x = x[500]      # OOB
      return x

    x = jnp.ones((2,))
    err, _ = checkify.checkify(multi_errors, errors=error_set)(x)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), expected_error)

  @jtu.skip_on_devices("tpu")
  def test_post_process_call(self):
    @partial(checkify.checkify, errors=checkify.float_checks)
    def g(x):
      @jax.jit
      def f(y):
        return jnp.sin(x * y)
      return f(jnp.inf)
    err, _ = g(2.)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "nan generated by primitive sin")

  @jtu.skip_on_devices("tpu")
  def test_post_process_map(self):
    @partial(checkify.checkify, errors=checkify.float_checks)
    def g(x):
      @jax.pmap
      def f(y):
        return jnp.sin(x * y)
      return f(jnp.array([jnp.inf]))[0]
    err, _ = g(2.)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), 'nan generated by primitive sin')

  @jtu.skip_on_devices("tpu")
  def test_custom_jvp(self):
    @jax.custom_jvp
    def sin(x):
      return jnp.sin(x)

    @sin.defjvp
    def sin_jvp(primals, tangents):
      (x,), (xdot,) = primals, tangents
      return sin(x), jnp.cos(x) * xdot

    f = checkify.checkify(sin, errors=checkify.float_checks)

    err, y = f(3.)
    self.assertIsNone(err.get())
    err, y = f(jnp.inf)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), 'nan generated by primitive sin')

    # When we hit the custom jvp rule with jvp-of-checkify, no checks are added.
    (err, y), (errdot, ydot) = jax.jvp(f, (3.,), (1.,))  # doesn't crash
    self.assertIsNone(err.get())  # no error
    self.assertEmpty(err.msgs)    # and no checks were added!
    self.assertEmpty(errdot.msgs)
    y_expected, ydot_expected = jax.jvp(jnp.sin, (3.,), (1.,))
    self.assertAllClose(y, y_expected)
    self.assertAllClose(ydot, ydot_expected)

    # Grad-of-checkify doesn't crash either.
    x_bar = jax.grad(lambda x: f(x)[1])(3.)
    self.assertAllClose(x_bar, jnp.cos(3.))

    # Checkify-of-jvp adds checks (unlike jvp-of-checkify above).
    g = checkify.checkify(lambda x, xdot: jax.jvp(sin, (x,), (xdot,)),
                          errors=checkify.float_checks)
    err, (y, ydot) = g(3., 1.)  # doesn't crash
    self.assertIsNone(err.get())  # no error
    self.assertNotEmpty(err.msgs) # but checks were added!
    self.assertAllClose(y, jnp.sin(3.))
    self.assertAllClose(ydot, jnp.cos(3.))
    err, _ = g(jnp.inf, 1.)
    self.assertIsNotNone(err.get())  # yes error
    self.assertStartsWith(err.get(), 'nan generated by primitive sin')

  @jtu.skip_on_devices("tpu")
  def test_custom_vjp(self):
    @jax.custom_vjp
    def sin(x):
      return jnp.sin(x)

    def sin_fwd(x):
      return jnp.sin(x), 2. * x
    def sin_bwd(x2, g):
      return jnp.cos(x2 / 2.) * g,
    sin.defvjp(sin_fwd, sin_bwd)

    f = checkify.checkify(sin, errors=checkify.float_checks)

    # no differentiation, no error
    err, y = f(3.)
    self.assertIsNone(err.get())

    # no differentiation, yes error
    err, y = f(jnp.inf)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), 'nan generated by primitive sin')

    # When we hit the custom vjp rule with vjp-of-checkify, no checks are added.
    (err, y), f_vjp = jax.vjp(f, 3.)
    self.assertIsNone(err.get())  # no error
    self.assertEmpty(err.msgs)    # and no checks were added!

    # Checkify-of-vjp adds checks (unlike vjp-of-checkify above).
    err, y = checkify.checkify(jax.grad(sin), errors=checkify.float_checks)(3.)
    self.assertIsNone(err.get())   # no error
    self.assertNotEmpty(err.msgs)  # but checks were added!
    err, y = checkify.checkify(jax.grad(sin),
                               errors=checkify.float_checks)(jnp.inf)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "nan generated by primitive sin")

  def test_scan_consts(self):
    def f(xs):
      def scan_body(carry, _):
        # closes oves xs
        return carry+1, xs[carry]
      return lax.scan(scan_body, 1, xs)[1]

    checked_f = checkify.checkify(f, errors=checkify.index_checks)
    err, _ = checked_f(jnp.ones((7, 3)))
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "out-of-bounds indexing")

  def test_scan_consts2(self):
    def f(xs):
      def scan_body(carry, _):
        # add more consts!
        _ = xs[carry], xs[carry], jnp.sin(np.arange(11.))
        return carry+1, xs[carry]
      return lax.scan(scan_body, 1, xs)[1]

    checked_f = checkify.checkify(f, errors=checkify.index_checks)
    err, _ = checked_f(jnp.ones((7, 3)))
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "out-of-bounds indexing")

  def test_while_consts(self):
    def f(xs):
      def while_cond(carry):
        i, _ = carry
        _ = xs[i], jnp.sin(np.arange(11.))
        return i > -1

      def while_body(carry):
        i, _ = carry
        x = xs[i]
        return i - 1, x/i

      return lax.while_loop(while_cond, while_body, (0, jnp.zeros_like(xs[0])))

    checked_f = checkify.checkify(f, errors=checkify.float_checks)
    err, _ = checked_f(jnp.ones((7, 3)))
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "divided by zero")

  def test_multiple_payloads(self):
    def f(x):
      _ = x[5]
      _ = x[6]

    err, _ = checkify.checkify(f, errors=checkify.index_checks)(jnp.ones((2,)))
    self.assertIsNotNone(err.get())
    self.assertIn("index 5", err.get())

  def test_nd_payloads(self):
    cf = checkify.checkify(lambda x, i: x[i], errors=checkify.index_checks)
    errs, _ = jax.vmap(cf)(jnp.ones((3, 2)), jnp.array([5, 0, 100]))
    self.assertIsNotNone(errs.get())
    self.assertIn("index 5", errs.get())
    self.assertIn("index 100", errs.get())

  def test_mapped_error_one_payload(self):
    def f(x, i):
      x = x[i]
      return x/0

    cf = checkify.checkify(f, errors=checkify.automatic_checks)
    errs, _ = jax.vmap(cf)(jnp.ones((2, 1)), jnp.array([0, 100]))
    self.assertIsNotNone(errs.get())
    self.assertIn("divided by zero", errs.get())
    self.assertIn("index 100", errs.get())


class AssertPrimitiveTests(jtu.JaxTestCase):

  def test_assert_primitive_impl(self):
    def f():
      checkify.check(False, "hi")

    with self.assertRaisesRegex(ValueError, "hi"):
      f()

  def test_assert_primitive_lowering(self):
    @jax.jit
    def f():
      checkify.check(False, "hi")

    with self.assertRaisesRegex(ValueError, "Cannot abstractly evaluate"):
      f()

  def test_assert_primitive_jaxpr_effects(self):
    def f():
      checkify.check(False, "hi")

    self.assertSetEqual(jax.make_jaxpr(f)().effects, {CheckEffect})

  def test_assert_primitive_eval_shape(self):
    # The check is abstractly evaluated but not lowered.
    def f():
      checkify.check(False, "hi")

    jax.eval_shape(f)  # does not crash.

  def test_assert_discharging(self):
    @checkify.checkify
    def f(x):
      checkify.check(x > 0, "must be positive!")
      return jnp.log(x)

    err, _ = f(1.)
    self.assertIsNone(err.get())

    err, _ = f(0.)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "must be positive")

    f = jax.jit(f)

    err, _ = f(1.)
    self.assertIsNone(err.get())

    err, _ = f(0.)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "must be positive")

  def test_assert_discharging_no_data_dependence(self):
    @jax.jit
    def g(x):
      @checkify.checkify
      def f():
        # Note that x is not an argument to the checkified function.
        checkify.check(x > 0, "must be positive!")
        return jnp.log(x)
      return f()

    err, _ = g(1.)
    self.assertIsNone(err.get())

    err, _ = g(0.)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "must be positive")

  def test_assert_discharging_scan(self):
    def body(carry, x):
      checkify.check(jnp.all(x > 0), "must be positive")
      return carry, x

    def f(x):
      return jax.lax.scan(body, (None,), x)

    err, _ = checkify.checkify(f)(jnp.array([-1]))
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "must be positive")

    err, _ = checkify.checkify(f)(jnp.array([1, 0, -1]))
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "must be positive")

  def test_assert_discharging_while_loop(self):
    def while_cond(val):
      i, _ = val
      checkify.check(i < 0, "i must be negative")
      return i < 2

    def while_body(val):
      i, x = val
      checkify.check(x < 0, "x must be negative")
      return i+1., x+1

    @jax.jit
    def f(init_i, init_val):
      return lax.while_loop(while_cond, while_body, (init_i, init_val))

    checked_f = checkify.checkify(f)

    err, _ = checked_f(0, 1)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "i must be negative")

    err, _ = checked_f(-1, 0)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "x must be negative")

  def test_assert_batching_rule(self):
    @jax.vmap
    def f(x):
      checkify.check(jnp.sum(x) == 1., "x must sum to one.")
      return x

    no_failures = jnp.array([[0.5, 0.5], [1., 0.]])
    one_batch_fails = jnp.array([[0.5, 0.5], [1, 1]])
    mult_batch_fail = jnp.array([[0.5, 0.5], [1, 1], [2, 2]])

    f(no_failures)
    with self.assertRaisesRegex(ValueError, "x must sum to one."):
      f(one_batch_fails)

    with self.assertRaisesRegex(ValueError, "x must sum to one."):
      f(mult_batch_fail)

    checked_f = checkify.checkify(f)
    err, _ = checked_f(no_failures)
    self.assertIsNone(err.get())

    err, _ = checked_f(one_batch_fails)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "x must sum to one")

    err, _ = checked_f(mult_batch_fail)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "x must sum to one")

  def test_check_error(self):
    def f():
      checkify.check_error(checkify.Error(True, 0, {0: "hi"}))

    with self.assertRaisesRegex(ValueError, "hi"):
      f()

    f = checkify.checkify(f)
    err, none = f()

    self.assertIsNone(none)
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "hi")

  def test_check_error_scanned(self):
    def body(carry, x):
      checkify.check(jnp.all(x > 0), "should be positive")
      return carry, x

    def checked_body(carry, x):
      err, (carry, x) = checkify.checkify(body)(carry, x)
      return carry, (x, err)

    def f(x):
      _, (xs, errs) = jax.lax.scan(checked_body, (None,), x)
      checkify.check_error(errs)
      return xs

    err, _ = checkify.checkify(f)(jnp.array([-1]))
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "should be positive")

    err, _ = checkify.checkify(f)(jnp.array([1, 0, -1]))
    self.assertIsNotNone(err.get())
    self.assertStartsWith(err.get(), "should be positive")

  def test_discharge_recharge(self):
    def ejit(f):
      f = checkify.checkify(f)
      f = jax.jit(f)
      def jitted_f(*args):
        err, out = f(*args)
        checkify.check_error(err)
        return out
      return jitted_f

    @ejit
    def f(pred):
      assert python_should_be_running
      checkify.check(pred, "foo")

    python_should_be_running = True
    f(True)

    python_should_be_running = False
    f(True)
    with self.assertRaisesRegex(ValueError, "foo"):
      f(False)

  def test_cond_of_named_call(self):
    def g(x):
      branch = jax.named_call(lambda x: x)
      out = jax.lax.cond(True, branch, branch, x)
      return out

    checkify.checkify(g)(0.)  # does not crash


class CheckifyWithArray:

  def setUp(self):
    super().setUp()
    self.array_enabled = config.jax_array
    config.update('jax_array', True)

  def tearDown(self):
    config.update('jax_array', self.array_enabled)
    super().tearDown()


class ArrayCheckifyTransformTests(CheckifyWithArray, CheckifyTransformTests):
  pass

class ArrayAssertPrimitiveTests(CheckifyWithArray, AssertPrimitiveTests):
  pass


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
