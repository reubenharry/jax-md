# Copyright 2019 Google LLC
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

"""Code to simulate systems in various statistical ensembles.

  This file contains a number of different methods that can be used to
  simulate systems in a variety of ensembles.

  In general, simulation code follows the same overall structure as optimizers
  in JAX. Simulations are tuples of two functions:

    init_fn:
      Function that initializes the  state of a system. Should take
      positions as an ndarray of shape `[n, output_dimension]`. Returns a state
      which will be a namedtuple.
    apply_fn:
      Function that takes a state and produces a new state after one
      step of optimization.

  One question that we need to think about is whether the simulations should
  also return a function that computes the invariant for that ensemble. This
  can be used for testing purposes, but is not often used otherwise.
"""

from collections import namedtuple

from typing import Any, Callable, TypeVar, Union, Tuple, Dict, Optional

import functools

from jax import grad
from jax import jit
from jax import ops
from jax import random
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map, tree_reduce, tree_flatten, tree_unflatten

from jax_md import quantity
from jax_md import util
from jax_md import space
from jax_md import dataclasses
from jax_md import partition
from jax_md import smap

import jax
import jax.numpy as jnp
import numpy as np
from scipy.fftpack import next_fast_len



static_cast = util.static_cast


# Types


Array = util.Array
f32 = util.f32
f64 = util.f64

Box = space.Box

ShiftFn = space.ShiftFn

T = TypeVar('T')
InitFn = Callable[..., T]
ApplyFn = Callable[[T], T]
Simulator = Tuple[InitFn, ApplyFn]


"""Dispatch By State Code.

JAX MD allows for simulations to be extensible using a dispatch strategy where
functions are dispatched to specific cases based on the type of state provided.
In particular, we make decisions about which function to call based on the type
of the position argument. For those familiar with C / C++, our dispatch code is
essentially function overloading based on the type of the positions.

If you are interested in setting up a simulation using a different type of
system you can do so in a relatively light weight manner by introducing a new
type for storing the state that is compatible with the JAX PyTree system
(we usually choose a dataclass) and then overriding the functions below.

These extensions allow a range of simulations to be run by just changing the
type of the position argument. There are essentially two types of functions to
be overloaded. Functions that compute physical quantities, such as the kinetic
energy, and functions that evolve a state according to the Suzuki-Trotter
decomposition. Specifically, one might want to override the position step,
momentum step for deterministic and stochastic simulations or the
`stochastic_step` for stochastic simulations (e.g Langevin).
"""


class dispatch_by_state:
  """Wrap a function and dispatch based on the type of positions."""
  def __init__(self, fn):
    self._fn = fn
    self._registry = {}

  def __call__(self, state, *args, **kwargs):
    if type(state.position) in self._registry:
      return self._registry[type(state.position)](state, *args, **kwargs)
    return self._fn(state, *args, **kwargs)

  def register(self, oftype):
    def register_fn(fn):
      self._registry[oftype] = fn
    return register_fn


@dispatch_by_state
def canonicalize_mass(state: T) -> T:
  """Reshape mass vector for broadcasting with positions."""
  def canonicalize_fn(mass):
    if isinstance(mass, float):
      return mass
    if mass.ndim == 2 and mass.shape[1] == 1:
      return mass
    elif mass.ndim == 1:
      return jnp.reshape(mass, (mass.shape[0], 1))
    elif mass.ndim == 0:
      return mass
    msg = (
      'Expected mass to be either a floating point number or a one-dimensional'
      'ndarray. Found {}.'.format(mass)
    )
    raise ValueError(msg)
  return state.set(mass=tree_map(canonicalize_fn, state.mass))

@dispatch_by_state
def initialize_momenta(state: T, key: Array, kT: float) -> T:
  """Initialize momenta with the Maxwell-Boltzmann distribution."""
  R, mass = state.position, state.mass

  R, treedef = tree_flatten(R)
  mass, _ = tree_flatten(mass)
  keys = random.split(key, len(R))

  def initialize_fn(k, r, m):
    p = jnp.sqrt(m * kT) * random.normal(k, r.shape, dtype=r.dtype)
    # If simulating more than one particle, center the momentum.
    if r.shape[0] > 1:
      p = p - jnp.mean(p, axis=0, keepdims=True)
    return p

  P = [initialize_fn(k, r, m) for k, r, m in zip(keys, R, mass)]

  return state.set(momentum=tree_unflatten(treedef, P))


@dispatch_by_state
def momentum_step(state: T, dt: float) -> T:
  """Apply a single step of the time evolution operator for momenta."""
  assert hasattr(state, 'momentum')
  new_momentum = tree_map(lambda p, f: p + dt * f,
                          state.momentum,
                          state.force)
  return state.set(momentum=new_momentum)


@dispatch_by_state
def position_step(state: T, shift_fn: Callable, dt: float, **kwargs) -> T:
  """Apply a single step of the time evolution operator for positions."""
  if isinstance(shift_fn, Callable):
    shift_fn = tree_map(lambda r: shift_fn, state.position)
  new_position = tree_map(lambda s_fn, r, p, m: s_fn(r, dt * p / m, **kwargs),
                          shift_fn,
                          state.position,
                          state.momentum,
                          state.mass)
  return state.set(position=new_position)


@dispatch_by_state
def kinetic_energy(state: T) -> Array:
  """Compute the kinetic energy of a state."""
  return quantity.kinetic_energy(momentum=state.momentum, mass=state.mass)


@dispatch_by_state
def temperature(state: T) -> Array:
  """Compute the temperature of a state."""
  return quantity.temperature(momentum=state.momentum, mass=state.mass)


"""Deterministic Simulations

JAX MD includes integrators for deterministic simulations of the NVE, NVT, and
NPT ensembles. For a qualitative description of statistical physics ensembles
see the wikipedia article here:
en.wikipedia.org/wiki/Statistical_ensemble_(mathematical_physics)

Integrators are based direct translation method outlined in the paper,

"A Liouville-operator derived measure-preserving integrator for molecular
dynamics simulations in the isothermal–isobaric ensemble"

M. E. Tuckerman, J. Alejandre, R. López-Rendón, A. L Jochim, and G. J. Martyna
J. Phys. A: Math. Gen. 39 5629 (2006)

As such, we define several primitives that are generically useful in describing
simulations of this type. Namely, the velocity-Verlet integration step that is
used in the NVE and NVT simulations. We also define a general Nose-Hoover chain
primitive that is used to couple components of the system to a chain that
regulates the temperature. These primitives can be combined to construct more
interesting simulations that involve e.g. temperature gradients.
"""


def velocity_verlet(force_fn: Callable[..., Array],
                    shift_fn: ShiftFn,
                    dt: float,
                    state: T,
                    **kwargs) -> T:
  """Apply a single step of velocity Verlet integration to a state."""
  dt = f32(dt)
  dt_2 = f32(dt / 2)

  state = momentum_step(state, dt_2)
  state = position_step(state, shift_fn, dt, **kwargs)
  state = state.set(force=force_fn(state.position, **kwargs))
  state = momentum_step(state, dt_2)

  return state


# Constant Energy Simulations


@dataclasses.dataclass
class NVEState:
  """A struct containing the state of an NVE simulation.

  This tuple stores the state of a simulation that samples from the
  microcanonical ensemble in which the (N)umber of particles, the (V)olume, and
  the (E)nergy of the system are held fixed.

  Attributes:
    position: An ndarray of shape `[n, spatial_dimension]` storing the position
      of particles.
    momentum: An ndarray of shape `[n, spatial_dimension]` storing the momentum
      of particles.
    force: An ndarray of shape `[n, spatial_dimension]` storing the force
      acting on particles from the previous step.
    mass: A float or an ndarray of shape `[n]` containing the masses of the
      particles.
  """
  position: Array
  momentum: Array
  force: Array
  mass: Array

  @property
  def velocity(self) -> Array:
    return self.momentum / self.mass


# pylint: disable=invalid-name
def nve(energy_or_force_fn, shift_fn, dt=1e-3, **sim_kwargs):
  """Simulates a system in the NVE ensemble.

  Samples from the microcanonical ensemble in which the number of particles
  (N), the system volume (V), and the energy (E) are held constant. We use a
  standard velocity Verlet integration scheme.

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`.
      Both `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
  Returns:
    See above.
  """
  force_fn = quantity.canonicalize_force(energy_or_force_fn)

  @jit
  def init_fn(key, R, kT, mass=f32(1.0), **kwargs):
    force = force_fn(R, **kwargs)
    state = NVEState(R, None, force, mass)
    state = canonicalize_mass(state)
    return initialize_momenta(state, key, kT)

  @jit
  def step_fn(state, **kwargs):
    _dt = kwargs.pop('dt', dt)
    return velocity_verlet(force_fn, shift_fn, _dt, state, **kwargs)

  return init_fn, step_fn


# Constant Temperature Simulations


# Suzuki-Yoshida weights for integrators of different order.
# These are copied from OpenMM at
# https://github.com/openmm/openmm/blob/master/openmmapi/src/NoseHooverChain.cpp


SUZUKI_YOSHIDA_WEIGHTS = {
    1: [1],
    3: [0.828981543588751, -0.657963087177502, 0.828981543588751],
    5: [0.2967324292201065, 0.2967324292201065, -0.186929716880426,
        0.2967324292201065, 0.2967324292201065],
    7: [0.784513610477560, 0.235573213359357, -1.17767998417887,
        1.31518632068391, -1.17767998417887, 0.235573213359357,
        0.784513610477560]
}


@dataclasses.dataclass
class NoseHooverChain:
  """State information for a Nose-Hoover chain.

  Attributes:
    position: An ndarray of shape `[chain_length]` that stores the position of
      the chain.
    momentum: An ndarray of shape `[chain_length]` that stores the momentum of
      the chain.
    mass: An ndarray of shape `[chain_length]` that stores the mass of the
      chain.
    tau: The desired period of oscillation for the chain. Longer periods result
      is better stability but worse temperature control.
    kinetic_energy: A float that stores the current kinetic energy of the
      system that the chain is coupled to.
    degrees_of_freedom: An integer specifying the number of degrees of freedom
      that the chain is coupled to.
  """
  position: Array
  momentum: Array
  mass: Array
  tau: Array
  kinetic_energy: Array
  degrees_of_freedom: int=dataclasses.static_field()


@dataclasses.dataclass
class NoseHooverChainFns:
  initialize: Callable
  half_step: Callable
  update_mass: Callable


def nose_hoover_chain(dt: float,
                      chain_length: int,
                      chain_steps: int,
                      sy_steps: int,
                      tau: float
                      ) -> NoseHooverChainFns:
  """Helper function to simulate a Nose-Hoover Chain coupled to a system.

  This function is used in simulations that sample from thermal ensembles by
  coupling the system to one, or more, Nose-Hoover chains. We use the direct
  translation method outlined in Martyna et al. [#martyna92]_ and the
  Nose-Hoover chains are updated using two half steps: one at the beginning of
  a simulation step and one at the end. The masses of the Nose-Hoover chains
  are updated automatically to enforce a specific period of oscillation, `tau`.
  Larger values of `tau` will yield systems that reach the target temperature
  more slowly but are also more stable.

  As described in Martyna et al. [#martyna92]_, the Nose-Hoover chain often
  evolves on a faster timescale than the rest of the simulation. Therefore, it
  sometimes necessary
  to integrate the chain over several substeps for each step of MD. To do this
  we follow the Suzuki-Yoshida scheme. Specifically, we subdivide our chain
  simulation into :math:`n_c` substeps. These substeps are further subdivided
  into :math:`n_sy` steps. Each :math:`n_sy` step has length
  :math:`\delta_i = \Delta t w_i / n_c` where :math:`w_i` are constants such
  that :math:`\sum_i w_i = 1`. See the table of Suzuki-Yoshida weights above
  for specific values. The number of substeps and the number of Suzuki-Yoshida
  steps are set using the `chain_steps` and `sy_steps` arguments.

  Consequently, the Nose-Hoover chains are described by three functions: an
  `init_fn` that initializes the state of the chain, a `half_step_fn` that
  updates the chain for one half-step, and an `update_chain_mass_fn` that
  updates the masses of the chain to enforce the correct period of oscillation.

  Note that a system can have many Nose-Hoover chains coupled to it to produce,
  for example, a temperature gradient. We also note that the NPT ensemble
  naturally features two chains: one that couples to the thermal degrees of
  freedom and one that couples to the barostat.

  Attributes:
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    chain_length: An integer specifying the number of particles in
      the Nose-Hoover chain.
    chain_steps: An integer specifying the number :math:`n_c` of outer substeps.
    sy_steps: An integer specifying the number of Suzuki-Yoshida steps. This
      must be either `1`, `3`, `5`, or `7`.
    tau: A floating point timescale over which temperature equilibration occurs.
      Measured in units of `dt`. The performance of the Nose-Hoover chain
      thermostat can be quite sensitive to this choice.
  Returns:
    A triple of functions that initialize the chain, do a half step of
    simulation, and update the chain masses respectively.
  """

  def init_fn(degrees_of_freedom, KE, kT):
    xi = jnp.zeros(chain_length, KE.dtype)
    p_xi = jnp.zeros(chain_length, KE.dtype)

    Q = kT * tau ** f32(2) * jnp.ones(chain_length, dtype=f32)
    Q = Q.at[0].multiply(degrees_of_freedom)
    return NoseHooverChain(xi, p_xi, Q, tau, KE, degrees_of_freedom)

  def substep_fn(delta, P, state, kT):
    """Apply a single update to the chain parameters and rescales velocity."""
    xi, p_xi, Q, _tau, KE, DOF = dataclasses.astuple(state)

    delta_2 = delta   / f32(2.0)
    delta_4 = delta_2 / f32(2.0)
    delta_8 = delta_4 / f32(2.0)

    M = chain_length - 1

    G = (p_xi[M - 1] ** f32(2) / Q[M - 1] - kT)
    p_xi = p_xi.at[M].add(delta_4 * G)

    def backward_loop_fn(p_xi_new, m):
      G = p_xi[m - 1] ** 2 / Q[m - 1] - kT
      scale = jnp.exp(-delta_8 * p_xi_new / Q[m + 1])
      p_xi_new = scale * (scale * p_xi[m] + delta_4 * G)
      return p_xi_new, p_xi_new
    idx = jnp.arange(M - 1, 0, -1)
    _, p_xi_update = lax.scan(backward_loop_fn, p_xi[M], idx, unroll=2)
    p_xi = p_xi.at[idx].set(p_xi_update)

    G = f32(2.0) * KE - DOF * kT
    scale = jnp.exp(-delta_8 * p_xi[1] / Q[1])
    p_xi = p_xi.at[0].set(scale * (scale * p_xi[0] + delta_4 * G))

    scale = jnp.exp(-delta_2 * p_xi[0] / Q[0])
    KE = KE * scale ** f32(2)
    P = tree_map(lambda p: p * scale, P)

    xi = xi + delta_2 * p_xi / Q

    G = f32(2) * KE - DOF * kT
    def forward_loop_fn(G, m):
      scale = jnp.exp(-delta_8 * p_xi[m + 1] / Q[m + 1])
      p_xi_update = scale * (scale * p_xi[m] + delta_4 * G)
      G = p_xi_update ** 2 / Q[m] - kT
      return G, p_xi_update
    idx = jnp.arange(M)
    G, p_xi_update = lax.scan(forward_loop_fn, G, idx, unroll=2)
    p_xi = p_xi.at[idx].set(p_xi_update)
    p_xi = p_xi.at[M].add(delta_4 * G)

    return P, NoseHooverChain(xi, p_xi, Q, _tau, KE, DOF), kT

  def half_step_chain_fn(P, state, kT):
    if chain_steps == 1 and sy_steps == 1:
      P, state, _ = substep_fn(dt, P, state, kT)
      return P, state

    delta = dt / chain_steps
    ws = jnp.array(SUZUKI_YOSHIDA_WEIGHTS[sy_steps])
    def body_fn(cs, i):
      d = f32(delta * ws[i % sy_steps])
      return substep_fn(d, *cs), 0
    P, state, _ = lax.scan(body_fn,
                           (P, state, kT),
                           jnp.arange(chain_steps * sy_steps))[0]
    return P, state

  def update_chain_mass_fn(state, kT):
    xi, p_xi, Q, _tau, KE, DOF = dataclasses.astuple(state)

    Q = kT * _tau ** f32(2) * jnp.ones(chain_length, dtype=f32)
    Q = Q.at[0].multiply(DOF)

    return NoseHooverChain(xi, p_xi, Q, _tau, KE, DOF)

  return NoseHooverChainFns(init_fn, half_step_chain_fn, update_chain_mass_fn)


def default_nhc_kwargs(tau: float, overrides: Dict) -> Dict:
  default_kwargs = {
      'chain_length': 3,
      'chain_steps': 2,
      'sy_steps': 3,
      'tau': tau
  }

  if overrides is None:
    return default_kwargs

  return {
      key: overrides.get(key, default_kwargs[key])
      for key in default_kwargs
  }


@dataclasses.dataclass
class NVTNoseHooverState:
  """State information for an NVT system with a Nose-Hoover chain thermostat.

  Attributes:
    position: The current position of particles. An ndarray of floats
      with shape `[n, spatial_dimension]`.
    momentum: The momentum of particles. An ndarray of floats
      with shape `[n, spatial_dimension]`.
    force: The current force on the particles. An ndarray of floats with shape
      `[n, spatial_dimension]`.
    mass: The mass of the particles. Can either be a float or an ndarray
      of floats with shape `[n]`.
    chain: The variables describing the Nose-Hoover chain.
  """
  position: Array
  momentum: Array
  force: Array
  mass: Array
  chain: NoseHooverChain

  @property
  def velocity(self):
    return self.momentum / self.mass


def nvt_nose_hoover(energy_or_force_fn: Callable[..., Array],
                    shift_fn: ShiftFn,
                    dt: float,
                    kT: float,
                    chain_length: int=5,
                    chain_steps: int=2,
                    sy_steps: int=3,
                    tau: Optional[float]=None,
                    **sim_kwargs) -> Simulator:
  """Simulation in the NVT ensemble using a Nose Hoover Chain thermostat.

  Samples from the canonical ensemble in which the number of particles (N),
  the system volume (V), and the temperature (T) are held constant. We use a
  Nose Hoover Chain (NHC) thermostat described in [#martyna92]_ [#martyna98]_
  [#tuckerman]_. We follow the direct translation method outlined in
  Tuckerman et al. [#tuckerman]_ and the interested reader might want to look
  at that paper as a reference.

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`.
      Both `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    kT: Floating point number specifying the temperature in units of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
    chain_length: An integer specifying the number of particles in
      the Nose-Hoover chain.
    chain_steps: An integer specifying the number, :math:`n_c`, of outer
      substeps.
    sy_steps: An integer specifying the number of Suzuki-Yoshida steps. This
      must be either `1`, `3`, `5`, or `7`.
    tau: A floating point timescale over which temperature equilibration
      occurs. Measured in units of `dt`. The performance of the Nose-Hoover
      chain thermostat can be quite sensitive to this choice.
  Returns:
    See above.

  .. rubric:: References
  .. [#martyna92] Martyna, Glenn J., Michael L. Klein, and Mark Tuckerman.
    "Nose-Hoover chains: The canonical ensemble via continuous dynamics."
    The Journal of chemical physics 97, no. 4 (1992): 2635-2643.
  .. [#martyna98] Martyna, Glenn, Mark Tuckerman, Douglas J. Tobias, and Michael L. Klein.
    "Explicit reversible integrators for extended systems dynamics."
    Molecular Physics 87. (1998) 1117-1157.
  .. [#tuckerman] Tuckerman, Mark E., Jose Alejandre, Roberto Lopez-Rendon,
    Andrea L. Jochim, and Glenn J. Martyna.
    "A Liouville-operator derived measure-preserving integrator for molecular
    dynamics simulations in the isothermal-isobaric ensemble."
    Journal of Physics A: Mathematical and General 39, no. 19 (2006): 5629.
  """
  force_fn = quantity.canonicalize_force(energy_or_force_fn)
  print("FOO")
  dt = f32(dt)
  dt_2 = f32(dt / 2)
  if tau is None:
    tau = dt * 100
  tau = f32(tau)

  thermostat = nose_hoover_chain(dt, chain_length, chain_steps, sy_steps, tau)

  @jit
  def init_fn(key, R, mass=f32(1.0), **kwargs):
    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    dof = quantity.count_dof(R)

    state = NVTNoseHooverState(R, None, force_fn(R, **kwargs), mass, None)
    state = canonicalize_mass(state)
    state = initialize_momenta(state, key, _kT)
    KE = kinetic_energy(state)
    return state.set(chain=thermostat.initialize(dof, KE, _kT))

  @jit
  def apply_fn(state, **kwargs):
    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    chain = state.chain

    chain = thermostat.update_mass(chain, _kT)

    p, chain = thermostat.half_step(state.momentum, chain, _kT)
    state = state.set(momentum=p)

    state = velocity_verlet(force_fn, shift_fn, dt, state, **kwargs)

    chain = chain.set(kinetic_energy=kinetic_energy(state))

    p, chain = thermostat.half_step(state.momentum, chain, _kT)
    state = state.set(momentum=p, chain=chain)

    return state
  return init_fn, apply_fn


def nvt_nose_hoover_invariant(energy_fn: Callable[..., Array],
                              state: NVTNoseHooverState,
                              kT: float,
                              **kwargs) -> float:
  """The conserved quantity for the NVT ensemble with a Nose-Hoover thermostat.

  This function is normally used for debugging the Nose-Hoover thermostat.

  Arguments:
    energy_fn: The energy function of the Nose-Hoover system.
    state: The current state of the system.
    kT: The current goal temperature of the system.

  Returns:
    The Hamiltonian of the extended NVT dynamics.
  """
  PE = energy_fn(state.position, **kwargs)
  KE = kinetic_energy(state)

  DOF = quantity.count_dof(state.position)
  E = PE + KE

  c = state.chain

  E += c.momentum[0] ** 2 / (2 * c.mass[0]) + DOF * kT * c.position[0]
  for r, p, m in zip(c.position[1:], c.momentum[1:], c.mass[1:]):
    E += p ** 2 / (2 * m) + kT * r
  return E


@dataclasses.dataclass
class NPTNoseHooverState:
  """State information for an NPT system with Nose-Hoover chain thermostats.

  Attributes:
    position: The current position of particles. An ndarray of floats
      with shape `[n, spatial_dimension]`.
    momentum: The velocity of particles. An ndarray of floats
      with shape `[n, spatial_dimension]`.
    force: The current force on the particles. An ndarray of floats with shape
      `[n, spatial_dimension]`.
    mass: The mass of the particles. Can either be a float or an ndarray
      of floats with shape `[n]`.
    reference_box: A box used to measure relative changes to the simulation
      environment.
    box_position: A positional degree of freedom used to describe the current
      box. box_position is parameterized as `box_position = (1/d)log(V/V_0)`
      where `V` is the current volume, `V_0` is the reference volume, and `d`
      is the spatial dimension.
    box_velocity: A velocity degree of freedom for the box.
    box_mass: The mass assigned to the box.
    barostat: The variables describing the Nose-Hoover chain coupled to the
      barostat.
    thermostsat: The variables describing the Nose-Hoover chain coupled to the
      thermostat.
  """
  position: Array
  momentum: Array
  force: Array
  mass: Array

  reference_box: Box

  box_position: Array
  box_momentum: Array
  box_mass: Array

  barostat: NoseHooverChain
  thermostat: NoseHooverChain

  @property
  def velocity(self) -> Array:
    return self.momentum / self.mass

  @property
  def box(self) -> Array:
    """Get the current box from an NPT simulation."""
    dim = self.position.shape[1]
    ref = self.reference_box
    V_0 = quantity.volume(dim, ref)
    V = V_0 * jnp.exp(dim * self.box_position)
    return (V / V_0) ** (1 / dim) * ref


def _npt_box_info(state: NPTNoseHooverState
                  ) -> Tuple[float, Callable[[float], float]]:
  """Gets the current volume and a function to compute the box from volume."""
  dim = state.position.shape[1]
  ref = state.reference_box
  V_0 = quantity.volume(dim, ref)
  V = V_0 * jnp.exp(dim * state.box_position)
  return V, lambda V: (V / V_0) ** (1 / dim) * ref


def npt_box(state: NPTNoseHooverState) -> Box:
  """Get the current box from an NPT simulation."""
  dim = state.position.shape[1]
  ref = state.reference_box
  V_0 = quantity.volume(dim, ref)
  V = V_0 * jnp.exp(dim * state.box_position)
  return (V / V_0) ** (1 / dim) * ref


def npt_nose_hoover(energy_fn: Callable[..., Array],
                    shift_fn: ShiftFn,
                    dt: float,
                    pressure: float,
                    kT: float,
                    barostat_kwargs: Optional[Dict]=None,
                    thermostat_kwargs: Optional[Dict]=None) -> Simulator:
  """Simulation in the NPT ensemble using a pair of Nose Hoover Chains.

  Samples from the canonical ensemble in which the number of particles (N),
  the system pressure (P), and the temperature (T) are held constant.
  We use a pair of Nose Hoover Chains (NHC) described in
  [#martyna92]_ [#martyna98]_ [#tuckerman]_ coupled to the
  barostat and the thermostat respectively. We follow the direct translation
  method outlined in Tuckerman et al. [#tuckerman]_ and the interested reader
  might want to look at that paper as a reference.

  Args:
    energy_fn: A function that produces either an energy from a set of particle
      positions specified as an ndarray of shape `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`. Both
      `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    pressure: Floating point number specifying the target pressure. To update
      the pressure dynamically during a simulation one should pass `pressure`
      as a keyword argument to the step function.
    kT: Floating point number specifying the temperature in units of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
    barostat_kwargs: A dictionary of keyword arguments passed to the barostat
      NHC. Any parameters not set are drawn from a relatively robust default
      set.
    thermostat_kwargs: A dictionary of keyword arguments passed to the
      thermostat NHC. Any parameters not set are drawn from a relatively robust
      default set.

  Returns:
    See above.

  """

  t = f32(dt)
  dt_2 = f32(dt / 2)

  force_fn = quantity.force(energy_fn)

  barostat_kwargs = default_nhc_kwargs(1000 * dt, barostat_kwargs)
  barostat = nose_hoover_chain(dt, **barostat_kwargs)

  thermostat_kwargs = default_nhc_kwargs(100 * dt, thermostat_kwargs)
  thermostat = nose_hoover_chain(dt, **thermostat_kwargs)

  def init_fn(key, R, box, mass=f32(1.0), **kwargs):
    N, dim = R.shape

    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    # The box position is defined via pos = (1 / d) log V / V_0.
    zero = jnp.zeros((), dtype=R.dtype)
    one = jnp.ones((), dtype=R.dtype)
    box_position = zero
    box_momentum = zero
    box_mass = dim * (N + 1) * kT * barostat_kwargs['tau'] ** 2 * one
    KE_box = quantity.kinetic_energy(momentum=box_momentum, mass=box_mass)

    if jnp.isscalar(box) or box.ndim == 0:
      # TODO(schsam): This is necessary because of JAX issue #5849.
      box = jnp.eye(R.shape[-1]) * box

    state = NPTNoseHooverState(
      R, None, force_fn(R, box=box, **kwargs),
      mass, box, box_position, box_momentum, box_mass,
      barostat.initialize(1, KE_box, _kT),
      None)  # pytype: disable=wrong-arg-count
    state = canonicalize_mass(state)
    state = initialize_momenta(state, key, _kT)
    KE = kinetic_energy(state)
    return state.set(
      thermostat=thermostat.initialize(quantity.count_dof(R), KE, _kT))

  def update_box_mass(state, kT):
    N, dim = state.position.shape
    dtype = state.position.dtype
    box_mass = jnp.array(dim * (N + 1) * kT * state.barostat.tau ** 2, dtype)
    return state.set(box_mass=box_mass)

  def box_force(alpha, vol, box_fn, position, momentum, mass, force, pressure,
                **kwargs):
    N, dim = position.shape

    def U(eps):
      return energy_fn(position, box=box_fn(vol), perturbation=(1 + eps),
                       **kwargs)

    dUdV = grad(U)
    KE2 = util.high_precision_sum(momentum ** 2 / mass)

    return alpha * KE2 - dUdV(0.0) - pressure * vol * dim

  def sinhx_x(x):
    """Taylor series for sinh(x) / x as x -> 0."""
    return (1 + x ** 2 / 6 + x ** 4 / 120 + x ** 6 / 5040 +
            x ** 8 / 362_880 + x ** 10 / 39_916_800)

  def exp_iL1(box, R, V, V_b, **kwargs):
    x = V_b * dt
    x_2 = x / 2
    sinhV = sinhx_x(x_2)  # jnp.sinh(x_2) / x_2
    return shift_fn(R, R * (jnp.exp(x) - 1) + dt * V * jnp.exp(x_2) * sinhV,
                    box=box, **kwargs)  # pytype: disable=wrong-keyword-args

  def exp_iL2(alpha, P, F, V_b):
    x = alpha * V_b * dt_2
    x_2 = x / 2
    sinhP = sinhx_x(x_2)  # jnp.sinh(x_2) / x_2
    return P * jnp.exp(-x) + dt_2 * F * sinhP * jnp.exp(-x_2)

  def inner_step(state, **kwargs):
    _pressure = kwargs.pop('pressure', pressure)

    R, P, M, F = state.position, state.momentum, state.mass, state.force
    R_b, P_b, M_b = state.box_position, state.box_momentum, state.box_mass

    N, dim = R.shape

    vol, box_fn = _npt_box_info(state)

    alpha = 1 + 1 / N
    G_e = box_force(alpha, vol, box_fn, R, P, M, F, _pressure, **kwargs)
    P_b = P_b + dt_2 * G_e
    P = exp_iL2(alpha, P, F, P_b / M_b)

    R_b = R_b + P_b / M_b * dt
    state = state.set( box_position=R_b)

    vol, box_fn = _npt_box_info(state)

    box = box_fn(vol)
    R = exp_iL1(box, R, P / M, P_b / M_b)
    F = force_fn(R, box=box, **kwargs)

    P = exp_iL2(alpha, P, F, P_b / M_b)
    G_e = box_force(alpha, vol, box_fn, R, P, M, F, _pressure, **kwargs)
    P_b = P_b + dt_2 * G_e

    return state.set(position=R, momentum=P, mass=M, force=F,
                     box_position=R_b, box_momentum=P_b, box_mass=M_b)

  def apply_fn(state, **kwargs):
    S = state
    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    bc = barostat.update_mass(S.barostat, _kT)
    tc = thermostat.update_mass(S.thermostat, _kT)
    S = update_box_mass(S, _kT)

    P_b, bc = barostat.half_step(S.box_momentum, bc, _kT)
    P, tc = thermostat.half_step(S.momentum, tc, _kT)

    S = S.set(momentum=P, box_momentum=P_b)
    S = inner_step(S, **kwargs)

    KE = quantity.kinetic_energy(momentum=S.momentum, mass=S.mass)
    tc = tc.set(kinetic_energy=KE)

    KE_box = quantity.kinetic_energy(momentum=S.box_momentum, mass=S.box_mass)
    bc = bc.set(kinetic_energy=KE_box)

    P, tc = thermostat.half_step(S.momentum, tc, _kT)
    P_b, bc = barostat.half_step(S.box_momentum, bc, _kT)

    S = S.set(thermostat=tc, barostat=bc, momentum=P, box_momentum=P_b)

    return S
  return init_fn, apply_fn


def npt_nose_hoover_invariant(energy_fn: Callable[..., Array],
                              state: NPTNoseHooverState,
                              pressure: float,
                              kT: float,
                              **kwargs) -> float:
  """The conserved quantity for the NPT ensemble with a Nose-Hoover thermostat.

  This function is normally used for debugging the NPT simulation.

  Arguments:
    energy_fn: The energy function of the system.
    state: The current state of the system.
    pressure: The current goal pressure of the system.
    kT: The current goal temperature of the system.

  Returns:
    The Hamiltonian of the extended NPT dynamics.
  """
  volume, box_fn = _npt_box_info(state)
  PE = energy_fn(state.position, box=box_fn(volume), **kwargs)
  KE = kinetic_energy(state)

  DOF = state.position.size
  E = PE + KE

  c = state.thermostat
  E += c.momentum[0] ** 2 / (2 * c.mass[0]) + DOF * kT * c.position[0]
  for r, p, m in zip(c.position[1:], c.momentum[1:], c.mass[1:]):
    E += p ** 2 / (2 * m) + kT * r

  c = state.barostat
  for r, p, m in zip(c.position, c.momentum, c.mass):
    E += p ** 2 / (2 * m) + kT * r

  E += pressure * volume
  E += state.box_momentum ** 2 / (2 * state.box_mass)

  return E


"""Stochastic Simulations

JAX MD includes integrators for stochastic simulations of Langevin dynamics and
Brownian motion for systems in the NVT ensemble with a solvent.
"""


@dataclasses.dataclass
class Normal:
  """A simple normal distribution."""
  mean: jnp.ndarray
  var: jnp.ndarray

  def sample(self, key):
    mu, sigma = self.mean, jnp.sqrt(self.var)
    return mu + sigma * random.normal(key, mu.shape ,dtype=mu.dtype)

  def log_prob(self, x):
    return (-0.5 * jnp.log(2 * jnp.pi * self.var) -
            1 / (2 * self.var) * (x - self.mean)**2)


@dataclasses.dataclass
class NVTLangevinState:
  """A struct containing state information for the Langevin thermostat.

  Attributes:
    position: The current position of the particles. An ndarray of floats with
      shape `[n, spatial_dimension]`.
    momentum: The momentum of particles. An ndarray of floats with shape
      `[n, spatial_dimension]`.
    force: The (non-stochastic) force on particles. An ndarray of floats with
      shape `[n, spatial_dimension]`.
    mass: The mass of particles. Will either be a float or an ndarray of floats
      with shape `[n]`.
    rng: The current state of the random number generator.
  """
  position: Array
  momentum: Array
  force: Array
  mass: Array
  rng: Array

  @property
  def velocity(self) -> Array:
    return self.momentum / self.mass


@dispatch_by_state
def stochastic_step(state: NVTLangevinState, dt:float, kT: float, gamma: float):
  """A single stochastic step (the `O` step)."""
  c1 = jnp.exp(-gamma * dt)
  c2 = jnp.sqrt(kT * (1 - c1**2))
  momentum_dist = Normal(c1 * state.momentum, c2**2 * state.mass)
  key, split = random.split(state.rng)
  return state.set(momentum=momentum_dist.sample(split), rng=key)


def nvt_langevin(energy_or_force_fn: Callable[..., Array],
                 shift_fn: ShiftFn,
                 dt: float,
                 kT: float,
                 gamma: float=0.1,
                 center_velocity: bool=True,
                 **sim_kwargs) -> Simulator:
  """Simulation in the NVT ensemble using the BAOAB Langevin thermostat.

  Samples from the canonical ensemble in which the number of particles (N),
  the system volume (V), and the temperature (T) are held constant. Langevin
  dynamics are stochastic and it is supposed that the system is interacting
  with fictitious microscopic degrees of freedom. An example of this would be
  large particles in a solvent such as water. Thus, Langevin dynamics are a
  stochastic ODE described by a friction coefficient and noise of a given
  covariance.

  Our implementation follows the paper [#davidcheck] by Davidchack, Ouldridge,
  and Tretyakov.

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`. Both
      `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    kT: Floating point number specifying the temperature in units of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
    gamma: A float specifying the friction coefficient between the particles
      and the solvent.
    center_velocity: A boolean specifying whether or not the center of mass
      position should be subtracted.
  Returns:
    See above.

  .. rubric:: References
  .. [#carlon] R. L. Davidchack, T. E. Ouldridge, and M. V. Tretyakov.
    "New Langevin and gradient thermostats for rigid body dynamics."
    The Journal of Chemical Physics 142, 144114 (2015)
  """
  force_fn = quantity.canonicalize_force(energy_or_force_fn)

  @jit
  def init_fn(key, R, mass=f32(1.0), **kwargs):
    _kT = kwargs.pop('kT', kT)
    key, split = random.split(key)
    force = force_fn(R, **kwargs)
    state = NVTLangevinState(R, None, force, mass, key)
    state = canonicalize_mass(state)
    return initialize_momenta(state, split, _kT)

  @jit
  def step_fn(state, **kwargs):
    _dt = kwargs.pop('dt', dt)
    _kT = kwargs.pop('kT', kT)
    dt_2 = _dt / 2

    state = momentum_step(state, dt_2)
    state = position_step(state, shift_fn, dt_2, **kwargs)
    state = stochastic_step(state, _dt, _kT, gamma)
    state = position_step(state, shift_fn, dt_2, **kwargs)
    state = state.set(force=force_fn(state.position, **kwargs))
    state = momentum_step(state, dt_2)

    return state

  return init_fn, step_fn


@dataclasses.dataclass
class BrownianState:
  """A tuple containing state information for Brownian dynamics.

  Attributes:
    position: The current position of the particles. An ndarray of floats with
      shape `[n, spatial_dimension]`.
    mass: The mass of particles. Will either be a float or an ndarray of floats
      with shape `[n]`.
    rng: The current state of the random number generator.
  """
  position: Array
  mass: Array
  rng: Array


def brownian(energy_or_force: Callable[..., Array],
             shift: ShiftFn,
             dt: float,
             kT: float,
             gamma: float=0.1) -> Simulator:
  """Simulation of Brownian dynamics.

  Simulates Brownian dynamics which are synonymous with the overdamped
  regime of Langevin dynamics. However, in this case we don't need to take into
  account velocity information and the dynamics simplify. Consequently, when
  Brownian dynamics can be used they will be faster than Langevin. As in the
  case of Langevin dynamics our implementation follows Carlon et al. [#carlon]_

  Args:
    energy_or_force: A function that produces either an energy or a force from
      a set of particle positions specified as an ndarray of shape
      `[n, spatial_dimension]`.
    shift_fn: A function that displaces positions, `R`, by an amount `dR`.
      Both `R` and `dR` should be ndarrays of shape `[n, spatial_dimension]`.
    dt: Floating point number specifying the timescale (step size) of the
      simulation.
    kT: Floating point number specifying the temperature in units of Boltzmann
      constant. To update the temperature dynamically during a simulation one
      should pass `kT` as a keyword argument to the step function.
    gamma: A float specifying the friction coefficient between the particles
      and the solvent.

  Returns:
    See above.
  """

  force_fn = quantity.canonicalize_force(energy_or_force)

  dt, gamma = static_cast(dt, gamma)

  def init_fn(key, R, mass=f32(1)):
    state = BrownianState(R, mass, key)
    return canonicalize_mass(state)

  def apply_fn(state, **kwargs):
    _kT = kT if 'kT' not in kwargs else kwargs['kT']

    R, mass, key = dataclasses.astuple(state)

    key, split = random.split(key)

    F = force_fn(R, **kwargs)
    xi = random.normal(split, R.shape, R.dtype)

    nu = f32(1) / (mass * gamma)

    dR = F * dt * nu + jnp.sqrt(f32(2) * _kT * dt * nu) * xi
    R = shift(R, dR, **kwargs)

    return BrownianState(R, mass, key)  # pytype: disable=wrong-arg-count

  return init_fn, apply_fn


"""Experimental Simulations.


Below are simulation environments whose implementation is somewhat
experimental / preliminary. These environments might not be as ergonomic
as the more polished environments above.
"""


@dataclasses.dataclass
class SwapMCState:
  """A struct containing state information about a Hybrid Swap MC simulation.

  Attributes:
    md: A NVTNoseHooverState containing continuous molecular dynamics data.
    sigma: An `[n,]` array of particle radii.
    key: A JAX PRGNKey used for random number generation.
    neighbor: A NeighborList for the system.
  """
  md: NVTNoseHooverState
  sigma: Array
  key: Array
  neighbor: partition.NeighborList


# pytype: disable=wrong-arg-count
# pytype: disable=wrong-keyword-args
def hybrid_swap_mc(space_fns: space.Space,
                   energy_fn: Callable[[Array, Array], Array],
                   neighbor_fn: partition.NeighborFn,
                   dt: float,
                   kT: float,
                   t_md: float,
                   N_swap: int,
                   sigma_fn: Optional[Callable[[Array], Array]]=None
                   ) -> Simulator:
  """Simulation of Hybrid Swap Monte-Carlo.

  This code simulates the hybrid Swap Monte Carlo algorithm introduced in
  Berthier et al. [#berthier]_
  Here an NVT simulation is performed for `t_md` time and then `N_swap` MC
  moves are performed that swap the radii of randomly chosen particles. The
  random swaps are accepted with Metropolis-Hastings step. Each call to the
  step function runs molecular dynamics for `t_md` and then performs the swaps.

  Note that this code doesn't feature some of the convenience functions in the
  other simulations. In particular, there is no support for dynamics keyword
  arguments and the energy function must be a simple callable of two variables:
  the distance between adjacent particles and the diameter of the particles.
  If you want support for a better notion of potential or dynamic keyword
  arguments, please file an issue!

  Args:
    space_fns: A tuple of a displacement function and a shift function defined
      in `space.py`.
    energy_fn: A function that computes the energy between one pair of
      particles as a function of the distance between the particles and the
      diameter. This function should not have been passed to `smap.xxx`.
    neighbor_fn: A function to construct neighbor lists outlined in
      `partition.py`.
    dt: The timestep used for the continuous time MD portion of the simulation.
    kT: The temperature of heat bath that the system is coupled to during MD.
    t_md: The time of each MD block.
    N_swap: The number of swapping moves between MD blocks.
    sigma_fn: An optional function for combining radii if they are to be
      non-additive.

  Returns:
    See above.

  .. rubric:: References
  .. [#berthier] L. Berthier, E. Flenner, C. J. Fullerton, C. Scalliet, and M. Singh.
    "Efficient swap algorithms for molecular dynamics simulations of
    equilibrium supercooled liquids", J. Stat. Mech. (2019) 064004
  """
  displacement_fn, shift_fn = space_fns
  metric_fn = space.metric(displacement_fn)
  nbr_metric_fn = space.map_neighbor(metric_fn)

  md_steps = int(t_md // dt)

  # Canonicalize the argument names to be dr and sigma.
  wrapped_energy_fn = lambda dr, sigma: energy_fn(dr, sigma)
  if sigma_fn is None:
    sigma_fn = lambda si, sj: 0.5 * (si + sj)
  nbr_energy_fn = smap.pair_neighbor_list(wrapped_energy_fn,
                                          metric_fn,
                                          sigma=sigma_fn)

  nvt_init_fn, nvt_step_fn = nvt_nose_hoover(nbr_energy_fn,
                                             shift_fn,
                                             dt,
                                             kT=kT,
                                             chain_length=3)
  def init_fn(key, position, sigma, nbrs=None):
    key, sim_key = random.split(key)
    nbrs = neighbor_fn(position, nbrs)  # pytype: disable=wrong-arg-count
    md_state = nvt_init_fn(sim_key, position, neighbor=nbrs, sigma=sigma)
    return SwapMCState(md_state, sigma, key, nbrs)  # pytype: disable=wrong-arg-count

  def md_step_fn(i, state):
    md, sigma, key, nbrs = dataclasses.unpack(state)
    md = nvt_step_fn(md, neighbor=nbrs, sigma=sigma)  # pytype: disable=wrong-keyword-args
    nbrs = neighbor_fn(md.position, nbrs)
    return SwapMCState(md, sigma, key, nbrs)  # pytype: disable=wrong-arg-count

  def swap_step_fn(i, state):
    md, sigma, key, nbrs = dataclasses.unpack(state)

    N = md.position.shape[0]

    # Swap a random pair of particle radii.
    key, particle_key, accept_key = random.split(key, 3)
    ij = random.randint(particle_key, (2,), jnp.array(0), jnp.array(N))
    new_sigma = sigma.at[ij].set([sigma[ij[1]], sigma[ij[0]]])

    # Collect neighborhoods around the two swapped particles.
    nbrs_ij = nbrs.idx[ij]
    R_ij = md.position[ij]
    R_neigh = md.position[nbrs_ij]

    sigma_ij = sigma[ij][:, None]
    sigma_neigh = sigma[nbrs_ij]

    new_sigma_ij = new_sigma[ij][:, None]
    new_sigma_neigh = new_sigma[nbrs_ij]

    dR = nbr_metric_fn(R_ij, R_neigh)

    # Compute the energy before the swap.
    energy = energy_fn(dR, sigma_fn(sigma_ij, sigma_neigh))
    energy = jnp.sum(energy * (nbrs_ij < N))

    # Compute the energy after the swap.
    new_energy = energy_fn(dR, sigma_fn(new_sigma_ij, new_sigma_neigh))
    new_energy = jnp.sum(new_energy * (nbrs_ij < N))

    # Accept or reject with a metropolis probability.
    p = random.uniform(accept_key, ())
    accept_prob = jnp.minimum(1, jnp.exp(-(new_energy - energy) / kT))
    sigma = jnp.where(p < accept_prob, new_sigma, sigma)

    return SwapMCState(md, sigma, key, nbrs)  # pytype: disable=wrong-arg-count

  def block_fn(state):
    state = lax.fori_loop(0, md_steps, md_step_fn, state)
    state = lax.fori_loop(0, N_swap, swap_step_fn, state)
    return state

  return init_fn, block_fn
# pytype: enable=wrong-arg-count
# pytype: enable=wrong-keyword-args










jax.config.update('jax_enable_x64', True)

lambda_c = 0.1931833275037836 #critical value of the lambda parameter for the minimal norm integrator



class Sampler:
    """the MCHMC (q = 0 Hamiltonian) sampler"""

    def __init__(self, Target, shift_fn, L = None, eps = None,
                 integrator = 'MN', varEwanted = 5e-4,
                 diagonal_preconditioning= True, sg = False,
                 frac_tune1 = 0.1, frac_tune2 = 0.1, frac_tune3 = 0.1):
        """Args:
                Target: the target distribution class

                L: momentum decoherence scale (it is then automaticaly tuned before the sampling starts unless you turn-off the tuning by setting frac_tune2 and 3 to zero (see below))

                eps: initial integration step-size (it is then automaticaly tuned before the sampling starts unless you turn-off the tuning by setting all frac_tune1 and 2 to zero (see below))

                integrator: 'LF' (leapfrog) or 'MN' (minimal norm). Typically MN performs better.

                varEwanted: if your posteriors are biased try smaller values (or larger values: perhaps the convergence is too slow). This is perhaps the parameter whose default value is the least well determined.

                diagonal_preconditioning: if you already have your own preconditioning or if you suspect diagonal preconditioning is not useful, turn this off as it can also make matters worse
                                          (but it can also be very useful if you did not precondition the parameters (make their posterior variances close to 1))

                frac_tune1: (num_samples * frac_tune1) steps will be used as a burn-in and to autotune the stepsize

                frac_tune2: (num_samples * frac_tune2) steps will be used to autotune L (should be around 10 effective samples long for the optimal performance)

                frac_tune3: (num_samples * frac_tune3) steps will be used to improve the L tuning (should be around 10 effective samples long for the optimal performance). This stage is not neccessary if the posterior is close to a Gaussian and does not change much in general.
                            It can be memory intensive in high dimensions so try turning it off if you have problems with the memory.
        """

        self.Target = Target
        self.shift_fn = shift_fn

        ### integrator ###
        if integrator == "LF": #leapfrog (first updates the velocity)
            self.hamiltonian_dynamics = self.leapfrog
            self.grad_evals_per_step = 1.0

        elif integrator== 'MN': #minimal norm integrator (velocity)
            self.hamiltonian_dynamics = self.minimal_norm
            self.grad_evals_per_step = 2.0
        # elif integrator == 'RM':
        #     self.hamiltonian_dynamics = self.randomized_midpoint
        #     self.grad_evals_per_step = 1.0
        else:
            raise ValueError('integrator = ' + integrator + 'is not a valid option.')


        ### option of stochastic gradient ###
        self.sg = sg
        self.dynamics = self.dynamics_generalized_sg if sg else self.dynamics_generalized

        ### preconditioning ###
        self.diagonal_preconditioning = diagonal_preconditioning

        ### autotuning parameters ###

        # length of autotuning
        self.frac_tune1 = frac_tune1 # num_samples * frac_tune1 steps will be used to autotune eps
        self.frac_tune2 = frac_tune2 # num_samples * frac_tune2 steps will be used to approximately autotune L
        self.frac_tune3 = frac_tune3 # num_samples * frac_tune3 steps will be used to improve L tuning.

        self.varEwanted = varEwanted # 1e-3 #targeted energy variance Var[E]/d
        neff = 50 # effective number of steps used to determine the stepsize in the adaptive step
        self.gamma = (neff - 1.0) / (neff + 1.0) # forgeting factor in the adaptive step
        self.sigma_xi= 1.5 # determines how much do we trust the stepsize predictions from the too large and too small stepsizes

        self.Lfactor = 0.4 #in the third stage we set L = Lfactor * (configuration space distance bewteen independent samples)


        ### default eps and L ###
        if L != None:
            self.L = L
        else: #default value (works if the target is well preconditioned). If you are not happy with the default value and have not run the grid search we suggest runing sample with the option tune= 'expensive'.
            self.L = jnp.sqrt(Target.d)
        if eps != None:
            self.eps = eps
        else: #defualt value (assumes preconditioned target and even then it might not work). Unless you have done a grid search to determine this value we suggest runing sample with the option tune= 'cheap' or tune= 'expensive'.
            self.eps = jnp.sqrt(Target.d) * 0.4



    def random_unit_vector(self, random_key):
        """Generates a random (isotropic) unit vector."""
        key, subkey = jax.random.split(random_key)
        u = jax.random.normal(subkey, shape = (self.Target.d, ), dtype = 'float64')
        u /= jnp.sqrt(jnp.sum(jnp.square(u)))
        return u, key


    def partially_refresh_momentum(self, u, nu, random_key):
        """Adds a small noise to u and normalizes."""
        key, subkey = jax.random.split(random_key)
        z = nu * jax.random.normal(subkey, shape = (self.Target.d, ), dtype = 'float64')

        return (u + z) / jnp.sqrt(jnp.sum(jnp.square(u + z))), key

    # naive update
    # def update_momentum(self, eps, g, u):
    #     """The momentum updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)"""
    #     g_norm = jnp.sqrt(jnp.sum(jnp.square(g)))
    #     e = - g / g_norm
    #     ue = jnp.dot(u, e)
    #     sh = jnp.sinh(eps * g_norm / (self.Target.d-1))
    #     ch = jnp.cosh(eps * g_norm / (self.Target.d-1))
    #     th = jnp.tanh(eps * g_norm / (self.Target.d-1))
    #     delta_r = jnp.log(ch) + jnp.log1p(ue * th)
    #
    #     return (u + e * (sh + ue * (ch - 1))) / (ch + ue * sh), delta_r


    def update_momentum(self, eps, g, u):
        """The momentum updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)
        similar to the implementation: https://github.com/gregversteeg/esh_dynamics
        There are no exponentials e^delta, which prevents overflows when the gradient norm is large."""
        g_norm = jnp.sqrt(jnp.sum(jnp.square(g)))
        e = - g / g_norm
        ue = jnp.dot(u, e)
        delta = eps * g_norm / (self.Target.d-1)
        zeta = jnp.exp(-delta)
        uu = e *(1-zeta)*(1+zeta + ue * (1-zeta)) + 2*zeta* u
        delta_r = delta - jnp.log(2) + jnp.log(1 + ue + (1-ue)*zeta**2)
        return uu/jnp.sqrt(jnp.sum(jnp.square(uu))), delta_r


    def leapfrog(self, x, u, g, random_key, eps, sigma):
        """leapfrog"""

        z = x / sigma # go to the latent space

        # half step in momentum
        uu, delta_r1 = self.update_momentum(eps * 0.5, g * sigma, u)

        # full step in x
        zz = z + eps * uu
        xx = sigma * zz # go back to the configuration space
        l, gg = self.Target.grad_nlogp(xx)

        # half step in momentum
        uu, delta_r2 = self.update_momentum(eps * 0.5, gg * sigma, uu)
        kinetic_change = (delta_r1 + delta_r2) * (self.Target.d-1)

        return xx, uu, l, gg, kinetic_change, random_key

    # def leapfrog(self, x, u, g, random_key, eps, sigma):
    #     """leapfrog"""
    #
    #     z = x / sigma  # go to the latent space
    #
    #     # half step in x
    #     zz = z + 0.5 * eps * u
    #     l, gg = self.Target.grad_nlogp(sigma * zz)
    #
    #     # full step in momentum
    #     uu, delta_r = self.update_momentum(eps, gg * sigma, u)
    #
    #     # half step in x
    #     zz += 0.5 * eps * uu
    #     xx = sigma * zz  # go back to the configuration space
    #
    #     l = self.Target.nlogp(xx)
    #     kinetic_change = delta_r * (self.Target.d - 1)
    #
    #     return xx, uu, l, gg, kinetic_change, random_key

    def leapfrog_sg(self, x, u, g, random_key, eps, sigma, data):
        """leapfrog"""

        z = x / sigma # go to the latent space

        # half step in momentum
        uu, delta_r1 = self.update_momentum(eps * 0.5, g * sigma, u)

        # full step in x
        zz = z + eps * uu
        xx = sigma * zz # go back to the configuration space
        l, gg = self.Target.grad_nlogp(xx, data)

        # half step in momentum
        uu, delta_r2 = self.update_momentum(eps * 0.5, gg * sigma, uu)
        kinetic_change = (delta_r1 + delta_r2) * (self.Target.d-1)

        return xx, uu, l, gg, kinetic_change, random_key

    #
    # def leapfrog_sg(self, x, u, g, random_key, eps, sigma, data):
    #     """leapfrog"""
    #
    #     z = x / sigma # go to the latent space
    #
    #     # half step in x
    #     zz = z + 0.5 * eps * u
    #     xx = sigma * zz # go back to the configuration space
    #     l, gg = self.Target.grad_nlogp(xx, data)
    #
    #     # half step in momentum
    #     uu, delta_r = self.update_momentum(eps, gg * sigma, u)
    #
    #     # full step in x
    #     zz += 0.5 * eps * uu
    #     xx = sigma * zz # go back to the configuration space
    #
    #     kinetic_change = delta_r * (self.Target.d-1)
    #
    #     return xx, uu, l, gg, kinetic_change, random_key



    def minimal_norm(self, x, u, g, random_key, eps, sigma):
        """Integrator from https://arxiv.org/pdf/hep-lat/0505020.pdf, see Equation 20."""

        # V T V T V
        # CHANGED: don't move to latent space
        z = x

        #V (momentum update)
        uu, r1 = self.update_momentum(eps * lambda_c, g * sigma, u)

        #T (postion update)
        zz = self.shift_fn(z, (0.5 * eps * uu) / sigma)
        xx = zz 
        ll, gg = self.Target.grad_nlogp(xx)

        #V (momentum update)
        uu, r2 = self.update_momentum(eps * (1 - 2 * lambda_c), gg * sigma, uu)

        #T (postion update)
        zz = self.shift_fn(zz, (0.5 * eps * uu) / sigma)
        xx = zz  
        ll, gg = self.Target.grad_nlogp(xx)

        #V (momentum update)
        uu, r3 = self.update_momentum(eps * lambda_c, gg, uu)

        #kinetic energy change
        kinetic_change = (r1 + r2 + r3) * (self.Target.d-1)

        return xx, uu, ll, gg, kinetic_change, random_key


    #
    # def randomized_midpoint(self, x, u, g, r, key):
    #
    #     key1, key2 = jax.random.split(key)
    #
    #     xx = x + jax.random.uniform(key2) * self.eps * u
    #
    #     gg = self.Target.grad_nlogp(xx)
    #
    #     uu, r1 = self.update_momentum(self.eps, gg, u)
    #
    #     xx = self.update_position_RM(xx, )
    #
    #
    #     return xx, uu, gg, r1 * (self.Target.d-1), key1



    def dynamics_bounces(self, x, u, g, random_key, time, L, eps, sigma):
        """One step of the dynamics (with bounces)"""

        # Hamiltonian step
        xx, uu, ll, gg, kinetic_change, key = self.hamiltonian_dynamics(x, u, g, random_key, eps, sigma)

        # bounce
        u_bounce, key = self.random_unit_vector(key)
        time += eps
        do_bounce = time > L
        time = time * (1 - do_bounce)  # reset time if the bounce is done
        u_return = uu * (1 - do_bounce) + u_bounce * do_bounce  # randomly reorient the momentum if the bounce is done

        return xx, u_return, ll, gg, kinetic_change, key, time


    def dynamics_generalized(self, x, u, g, random_key, time, L, eps, sigma):
        """One step of the generalized dynamics."""

        # Hamiltonian step
        xx, uu, ll, gg, kinetic_change, key = self.hamiltonian_dynamics(x, u, g, random_key, eps, sigma)

        # Langevin-like noise
        nu = jnp.sqrt((jnp.exp(2 * eps / L) - 1.0) / self.Target.d)
        uu, key = self.partially_refresh_momentum(uu, nu, key)

        return xx, uu, ll, gg, kinetic_change, key, time + eps



    def dynamics_generalized_sg(self, x, u, g, random_key, time, L, eps, sigma):
        """One sweep over the entire dataset. Perfomrs self.Target.num_batches steps with the stochastic gradient."""

        #reshufle data and arange in batches

        key_reshuffle, key = jax.random.split(random_key)
        data_shape = self.Target.data.shape
        data = jax.random.permutation(key_reshuffle, self.Target.data).reshape(self.Target.num_batches, data_shape[0]//self.Target.num_batches, data_shape[1])

        def substep(state, data_batch):
            x, u, l, g, key, K, t = state
            # Hamiltonian step
            xx, uu, ll, gg, dK, key = self.leapfrog_sg(x, u, g, key, eps, sigma, data_batch)

            # Langevin-like noise
            nu = jnp.sqrt((jnp.exp(2 * eps / L) - 1.0) / self.Target.d)
            uu, key = self.partially_refresh_momentum(uu, nu, key)

            return (xx, uu, ll, gg, key, K + dK, t + eps), None

        xx, uu, ll, gg, key, kinetic_change, time = jax.lax.scan(substep, init= (x, u, 0.0, g, key, 0.0, time), xs= data, length= self.Target.num_batches)[0]

        return xx, uu, ll, gg, kinetic_change, key, time



    def nan_reject(self, x, u, l, g, t, xx, uu, ll, gg, tt, eps, eps_max, kk):
        """if there are nans, let's reduce the stepsize, and not update the state. The function returns the old state in this case."""
        tru = jnp.all(jnp.isfinite(xx))
        false = (1 - tru)
        return tru,\
               jnp.nan_to_num(xx) * tru + x * false, \
               jnp.nan_to_num(uu) * tru + u * false, \
               jnp.nan_to_num(ll) * tru + l * false, \
               jnp.nan_to_num(gg) * tru + g * false, \
               jnp.nan_to_num(tt) * tru + t * false, \
               eps_max * tru + 0.8 * eps * false, \
               jnp.nan_to_num(kk) * tru


    def dynamics_adaptive(self, state, L, sigma):
        """One step of the dynamics with the adaptive stepsize"""

        x, u, l, g, E, Feps, Weps, eps_max, key, t = state

        eps = jnp.power(Feps/Weps, -1.0/6.0) #We use the Var[E] = O(eps^6) relation here.
        eps = (eps < eps_max) * eps + (eps > eps_max) * eps_max  # if the proposed stepsize is above the stepsize where we have seen divergences

        # dynamics
        xx, uu, ll, gg, kinetic_change, key, tt = self.dynamics(x, u, g, key, t, L, eps, sigma)

        # step updating
        success, xx, uu, ll, gg, time, eps_max, kinetic_change = self.nan_reject(x, u, l, g, t, xx, uu, ll, gg, tt, eps, eps_max, kinetic_change)

        DE = kinetic_change + ll - l  # energy difference
        EE = E + DE  # energy
        # Warning: var = 0 if there were nans, but we will give it a very small weight
        xi = ((DE ** 2) / (self.Target.d * self.varEwanted)) + 1e-8  # 1e-8 is added to avoid divergences in log xi
        w = jnp.exp(-0.5 * jnp.square(jnp.log(xi) / (6.0 * self.sigma_xi)))  # the weight which reduces the impact of stepsizes which are much larger on much smaller than the desired one.
        Feps = self.gamma * Feps + w * (xi/jnp.power(eps, 6.0))  # Kalman update the linear combinations
        Weps = self.gamma * Weps + w

        return xx, uu, ll, gg, EE, Feps, Weps, eps_max, key, time, eps * success



    ### sampling routine ###

    def get_initial_conditions(self, x_initial, random_key):

        ### random key ###
        if random_key is None:
            key = jax.random.PRNGKey(0)
        else:
            key = random_key

        ### initial conditions ###
        if isinstance(x_initial, str):
            if x_initial == 'prior':  # draw the initial x from the prior
                key, prior_key = jax.random.split(key)
                x = self.Target.prior_draw(prior_key)
            else:  # if not 'prior' the x_initial should specify the initial condition
                raise KeyError('x_initial = "' + x_initial + '" is not a valid argument. \nIf you want to draw initial condition from a prior use x_initial = "prior", otherwise specify the initial condition with an array')
        else: #initial x is given
            x = x_initial

        l, g = self.Target.grad_nlogp(x)

        u, key = self.random_unit_vector(key)
        #u = - g / jnp.sqrt(jnp.sum(jnp.square(g))) #initialize momentum in the direction of the gradient of log p

        return x, u, l, g, key



    def sample(self, num_steps, num_chains = 1, x_initial = 'prior', random_key= None, output = 'normal', thinning= 1, adaptive = False):
        """Args:
               num_steps: number of integration steps to take.

               num_chains: number of independent chains, defaults to 1. If different than 1, jax will parallelize the computation with the number of available devices (CPU, GPU, TPU),
               as returned by jax.local_device_count().

               x_initial: initial condition for x, shape: (d, ). Defaults to 'prior' in which case the initial condition is drawn from the prior distribution (self.Target.prior_draw).

               random_key: jax random seed, defaults to jax.random.PRNGKey(0)

               output: determines the output of the function:

                        'normal': samples, burn in steps.
                            samples were transformed by the Target.transform to save memory and have shape: (num_samples, len(Target.transform(x)))

                        'expectation': exepcted value of transform(x)
                            most memory efficient. If you are after memory it might be usefull to turn off the third tuning stage

                        'detailed': samples, energy for each step, L and eps used for sampling

                        'ess': Effective Sample Size per gradient evaluation, float.
                            In this case, self.Target.variance = <x_i^2>_true should be defined.

                thinning: only one every 'thinning' steps is stored. Defaults to 1.
                        This is not the recommended solution to save memory. It is better to use the transform functionality.
                        If this is not sufficient consider saving only the expected values, by setting output= 'expectation'.

               adaptive: use the adaptive step size for sampling. This is experimental and not well developed yet.
        """

        if num_chains == 1:
            results = self.single_chain_sample(num_steps, x_initial, random_key, output, thinning, adaptive) #the function which actually does the sampling
            if output == 'ess':
                import matplotlib.pyplot as plt
                plt.plot(jnp.sqrt(results))
                plt.plot([0, len(results)], np.ones(2) * 0.1, '--', color='black', alpha=0.5)
                plt.yscale('log')
                plt.show()

                cutoff_reached = results[-1] < 0.01
                return (100.0 / (find_crossing(results, 0.01) * self.grad_evals_per_step)) * cutoff_reached
            else:
                return results
        else:
            num_cores = jax.local_device_count()
            if random_key is None:
                key = jax.random.PRNGKey(0)
            else:
                key = random_key

            if isinstance(x_initial, str):
                if x_initial == 'prior':  # draw the initial x from the prior
                    keys_all = jax.random.split(key, num_chains * 2)
                    x0 = jnp.array([self.Target.prior_draw(keys_all[num_chains+i]) for i in range(num_chains)])
                    keys = keys_all[:num_chains]

                else:  # if not 'prior' the x_initial should specify the initial condition
                    raise KeyError('x_initial = "' + x_initial + '" is not a valid argument. \nIf you want to draw initial condition from a prior use x_initial = "prior", otherwise specify the initial condition with an array')
            else: #initial x is given
                x0 = jnp.copy(x_initial)
                keys = jax.random.split(key, num_chains)


            f = lambda i: self.single_chain_sample(num_steps, x0[i], keys[i], output, thinning, adaptive)

            if num_cores != 1: #run the chains on parallel cores
                parallel_function = jax.pmap(jax.vmap(f))
                results = parallel_function(jnp.arange(num_chains).reshape(num_cores, num_chains // num_cores))
                if output == 'ess' or output == 'ess funnel':
                    bsq = jnp.average(results.reshape(results.shape[0] * results.shape[1], results.shape[2]), axis = 0)

                    import matplotlib.pyplot as plt
                    plt.plot(jnp.sqrt(bsq))
                    plt.plot([0, len(bsq)], np.ones(2) * 0.1, '--', color = 'black', alpha= 0.5)
                    plt.yscale('log')
                    plt.show()

                    cutoff_reached = bsq[-1] < 0.01
                    return (100.0 / (find_crossing(bsq, 0.01) * self.grad_evals_per_step) ) * cutoff_reached

                ### reshape results ###
                if type(results) is tuple: #each chain returned a tuple
                    results_reshaped =[]
                    for i in range(len(results)):
                        res = jnp.array(results[i])
                        results_reshaped.append(res.reshape([num_chains, ] + [res.shape[j] for j in range(2, len(res.shape))]))
                    return results_reshaped

                else:
                    return results.reshape([num_chains, ] + [results.shape[j] for j in range(2, len(results.shape))])


            else: #run chains serially on a single core

                return jax.vmap(f)(jnp.arange(num_chains))



    def single_chain_sample(self, num_steps, x_initial, random_key, output, thinning, adaptive):
        """sampling routine. It is called by self.sample"""

        ### initial conditions ###
        x, u, l, g, key = self.get_initial_conditions(x_initial, random_key)
        L, eps = self.L, self.eps #the initial values, given at the class initialization (or set to the default values)

        sigma = jnp.ones(self.Target.d)  # no diagonal preconditioning

        ### auto-tune the hyperparameters L and eps ###
        if self.frac_tune1 + self.frac_tune2 + self.frac_tune3 != 0.0:
            L, eps, sigma, x, u, l, g, key = self.tune12(x, u, l, g, key, L, eps, sigma, (int)(num_steps * self.frac_tune1), (int)(num_steps * self.frac_tune2)) #the cheap tuning (100 steps)
            if self.frac_tune3 != 0: #if we want to further improve L tuning we go to the second stage (which is a bit slower)
                L, x, u, l, g, key = self.tune3(x, u, l, g, key, L, eps, sigma, (int)(num_steps * self.frac_tune3))

        ### sampling ###

        if adaptive: #adaptive stepsize

            if output == 'normal' or output == 'detailed':
                X, W, _, E = self.sample_adaptive_normal(num_steps, x, u, l, g, key, L, eps, sigma)

                if output == 'detailed':
                    return X, W, E, L
                else:
                    return X, W

            elif output == 'ess':  # return the samples X
                return self.sample_adaptive_ess(num_steps, x, u, l, g, key, L, eps, sigma)
            elif output == 'expectation':
                raise ValueError('output = ' + output + ' is not yet implemented for the adaptive step-size. Let me know if you need it.')
            else:
                raise ValueError('output = ' + output + ' is not a valid argument for the Sampler.sample')


        else: #fixed stepsize

            if output == 'normal' or output == 'detailed':
                X, _, E = self.sample_normal(num_steps, x, u, l, g, key, L, eps, sigma, thinning)
                if output == 'detailed':
                    return X, E, L, eps
                else:
                    return X
            elif output == 'expectation':
                return self.sample_expectation(num_steps, x, u, l, g, key, L, eps, sigma)

            elif output == 'ess':
                return self.sample_ess(num_steps, x, u, l, g, key, L, eps, sigma)

            elif output == 'ess funnel':
                return self.sample_ess_funnel(num_steps, x, u, l, g, key, L, eps, sigma)

            else:
                raise ValueError('output = ' + output + 'is not a valid argument for the Sampler.sample')


    ### for loops which do the sampling steps: ###

    def sample_normal(self, num_steps, x, u, l, g, random_key, L, eps, sigma, thinning):
        """Stores transform(x) for each step."""
        
        def step(state, useless):

            x, u, l, g, E, key, time = state
            xx, uu, ll, gg, kinetic_change, key, time = self.dynamics(x, u, g, key, time, L, eps, sigma)
            EE = E + kinetic_change + ll - l
            return (xx, uu, ll, gg, EE, key, time), (self.Target.transform(xx), ll, EE)

        if thinning == 1:
            return jax.lax.scan(step, init=(x, u, l, g, 0.0, random_key, 0.0), xs=None, length=num_steps)[1]

        else:
            return self.sample_thinning(num_steps, x, u, l, g, random_key, L, eps, sigma, thinning)


    def sample_thinning(self, num_steps, x, u, l, g, random_key, L, eps, sigma, thinning):
        """Stores transform(x) for each step."""

        def step(state, useless):

            def substep(state, useless):
                x, u, l, g, E, key, time = state
                xx, uu, ll, gg, kinetic_change, key, time = self.dynamics(x, u, g, key, time, L, eps, sigma)
                EE = E + kinetic_change + ll - l
                return (xx, uu, ll, gg, EE, key, time), None

            state = jax.lax.scan(substep, init=state, xs=None, length= thinning)[0] #do 'thinning' steps without saving

            return state, (self.Target.transform(state[0]), state[2], state[4]) #save one sample

        return jax.lax.scan(step, init=(x, u, l, g, 0.0, random_key, 0.0), xs=None, length= num_steps // thinning)[1]



    def sample_expectation(self, num_steps, x, u, l, g, random_key, L, eps, sigma):
        """Stores no history but keeps the expected value of transform(x)."""
        
        def step(state, useless):
            
            x, u, g, key, time = state[0]
            x, u, _, g, _, key, time = self.dynamics(x, u, g, key, time, L, eps, sigma)
            W, F = state[1]
        
            F = (W * F + self.Target.transform(x)) / (W + 1)  # Update <f(x)> with a Kalman filter
            W += 1
            return ((x, u, g, key, time), (W, F)), None


        return jax.lax.scan(step, init=(x, u, g, random_key, 0.0), xs=None, length=num_steps)[0][1][1]



    def sample_ess(self, num_steps, x, u, l, g, random_key, L, eps, sigma):
        """Stores the bias of the second moments for each step."""
        
        def step(state_track, useless):
            
            x, u, l, g, E, key, time = state_track[0]
            x, u, ll, g, kinetic_change, key, time = self.dynamics(x, u, g, key, time, L, eps, sigma)
            W, F2 = state_track[1]
        
            F2 = (W * F2 + jnp.square(self.Target.transform(x))) / (W + 1)  # Update <f(x)> with a Kalman filter
            W += 1
            bias_d = jnp.square(F2 - self.Target.second_moments) / self.Target.variance_second_moments
            bias = jnp.average(bias_d)
            #bias = jnp.max(bias_d)

            return ((x, u, ll, g, E + kinetic_change + ll - l, key, time), (W, F2)), bias

        
        _, b = jax.lax.scan(step, init=((x, u, l, g, 0.0, random_key, 0.0), (1, jnp.square(self.Target.transform(x)))), xs=None, length=num_steps)

        #nans = jnp.any(jnp.isnan(b))

        return b #+ nans * 1e5 #return a large bias if there were nans


    def sample_ess_funnel(self, num_steps, x, u, l, g, random_key, L, eps, sigma):
        """Stores the bias of the second moments for each step."""

        def step(state_track, useless):
            x, u, l, g, E, key, time = state_track[0]
            eps1 = eps * jnp.exp(0.5 * x[-1])
            eps_max = eps *0.5#* jnp.exp(0.5)
            too_large = eps1 > eps_max
            eps_real = eps1 * (1-too_large) + eps_max * too_large
            x, u, ll, g, kinetic_change, key, time = self.dynamics(x, u, g, key, time, L, eps_real, sigma)
            W, F2 = state_track[1]
            F2 = (W * F2 + eps_real * jnp.square(self.Target.transform(x))) / (W + eps_real)  # Update <f(x)> with a Kalman filter
            W += eps_real
            bias = jnp.average(jnp.square((F2 - self.Target.variance) / self.Target.variance))
            # bias = jnp.average((F2 - self.Target.variance) / self.Target.variance)

            return ((x, u, ll, g, E + kinetic_change + ll - l, key, time), (W, F2)), bias

        _, b = jax.lax.scan(step, init=((x, u, l, g, 0.0, random_key, 0.0), (eps * jnp.exp(0.5 * x[-1]), jnp.square(self.Target.transform(x)))),
                            xs=None, length=num_steps)

        return b  # + nans * 1e5 #return a large bias if there were nans


    def sample_adaptive_normal(self, num_steps, x, u, l, g, random_key, L, eps, sigma):
        """Stores transform(x) for each iteration. It uses the adaptive stepsize."""

        def step(state, useless):
            
            x, u, l, g, E, Feps, Weps, eps_max, key, time, eps = self.dynamics_adaptive(state, L, sigma)

            return (x, u, l, g, E, Feps, Weps, eps_max, key, time), (self.Target.transform(x), l, E, eps)

        state, track = jax.lax.scan(step, init=(x, u, l, g, 0.0, jnp.power(eps, -6.0) * 1e-5, 1e-5, jnp.inf, random_key, 0.0), xs=None, length=num_steps)
        X, nlogp, E, eps = track
        W = jnp.concatenate((0.5 * (eps[1:] + eps[:-1]), 0.5 * eps[-1:]))  # weights (because Hamiltonian time does not flow uniformly if the step size changes)
        
        return X, W, nlogp, E


    def sample_adaptive_ess(self, num_steps, x, u, l, g, random_key, L, eps, sigma):
        """Stores the bias of the second moments for each step."""

        def step(state, useless):
            x, u, l, g, E, Feps, Weps, eps_max, key, time, eps = self.dynamics_adaptive(state[0], L, sigma)

            W, F2 = state[1]
            w = eps
            F2 = (W * F2 + w * jnp.square(self.Target.transform(x))) / (W + w)  # Update <f(x)> with a Kalman filter
            W += w
            bias = jnp.average(jnp.square((F2 - self.Target.variance) / self.Target.variance))

            return ((x, u, l, g, E, Feps, Weps, eps_max, key, time), (W, F2)), bias



        _, b = jax.lax.scan(step, init= ((x, u, l, g, 0.0, jnp.power(eps, -6.0) * 1e-5, 1e-5, jnp.inf, random_key, 0.0),
                                                 (eps, jnp.square(self.Target.transform(x)))),
                                    xs=None, length=num_steps)

        return b  # + nans * 1e5 #return a large bias if there were nans


    ### tuning phase: ###

    def tune12(self, x, u, l, g, random_key, L_given, eps, sigma_given, num_steps1, num_steps2):
        """cheap hyperparameter tuning"""

        # during the tuning we will be using a different gamma
        gamma_save = self.gamma # save the old value
        neff = 150.0
        self.gamma = (neff - 1)/(neff + 1.0)
        sigma = sigma_given

        def step(state, outer_weight):
            """one adaptive step of the dynamics"""
            x, u, l, g, E, Feps, Weps, eps_max, key, time, eps = self.dynamics_adaptive(state[0], L, sigma)
            W, F1, F2 = state[1]
            w = outer_weight * eps
            zero_prevention = 1-outer_weight
            F1 = (W*F1 + w*x) / (W + w + zero_prevention)  # Update <f(x)> with a Kalman filter
            F2 = (W*F2 + w*jnp.square(x)) / (W + w + zero_prevention)  # Update <f(x)> with a Kalman filter
            W += w

            return ((x, u, l, g, E, Feps, Weps, eps_max, key, time), (W, F1, F2)), eps

        L = L_given

        # we use the last num_steps2 to compute the diagonal preconditioner
        outer_weights = jnp.concatenate((jnp.zeros(num_steps1), jnp.ones(num_steps2)))

        #initial state
        state = ((x, u, l, g, 0.0, jnp.power(eps, -6.0) * 1e-5, 1e-5, jnp.inf, random_key, 0.0), (0.0, jnp.zeros(len(x)), jnp.zeros(len(x))))

        # run the steps
        state, eps = jax.lax.scan(step, init=state, xs= outer_weights, length= num_steps1 + num_steps2)

        # determine L
        F1, F2 = state[1][1], state[1][2]
        variances = F2 - jnp.square(F1)
        sigma2 = jnp.average(variances)
        # print(F1)
        # print(F2 / self.Target.second_moments)

        #variances = self.Target.second_moments
        #resc = jnp.diag(1.0/jnp.sqrt(variances))
        #Sigma = resc @ self.Target.Cov @ resc
        #print(jnp.linalg.cond(Sigma) / jnp.linalg.cond(self.Target.Cov))

        # optionally we do the diagonal preconditioning (and readjust the stepsize)
        if self.diagonal_preconditioning:

            # diagonal preconditioning
            sigma = jnp.sqrt(variances)
            L = jnp.sqrt(self.Target.d)

            # state = ((state[0][0], state[0][1], state[0][2], state[0][3], 0.0, jnp.power(eps[-1], -6.0) * 1e-5, 1e-5, jnp.inf, state[0][-2], 0.0),
            #         (0.0, jnp.zeros(len(x)), jnp.zeros(len(x))))

            # print(L, eps[-1])
            # print(sigma**2 / self.Target.variance)

            #readjust the stepsize
            steps = num_steps2 // 3 #we do some small number of steps
            state, eps = jax.lax.scan(step, init= state, xs= jnp.ones(steps), length= steps)

        else:
            L = jnp.sqrt(sigma2 * self.Target.d)
        #print(L, eps[-1])
        xx, uu, ll, gg, key = state[0][0], state[0][1], state[0][2], state[0][3], state[0][-2] # the final state
        self.gamma = gamma_save #set gamma to the previous value
        #print(L, eps[-1])
        return L, eps[-1], sigma, xx, uu, ll, gg, key #return the tuned hyperparameters and the final state



    def tune3(self, x, u, l, g, random_key, L, eps, sigma, num_steps):
        """determine L by the autocorrelations (around 10 effective samples are needed for this to be accurate)"""

        X, xx, uu, ll, gg, key = self.sample_full(num_steps, x, u, l, g, random_key, L, eps, sigma)
        ESS = ess_corr(X)
        Lnew = self.Lfactor * eps / ESS # = 0.4 * correlation length

        return Lnew, xx, uu, ll, gg, key


    def sample_full(self, num_steps, x, u, l, g, random_key, L, eps, sigma):
        """Stores full x for each step. Used in tune2."""

        def step(state, useless):
            x, u, l, g, E, key,   time = state
            xx, uu, ll, gg, kinetic_change, key, time = self.dynamics(x, u, g, key, time, L, eps, sigma)
            EE = E + kinetic_change + ll - l
            return (xx, uu, ll, gg, EE, key, time), xx

        state, track = jax.lax.scan(step, init=(x, u, l, g, 0.0, random_key, 0.0), xs=None, length=num_steps)
        xx, uu, ll, gg, key = state[0], state[1], state[2], state[3], state[5]
        return track, xx, uu, ll, gg, key




def find_crossing(array, cutoff):
    """the smallest M such that array[m] < cutoff for all m > M"""

    def step(carry, element):
        """carry = (, 1 if (array[i] > cutoff for all i < current index) else 0"""
        above_threshold = element > cutoff
        never_been_below = carry[1] * above_threshold  #1 if (array[i] > cutoff for all i < current index) else 0
        return (carry[0] + never_been_below, never_been_below), above_threshold

    state, track = jax.lax.scan(step, init=(0, 1), xs=array, length=len(array))

    return state[0]
    #return jnp.sum(track) #total number of indices for which array[m] < cutoff



def point_reduction(num_points, reduction_factor):
    """reduces the number of points for plotting purposes"""

    indexes = np.concatenate((np.arange(1, 1 + num_points // reduction_factor, dtype=int),
                              np.arange(1 + num_points // reduction_factor, num_points, reduction_factor, dtype=int)))
    return indexes



def burn_in_ending(loss):
    loss_avg = jnp.median(loss[len(loss)//2:])
    return 2 * find_crossing(loss - loss_avg, 0.0) #we add a safety factor of 2

    ### plot the removal ###
    # t= np.arange(len(loss))
    # plt.plot(t[:i*2], loss[:i*2], color= 'tab:red')
    # plt.plot(t[i*2:], loss[i*2:], color= 'tab:blue')
    # plt.yscale('log')
    # plt.show()



def my_while(cond_fun, body_fun, initial_state):
    """see https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html"""

    state = initial_state

    while cond_fun(state):
        state = body_fun(state)

    return state


def ess_corr(x):
    """Taken from: https://blackjax-devs.github.io/blackjax/diagnostics.html
        shape(x) = (num_samples, d)"""

    input_array = jnp.array([x, ])

    num_chains = 1#input_array.shape[0]
    num_samples = input_array.shape[1]

    mean_across_chain = input_array.mean(axis=1, keepdims=True)
    # Compute autocovariance estimates for every lag for the input array using FFT.
    centered_array = input_array - mean_across_chain
    m = next_fast_len(2 * num_samples)
    ifft_ary = jnp.fft.rfft(centered_array, n=m, axis=1)
    ifft_ary *= jnp.conjugate(ifft_ary)
    autocov_value = jnp.fft.irfft(ifft_ary, n=m, axis=1)
    autocov_value = (
        jnp.take(autocov_value, jnp.arange(num_samples), axis=1) / num_samples
    )
    mean_autocov_var = autocov_value.mean(0, keepdims=True)
    mean_var0 = (jnp.take(mean_autocov_var, jnp.array([0]), axis=1) * num_samples / (num_samples - 1.0))
    weighted_var = mean_var0 * (num_samples - 1.0) / num_samples
    weighted_var = jax.lax.cond(
        num_chains > 1,
        lambda _: weighted_var+ mean_across_chain.var(axis=0, ddof=1, keepdims=True),
        lambda _: weighted_var,
        operand=None,
    )

    # Geyer's initial positive sequence
    num_samples_even = num_samples - num_samples % 2
    mean_autocov_var_tp1 = jnp.take(mean_autocov_var, jnp.arange(1, num_samples_even), axis=1)
    rho_hat = jnp.concatenate([jnp.ones_like(mean_var0), 1.0 - (mean_var0 - mean_autocov_var_tp1) / weighted_var,], axis=1,)

    rho_hat = jnp.moveaxis(rho_hat, 1, 0)
    rho_hat_even = rho_hat[0::2]
    rho_hat_odd = rho_hat[1::2]

    mask0 = (rho_hat_even + rho_hat_odd) > 0.0
    carry_cond = jnp.ones_like(mask0[0])
    max_t = jnp.zeros_like(mask0[0], dtype=int)

    def positive_sequence_body_fn(state, mask_t):
        t, carry_cond, max_t = state
        next_mask = carry_cond & mask_t
        next_max_t = jnp.where(next_mask, jnp.ones_like(max_t) * t, max_t)
        return (t + 1, next_mask, next_max_t), next_mask

    (*_, max_t_next), mask = jax.lax.scan(
        positive_sequence_body_fn, (0, carry_cond, max_t), mask0
    )
    indices = jnp.indices(max_t_next.shape)
    indices = tuple([max_t_next + 1] + [indices[i] for i in range(max_t_next.ndim)])
    rho_hat_odd = jnp.where(mask, rho_hat_odd, jnp.zeros_like(rho_hat_odd))
    # improve estimation
    mask_even = mask.at[indices].set(rho_hat_even[indices] > 0)
    rho_hat_even = jnp.where(mask_even, rho_hat_even, jnp.zeros_like(rho_hat_even))

    # Geyer's initial monotone sequence
    def monotone_sequence_body_fn(rho_hat_sum_tm1, rho_hat_sum_t):
        update_mask = rho_hat_sum_t > rho_hat_sum_tm1
        next_rho_hat_sum_t = jnp.where(update_mask, rho_hat_sum_tm1, rho_hat_sum_t)
        return next_rho_hat_sum_t, (update_mask, next_rho_hat_sum_t)

    rho_hat_sum = rho_hat_even + rho_hat_odd
    _, (update_mask, update_value) = jax.lax.scan(
        monotone_sequence_body_fn, rho_hat_sum[0], rho_hat_sum
    )

    rho_hat_even_final = jnp.where(update_mask, update_value / 2.0, rho_hat_even)
    rho_hat_odd_final = jnp.where(update_mask, update_value / 2.0, rho_hat_odd)

    # compute effective sample size
    ess_raw = num_chains * num_samples
    tau_hat = (-1.0
        + 2.0 * jnp.sum(rho_hat_even_final + rho_hat_odd_final, axis=0)
        - rho_hat_even_final[indices]
    )

    tau_hat = jnp.maximum(tau_hat, 1 / np.log10(ess_raw))
    ess = ess_raw / tau_hat

    ### my part (combine all dimensions): ###
    neff = ess.squeeze() / num_samples
    return 1.0 / jnp.average(1 / neff)


import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp



class vmap_target:
    """A wrapper target class, where jax.vmap has been applied to the functions of a given target"""

    def __init__(self, target):
        """target: a given target to vmap"""

        # obligatory attributes
        self.grad_nlogp = jax.vmap(target.grad_nlogp)
        self.d = target.d

        # optional attributes
        if hasattr(target, 'prior_draw'):
            self.prior_draw = jax.vmap(target.prior_draw)


class AnnealingSampler:
    """Ensamble MCHMC (q = 0 Hamiltonian) sampler"""

    def __init__(self, Target, shift_fn, alpha = 1.0, varE_wanted = 1e-4):
        """Args:
                Target: the target distribution class.
                alpha: the momentum decoherence scale L = alpha sqrt(d). Optimal alpha is typically around 1, but can also be 10 or so.
                varE_wanted: controls the stepsize after the burn-in. We aim for Var[E] / d = 'varE_wanted'.
        """

        self.Target = vmap_target(Target)

        self.shift_fn = shift_fn

        self.alpha = alpha
        self.L = jnp.sqrt(self.Target.d) * alpha
        self.varEwanted = varE_wanted

        self.grad_evals_per_step = 1.0 # per chain (leapfrog)

        self.eps_initial = jnp.sqrt(self.Target.d)    # this will be changed during the burn-in


    def random_unit_vector(self, random_key, num_chains):
        """Generates a random (isotropic) unit vector."""
        key, subkey = jax.random.split(random_key)
        u = jax.random.normal(subkey, shape = (num_chains, self.Target.d), dtype = 'float64')
        normed_u = u / jnp.sqrt(jnp.sum(jnp.square(u), axis = 1))[:, None]
        return normed_u, key


    def partially_refresh_momentum(self, u, random_key, nu):
        """Adds a small noise to u and normalizes."""
        key, subkey = jax.random.split(random_key)
        noise = nu * jax.random.normal(subkey, shape= u.shape, dtype=u.dtype)

        return (u + noise) / jnp.sqrt(jnp.sum(jnp.square(u + noise), axis = 1))[:, None], key



    def update_momentum(self, eps, g, u):
        """The momentum updating map of the esh dynamics (see https://arxiv.org/pdf/2111.02434.pdf)
        similar to the implementation: https://github.com/gregversteeg/esh_dynamics
        There are no exponentials e^delta, which prevents overflows when the gradient norm is large."""
        g_norm = jnp.sqrt(jnp.sum(jnp.square(g), axis=1)).T
        nonzero = g_norm > 1e-13  # if g_norm is zero (we are at the MAP solution) we also want to set e to zero and the function will return u
        inv_g_norm = jnp.nan_to_num(1.0 / g_norm) * nonzero
        e = - g * inv_g_norm[:, None]
        ue = jnp.sum(u * e, axis=1)
        delta = eps * g_norm / (self.Target.d - 1)
        zeta = jnp.exp(-delta)
        uu = e * ((1 - zeta) * (1 + zeta + ue * (1 - zeta)))[:, None] + 2 * zeta[:, None] * u
        delta_r = delta - jnp.log(2) + jnp.log(1 + ue + (1 - ue) * zeta ** 2)
        return uu / (jnp.sqrt(jnp.sum(jnp.square(uu), axis=1)).T)[:, None], delta_r


    def hamiltonian_dynamics(self, x, u, g, key, eps, T):
        """leapfrog"""

        # half step in momentum
        uu, delta_r1 = self.update_momentum(eps * 0.5, g / T, u)

        # full step in x
        xx = self.shift_fn(x, eps * uu)
        l, gg = self.Target.grad_nlogp(xx)

        # half step in momentum
        uu, delta_r2 = self.update_momentum(eps * 0.5, gg / T, uu)
        kinetic_change = (delta_r1 + delta_r2) * (self.Target.d-1)

        return xx, uu, l, gg, kinetic_change, key


    def dynamics(self, x, u, g, random_key, L, eps, T):
        """One step of the generalized dynamics."""

        # Hamiltonian step
        xx, uu, ll, gg, kinetic_change, key = self.hamiltonian_dynamics(x, u, g, random_key, eps, T)

        # bounce
        nu = jnp.sqrt((jnp.exp(2 * eps / L) - 1.0) / self.Target.d)
        uu, key = self.partially_refresh_momentum(uu, key, nu)

        return xx, uu, ll, gg, kinetic_change, key


    def initialize(self, random_key, x_initial, num_chains):


        if random_key is None:
            key = jax.random.PRNGKey(0)
        else:
            key = random_key

        if isinstance(x_initial, str):
            if x_initial == 'prior':  # draw the initial x from the prior
                keys_all = jax.random.split(key, num_chains + 1)
                x = self.Target.prior_draw(keys_all[1:])
                key = keys_all[0]

            else:  # if not 'prior' the x_initial should specify the initial condition
                raise KeyError('x_initial = "' + x_initial + '" is not a valid argument. \nIf you want to draw initial condition from a prior use x_initial = "prior", otherwise specify the initial condition with an array')

        else:  # initial x is given
            x = jnp.copy(x_initial)

        l, g = self.Target.grad_nlogp(x)


        ### initial velocity ###
        u, key = self.random_unit_vector(key, num_chains)  # random velocity orientations


        return x, u, l, g, key



    def sample_temp_level(self, num_steps, tune_steps, x0, u0, l0, g0, key0, L0, eps0, T):


        def step(state, tune):

            x, u, l, g, key, L, eps = state

            x, u, ll, g, kinetic_change, key = self.dynamics(x, u, g, key, L, eps, T)  # update particles by one step


            ### stepsize tuning ###
            de = jnp.square(kinetic_change + ll - l) / self.Target.d #square energy error per dimension
            varE = jnp.average(de) #averaged over the ensamble

                                #if we are in the tuning phase            #else
            eps *= (tune * jnp.power(varE / self.varEwanted, -1./6.) + (1-tune))

            ### L tuning ###
            moment1 = jnp.average(x, axis=0)
            moment2 = jnp.average(jnp.square(x), axis = 0)
            var= moment2 - jnp.square(moment1)
            sig = jnp.sqrt(jnp.average(var)) # average over dimensions (= typical width of the posterior)
            Lnew = self.alpha * sig * jnp.sqrt(self.Target.d)
            L = tune * Lnew + (1-tune) * L #update L if we are in the tuning phase


            return (x, u, l, g, key, L, eps), None


                                    #tune in the first 1/3 of time        #stop tuning
        tune_schedule = jnp.concatenate((jnp.ones(tune_steps), jnp.zeros(num_steps - tune_steps)))

        return jax.lax.scan(step, init= (x0, u0, l0, g0, key0, L0, eps0), xs= tune_schedule, length= num_steps)[0]



    def sample(self, steps_at_each_temp, tune_steps, num_chains, temp_schedule, x_initial= 'prior', random_key= None):

        x0, u0, l0, g0, key0 = self.initialize(random_key, x_initial, num_chains) #initialize the chains

        temp_schedule_ext = jnp.insert(temp_schedule, 0, temp_schedule[0]) # as if the temp level before the first temp level was the same


        def temp_level(state, iter):
            x, u, l, g, key, L, eps = state

            T, Tprev = temp_schedule_ext[iter], temp_schedule_ext[iter-1]

            L *= jnp.sqrt(T / Tprev)
            eps *= jnp.sqrt(T / Tprev)

            state = self.sample_temp_level(steps_at_each_temp, tune_steps, x, u, l, g, key, L, eps, T)

            return state, None


        state = jax.lax.scan(temp_level, init= (x0, u0, l0, g0, key0, self.L, self.eps_initial), xs= jnp.arange(1, len(temp_schedule_ext)))[0]

        return state[0] # final x




