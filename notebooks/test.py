from moleculekit.molecule import Molecule
import jax_md
import os
from parameters import Parameters, set_box, set_positions
import jax.numpy as nnp
from torchmd.forcefields.forcefield import ForceField
import numpy.linalg as npl
import seaborn as sns
import matplotlib.pyplot as plt 
from plotting import rama_plot
# from jax_md.simulate import Sampler
# from jax_md.old_sequential_sampler import Sampler as OldSampler
from jax_md import space, quantity
# from jax_md import simulate, energy
# from jax_md.simulate import ess_corr
from sampling.sampler import Sampler
from sampling.old_sampler import Sampler as OldSampler
from sampling.correlation_length import ess_corr
import math
import jax
import scipy
import mdtraj as md
from sampling.old_annealing import Sampler as OAS
from sampling.annealing import Sampler as AS


# load alanine dipeptide
testdir = "data/prod_alanine_dipeptide_amber/"
mol = Molecule(os.path.join(testdir, "structure.prmtop"))  # Reading the system topology
mol.read(os.path.join(testdir, "input.coor"))  # Reading the initial simulation coordinates
mol.read(os.path.join(testdir, "input.xsc"))  # Reading the box dimensions

ff = ForceField.create(mol, os.path.join(testdir, "structure.prmtop"))
parameters = Parameters(ff, mol, precision=float, device='cpu')
nreplicas = 1 # don't change
pos = set_positions(nreplicas, mol.coords)
box = nnp.array(set_box(nreplicas, mol.box), dtype='float32')

from forces import Forces
forces = Forces(parameters, cutoff=9, rfa=True, switch_dist=7.5, terms=["bonds", "angles", "dihedrals", "impropers", "1-4", "electrostatics", "lj"])
forces.compute(pos, box)

psi_indices, phi_indices = [6, 8, 14, 16], [4, 6, 8, 14]




BOLTZMAN = 0.001987191

displacement_fn, shift_fn = space.periodic(box[0][0][0].item())

nlogp = (lambda x : forces.compute(nnp.reshape(x, pos.shape), box))

def run_sequential(T, dt, L_factor, chain_length, old):
    
    energy_fn = lambda x : nlogp(x) / (BOLTZMAN * 300)
    value_grad = jax.value_and_grad(energy_fn)

    class MD():


        def __init__(self, d):
            self.d = d
            self.nbrs = None

        def grad_nlogp(self, x):
            return value_grad(x)

        def transform(self, x):
            return x

        def prior_draw(self, key):
            return nnp.array(nnp.reshape(pos, math.prod(pos.shape)), dtype='float64')

    print(f'T={T}, dt={dt}, and with {chain_length} steps ')

    eps_in_si = dt*scipy.constants.femto * nnp.sqrt(3 * 688 * scipy.constants.k * T)
    si_to_gmol = nnp.sqrt(1000*scipy.constants.Avogadro)/scipy.constants.angstrom
    eps = eps_in_si * si_to_gmol

    target = MD(d = math.prod(pos.shape))
    if old:
        sampler = OldSampler(target, frac_tune1=0.0, frac_tune2=0.0, frac_tune3=0.0, L = L_factor*eps,
                  eps=eps)    
    else:
        sampler = Sampler(target, shift_fn=shift_fn , frac_tune1=0.0, frac_tune2=0.0, frac_tune3=0.0, L = L_factor*eps,
                  eps=eps)
        # jax.numpy.tile(mol.masses,3)
    num_chains = 1
    samples, energy, L, _ = sampler.sample(chain_length, num_chains, output= 'detailed', random_key=jax.random.PRNGKey(0))


    print("MCLMC\n\n")

    print("L: ", L)
    print("eps: ", eps)

    rmses = nnp.sqrt(nnp.mean(samples**2, axis=0))
    print("Mean RMS: ", rmses.mean())
    print("Max RMS: ", nnp.max(rmses))
    print("Min RMS: ", nnp.min(rmses))
    print("Energy error: ", (nnp.square(energy[1:]-energy[:-1])/math.prod(pos.shape)).mean())

    print("ESS (via ess_corr): ", ess_corr(samples))


    # name = 'mclmc' + str(eps) + str(L) + str(num_chains)
    # trajectory.save_pdb('./data/prod_alanine_dipeptide_amber/traj'+name+'.pdb')

    return samples,energy, L, eps


def run_annealing(old):

    # eps_in_si = dt*scipy.constants.femto * nnp.sqrt(3 * 688 * scipy.constants.k * T)
    # si_to_gmol = nnp.sqrt(1000*scipy.constants.Avogadro)/scipy.constants.angstrom
    # eps = eps_in_si * si_to_gmol

    energy_fn = lambda x : nlogp(x)
    value_grad = jax.value_and_grad(energy_fn)

    class MD():


        def __init__(self, d):
            self.d = d
            self.nbrs = None

        def grad_nlogp(self, x):
            return value_grad(x)

        def transform(self, x):
            return x

        def prior_draw(self, key):
            return nnp.array(nnp.reshape(pos, math.prod(pos.shape)), dtype='float64')

    target = MD(d = math.prod(pos.shape))
    # sampler = OAS(target, shift_fn=shift_fn)
    if old:
        sampler = OAS(target, shift_fn=shift_fn) # , masses=jax.numpy.tile(mol.masses,3))
    else: sampler = AS(target, shift_fn=shift_fn) # masses=jax.numpy.tile(mol.masses,3))

    def temp_func(T,Tprev, L, eps):

        dt = 2
        eps_in_si = dt*scipy.constants.femto * nnp.sqrt(3 * 688 * scipy.constants.k * (T/BOLTZMAN))
        si_to_gmol = nnp.sqrt(1000*scipy.constants.Avogadro)/scipy.constants.angstrom
        eps = eps_in_si * si_to_gmol

        return eps*30, eps
    
    sampler.temp_func = temp_func

    
    if not old:
        samples, energy = sampler.sample(steps_at_each_temp=10, temp_schedule=nnp.array([300.0*BOLTZMAN]), num_chains=10, tune_steps=0, random_key=jax.random.PRNGKey(0))
    else:
        samples = sampler.sample(steps_at_each_temp=10, temp_schedule=nnp.array([300.0*BOLTZMAN]), num_chains=10, tune_steps=0, random_key=jax.random.PRNGKey(0))
        energy = 0
    # , x_initial=x_initial[::100])   

    print("shape\n\n\n", samples.shape)

    # subsampled = nnp.reshape(samples[0], (10, 2064))
    # print(subsampled.shape, "shape")
    # trajectory = md.load('./data/prod_alanine_dipeptide_amber/structure.pdb')
    # trajectory.xyz=nnp.array(nnp.reshape(subsampled, (subsampled.shape[0], 688, 3)))[::1]
    # unitC = nnp.array([(subsampled.shape[0])*[nnp.diag(mol.box[:,0])]]).squeeze()
    # trajectory.unitcell_vectors = unitC # traj.unitcell_vectors[:10000]
    # angles = md.compute_dihedrals(trajectory, [phi_indices, psi_indices])

    
    return samples, energy


def run_smc(old, x_initial):

    # eps_in_si = dt*scipy.constants.femto * nnp.sqrt(3 * 688 * scipy.constants.k * T)
    # si_to_gmol = nnp.sqrt(1000*scipy.constants.Avogadro)/scipy.constants.angstrom
    # eps = eps_in_si * si_to_gmol

    energy_fn = lambda x : nlogp(x)
    value_grad = jax.value_and_grad(energy_fn)

    class MD():


        def __init__(self, d):
            self.d = d
            self.nbrs = None

        def grad_nlogp(self, x):
            return value_grad(x)

        def transform(self, x):
            return x

        def prior_draw(self, key):
            return nnp.array(nnp.reshape(pos, math.prod(pos.shape)), dtype='float64')

    target = MD(d = math.prod(pos.shape))
    # sampler = OAS(target, shift_fn=shift_fn)
    if old:
        sampler = OAS(target, shift_fn=shift_fn) # , masses=jax.numpy.tile(mol.masses,3))
    else: sampler = AS(target, shift_fn=shift_fn) # masses=jax.numpy.tile(mol.masses,3))

    def temp_func(T,Tprev, L, eps):

        dt = 2
        eps_in_si = dt*scipy.constants.femto * nnp.sqrt(3 * 688 * scipy.constants.k * (T/BOLTZMAN))
        si_to_gmol = nnp.sqrt(1000*scipy.constants.Avogadro)/scipy.constants.angstrom
        eps = eps_in_si * si_to_gmol

        return eps*30, eps
    
    sampler.temp_func = temp_func

    
    if not old:
        samples, energy = sampler.sample(steps_at_each_temp=10, temp_schedule=(4000.0*BOLTZMAN,300.0*BOLTZMAN), num_chains=10, tune_steps=0, random_key=jax.random.PRNGKey(0), x_initial=x_initial)
    else:
        samples = sampler.sample(steps_at_each_temp=10, temp_schedule=nnp.array([300.0*BOLTZMAN]), num_chains=10, tune_steps=0, random_key=jax.random.PRNGKey(0))
        energy = 0
    # , x_initial=x_initial[::100])   

    print("shape\n\n\n", samples.shape)

    # subsampled = nnp.reshape(samples[0], (10, 2064))
    # print(subsampled.shape, "shape")
    # trajectory = md.load('./data/prod_alanine_dipeptide_amber/structure.pdb')
    # trajectory.xyz=nnp.array(nnp.reshape(subsampled, (subsampled.shape[0], 688, 3)))[::1]
    # unitC = nnp.array([(subsampled.shape[0])*[nnp.diag(mol.box[:,0])]]).squeeze()
    # trajectory.unitcell_vectors = unitC # traj.unitcell_vectors[:10000]
    # angles = md.compute_dihedrals(trajectory, [phi_indices, psi_indices])

    
    return samples, energy

# Energy error:  0.10286242477917347
x_initial,energy,L,eps = run_sequential(T=4000, dt=2, L_factor=30, chain_length=10, old=False)
# print(f"mean of samples: {samples.mean()}")
# print(f"energy of shape {energy.shape}")
# print(f"energy: {(nnp.square(energy[1:]-energy[:-1])/2064).mean()}")
# samples,energy,L,eps = run_sequential(T=4000, dt=2, L_factor=30, chain_length=10, old=True)
# print("Sequential, old:\n\n")
# print(f"mean of samples: {samples.mean()}")
# print(f"energy: {(nnp.square(energy[1:]-energy[:-1])/2064).mean()}")




# print("Annealing\n\n")
# samples, energy = run_annealing(old=False)
# # print(f"energy of shape {energy[0,:,0].shape}")
# energy_error = (nnp.square(energy[:, 1:, :]-energy[:, :-1, :])/2064).mean()
# # print(f"error: {energy_error}")
# # print(f"mean of samples: {samples.mean()}")
# # print("Annealing, old:\n\n")
# # samples, energy = run_annealing(old=True)
# # print(f"mean of samples: {samples.mean()}")


print(x_initial.shape)
samples, energy = run_smc(old=False, x_initial=x_initial)