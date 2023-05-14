# port of the TorchMD code to load up an amber/charmm potential

from scipy import constants as const
# import torch
import numpy as np
import numpy.linalg as npl 
import jax.numpy as nnp
import jax.numpy.linalg as nnpl
import jax.lax as lnp
import jax
from math import pi


class Forces:
    """
    Parameters
    ----------
    cutoff : float
        If set to a value it will only calculate LJ, electrostatics and bond energies for atoms which are closer
        than the threshold
    rfa : bool
        Use with `cutoff` to enable the reaction field approximation for scaling of the electrostatics up to the cutoff.
        Uses the value of `solventDielectric` to model everything beyond the cutoff distance as solvent with uniform
        dielectric.
    solventDielectric : float
        Used together with `cutoff` and `rfa`
    """

    # 1-4 is nonbonded but we put it currently in bonded to not calculate all distances
    bonded = ["bonds", "angles", "dihedrals", "impropers", "1-4"]
    nonbonded = ["electrostatics", "lj", "repulsion", "repulsioncg"]
    terms = bonded + nonbonded

    def __init__(
        self,
        parameters,
        terms=None,
        external=None,
        cutoff=None,
        rfa=False,
        solventDielectric=78.5,
        switch_dist=None,
        exclusions=("bonds", "angles", "1-4"),
    ):
        self.par = parameters
        if terms is None:
            raise RuntimeError(
                'Set force terms or leave empty brackets [].\nAvailable options: "bonds", "angles", "dihedrals", "impropers", "1-4", "electrostatics", "lj", "repulsion", "repulsioncg".'
            )

        self.energies = [ene.lower() for ene in terms]
        for et in self.energies:
            if et not in Forces.terms:
                raise ValueError(f"Force term {et} is not implemented.")

        if "1-4" in self.energies and "dihedrals" not in self.energies:
            raise RuntimeError(
                "You cannot enable 1-4 interactions without enabling dihedrals"
            )

        self.natoms = len(parameters.masses)
        self.require_distances = any(f in self.nonbonded for f in self.energies)
        self.ava_idx = (
            self._make_indeces(
                self.natoms, parameters.get_exclusions(exclusions), parameters.device
            )
            if self.require_distances
            else None
        )
        self.external = external
        self.cutoff = cutoff
        self.rfa = rfa
        self.solventDielectric = solventDielectric
        self.switch_dist = switch_dist

    def _filter_by_cutoff(self, dist, arrays):
        under_cutoff = dist <= self.cutoff
        # indexedarrays = []
        # print(arrays)
        # print(under_cutoff.shape)

        # return jax.lax.cond(len(arrays)==4, 
        #     lambda arr : (arr[0][under_cutoff], arr[1][under_cutoff], arr[2][under_cutoff], arr[3][under_cutoff]) ,  
        #     lambda arr : (arr[0][under_cutoff], arr[1][under_cutoff], arr[2][under_cutoff]) ,  
            
        #     arrays
        #     )

        # todo: maybe inefficient
        return jax.tree_map(lambda arr : (nnp.where(under_cutoff, arr.T, 0.0).T).astype(arr.dtype), arrays)
        # return jax.tree_map(lambda arr : arr[under_cutoff], arrays)

        

        # if len(arrays) == 4:
        #     (a, b, c, d) = arrays
        #     return (
        #             a[under_cutoff], 
        #             b[under_cutoff], 
        #             c[under_cutoff], 
        #             d[under_cutoff]
        #             )
        # elif len(arrays) == 3:
        #     (a,b,c) = arrays
        #     return (
        #             a[under_cutoff], 
        #             b[under_cutoff], 
        #             c[under_cutoff] 
        #             )
        # return jax.vmap(lambda x : x[under_cutoff])(arrays)
        # for arr in arrays:
        #     indexedarrays.append(arr[under_cutoff])
        # return indexedarrays

    def compute(self, pos, box):



        nsystems = pos.shape[0]
        # if nnp.any(np.isnan(pos)):
        #     raise RuntimeError("Found NaN coordinates.")

        # pot = []
        # for i in range(nsystems):
        #     pp = {
        #         v: np.zeros((1,))
        #         for v in self.energies
        #     }
        #     pp["external"] = np.zeros((1, ))
        #     pot.append(pp)

        


        
        spos = pos[0]
        sbox = nnp.diagonal(box[0]) # box[0][np.array(nnp.eye(3), dtype=bool)]  # Use only the diagonal

        # Bonded terms
        # TODO: We are for sure doing duplicate distance calculations here!
        if "bonds" in self.energies and self.par.bonds is not None:
            bond_dist, bond_unitvec, _ = calculate_distances(
                spos, self.par.bonds, sbox
            )
            pairs = self.par.bonds
            bond_params = self.par.bond_params
            if self.cutoff is not None:
                (
                    bond_dist,
                    bond_unitvec,
                    pairs,
                    bond_params,
                ) = self._filter_by_cutoff(
                    bond_dist, (bond_dist, bond_unitvec, pairs, bond_params)
                )
            # print(bond_dist.shape, bond_params.shape, "shapes")
            E = evaluate_bonds(bond_dist, bond_params)
            
            pot_bonds = E.sum()
        else:
            pot_bonds = 0.0
            # pot[0]["bonds"] += E.sum()

        

            # return [nnp.sum(nnp.concatenate(list(pp.values()))) for pp in pot]
            
        if "angles" in self.energies and self.par.angles is not None:
            _, _, r21 = calculate_distances(spos, self.par.angles[:, [0, 1]], sbox)
            _, _, r23 = calculate_distances(spos, self.par.angles[:, [2, 1]], sbox)
            E = evaluate_angles(
                r21, r23, self.par.angle_params
            )

            pot_angles = E.sum()
        else:
            pot_angles = 0.0
            
        if "dihedrals" in self.energies and self.par.dihedrals is not None:
            _, _, r12 = calculate_distances(
                spos, self.par.dihedrals[:, [0, 1]], sbox
            )
            _, _, r23 = calculate_distances(
                spos, self.par.dihedrals[:, [1, 2]], sbox
            )
            _, _, r34 = calculate_distances(
                spos, self.par.dihedrals[:, [2, 3]], sbox
            )
            E = evaluate_torsion(
                r12, r23, r34, self.par.dihedral_params
            )

            pot_tor = E.sum()
        else:
            pot_tor = 0.0
            

        if "1-4" in self.energies and self.par.idx14 is not None:
            nb_dist, nb_unitvec, _ = calculate_distances(spos, self.par.idx14, sbox)

            nonbonded_14_params = self.par.nonbonded_14_params
            idx14 = self.par.idx14


            aa = nonbonded_14_params[:, 0]
            bb = nonbonded_14_params[:, 1]
            scnb = nonbonded_14_params[:, 2]
            scee = nonbonded_14_params[:, 3]

            if "lj" in self.energies:
                E = evaluate_LJ_internal(
                    nb_dist, aa, bb, scnb, None, None
                )
                pot_lj = nnp.nansum(E)
            else:
                pot_lj = 0.0

            # return pot_lj

            


            if "electrostatics" in self.energies:
                pot_electrostatics_unsummed = evaluate_electrostatics(
                    nb_dist,
                    idx14,
                    self.par.charges,
                    scee,
                    cutoff=None,
                    rfa=False,
                    solventDielectric=self.solventDielectric
                )
                pot_electrostatics = pot_electrostatics_unsummed.sum()
            else:
                pot_electrostatics = 0.0


        if "impropers" in self.energies and self.par.impropers is not None:
            _, _, r12 = calculate_distances(
                spos, self.par.impropers[:, [0, 1]], sbox
            )
            _, _, r23 = calculate_distances(
                spos, self.par.impropers[:, [1, 2]], sbox
            )
            _, _, r34 = calculate_distances(
                spos, self.par.impropers[:, [2, 3]], sbox
            )
            # print("R34", r34.shape, r23.shape, r12.shape)
            E = evaluate_torsion(
                r12, r23, r34, self.par.improper_params
            )

            pot_impropers = E.sum()
        else:
            pot_impropers = 0.0



        # Non-bonded terms
        if self.require_distances and len(self.ava_idx):
            # Lazy mode: Do all vs all distances
            # TODO: These distance calculations are fucked once we do neighbourlists since they will vary per system!!!!
            nb_dist, nb_unitvec, _ = calculate_distances(spos, self.ava_idx, sbox)
            ava_idx = self.ava_idx
            if self.cutoff is not None:
                nb_dist, nb_unitvec, ava_idx = self._filter_by_cutoff(
                    nb_dist, (nb_dist, nb_unitvec, ava_idx)
                )

                if 'electrostatics' in self.energies:
                    pot_electrostatics2_unsummed = evaluate_electrostatics(
                        nb_dist,
                        ava_idx,
                        self.par.charges,
                        cutoff=self.cutoff,
                        rfa=self.rfa,
                        solventDielectric=self.solventDielectric
                    )
                    pot_electrostatics2 = infsum(pot_electrostatics2_unsummed)
                else: pot_electrostatics2 = 0.0


                if 'lj' in self.energies:
                    pot_lj2_unsummed = evaluate_LJ(
                        nb_dist,
                        ava_idx,
                        self.par.mapped_atom_types,
                        self.par.A,
                        self.par.B,
                        self.switch_dist,
                        self.cutoff
                    )
                    pot_lj2 = nnp.nansum(pot_lj2_unsummed)
                else:
                    pot_lj2 = 0.0




                if 'repulsion' in self.energies:
                    pot_repulsion_unsummed = evaluate_repulsion(
                        nb_dist,
                        ava_idx,
                        self.par.mapped_atom_types,
                        self.par.A
                    )
                    pot_repulsion = pot_repulsion_unsummed.sum()
                else:
                    pot_repulsion = 0.0
                
                if 'repulsioncg' in self.energies:
                    pot_repulsioncg_unsummed = evaluate_repulsion_CG(
                        nb_dist,
                        ava_idx,
                        self.par.mapped_atom_types,
                        self.par.B
                    )
                    pot_repulsioncg = pot_repulsioncg_unsummed.sum()
                else:
                    pot_repulsioncg = 0.0

                



        if self.external:
            ext_ene, ext_force = self.external.calculate(pos, box)
            pot_external = ext_ene[0]
        else:
            pot_external = 0


        # print(pot_bonds, "pot_bonds")
        # print(pot_angles, "pot_angles")
        # print(pot_tor, "pot_tor")
        # print(pot_lj, "pot_lj")
        # print(pot_electrostatics, "pot_electrostatics")
        # print(pot_impropers, "pot_impropers")
        # print(pot_electrostatics2, "pot_electrostatics2")
        # print(pot_lj2, "pot_lj2")
        # print(pot_repulsion, "pot_repulsion")
        # print(pot_repulsioncg, "pot_repulsioncg")
        # print(pot_external, "pot_external")
        # print(pot_electrostatics2, "pot_electrostatics2")

        pot = sum([
            pot_bonds, 
            pot_angles , 
            pot_tor, 
            pot_lj , 
            pot_electrostatics, 
            pot_impropers, 
            pot_electrostatics2, 
            pot_lj2,
            pot_repulsion, 
            pot_repulsioncg, 
            pot_external
        ])
        return pot
        # forces[:] = -jax.grad(
        #     enesum, pos)
        # return nnp.sum(nnp.concatenate(list(pot[0].values())))


    def _make_indeces(self, natoms, excludepairs, device):
        fullmat = nnp.full((natoms, natoms), True, dtype=bool)
        if len(excludepairs):
            excludepairs = nnp.array(excludepairs)
            # fullmat[excludepairs[:, 0], excludepairs[:, 1]] = False
            # fullmat[excludepairs[:, 1], excludepairs[:, 0]] = False
            fullmat = fullmat.at[excludepairs[:, 0], excludepairs[:, 1]].set(False)
            fullmat = fullmat.at[excludepairs[:, 1], excludepairs[:, 0]].set(False)
        fullmat = nnp.triu(fullmat, +1)
        allvsall_indeces = nnp.vstack(nnp.where(fullmat)).T
        ava_idx = nnp.array(allvsall_indeces)
        return ava_idx


def wrap_dist(dist, box):
    # return dist 
    return jax.lax.cond(box is None or nnp.all(box == 0),
             
        lambda d : d, 
        lambda d: d - nnp.expand_dims(box,0) * nnp.round(dist / nnp.expand_dims(box,0)),
        dist)
    # else:
    #     wdist = dist - nnp.expand_dims(box,0) * nnp.round(dist / nnp.expand_dims(box,0))
    # return wdist


def calculate_distances(atom_pos, atom_idx, box):
    
    # direction_vec = wrap_dist(atom_pos[atom_idx[:, 0]] - atom_pos[atom_idx[:, 1]], box)
    direction_vec = wrap_dist(nnp.take(atom_pos,atom_idx[:, 0], axis=0) - nnp.take(atom_pos, atom_idx[:, 1], axis=0), box)

    # print(direction_vec, "DIR VEC")
    # print(atom_pos, "ATOM POS")
    # print(atom_idx, "ATOM IDX")
    dist = nnpl.norm(direction_vec, axis=1)
    # print(dist, 'DIST')
    direction_unitvec = direction_vec / nnp.expand_dims(dist,1)
    return dist, direction_unitvec, direction_vec


ELEC_FACTOR = 1 / (4 * const.pi * const.epsilon_0)  # Coulomb's constant
ELEC_FACTOR *= const.elementary_charge**2  # Convert elementary charges to Coulombs
ELEC_FACTOR /= const.angstrom  # Convert Angstroms to meters
ELEC_FACTOR *= const.Avogadro / (const.kilo * const.calorie)  # Convert J to kcal/mol


def evaluate_LJ(
    dist, pair_indeces, atom_types, A, B, switch_dist, cutoff
):
    atomtype_indices = nnp.take(atom_types, pair_indeces)
    # print(" so far so good")
    aa = index_matrix(A, atomtype_indices[:, 0], atomtype_indices[:, 1])
    # print(aa.shape, atom_types.shape, A.shape, atomtype_indices[:, 1].shape)
    # aa = nnp.take(A, nnp.take(atom_types, pair_indeces))[:, :2]
    bb = index_matrix(B, atomtype_indices[:, 0], atomtype_indices[:, 1])
    
    # print("first")
    # print(aa, bb)
    # aa = A[atomtype_indices[:, 0], atomtype_indices[:, 1]]
    # bb = B[atomtype_indices[:, 0], atomtype_indices[:, 1]]
    # print("second")
    
    # print(aa, bb)
    # print(" so far so good")
    return evaluate_LJ_internal(dist, aa, bb, 1, switch_dist, cutoff)


# todo: gets this wrong
def evaluate_LJ_internal(
    dist, aa, bb, scale, switch_dist, cutoff
):
    
    # TODO REMOVE

    def denan(arr):
        out1 = nnp.where(nnp.isnan(arr), 0.0, arr)
        return nnp.where(nnp.isinf(out1), 0.0, out1)

    rinv1 = nnp.where(nnp.isinf(1 / dist), 0.0, 1/dist)

    rinv3 = denan(rinv1*rinv1*rinv1)
    rinv6 = denan(rinv3*rinv3)
    rinv12 = denan(rinv6 * rinv6)


    pot = ((aa * rinv12) - (bb * rinv6)) / scale
    # Switching function
    if switch_dist is not None and cutoff is not None:
        mask = dist > switch_dist
        
        t = ((nnp.where(mask, dist.T, 0.0).T) - switch_dist) / (cutoff - switch_dist)
        switch_val = 1 + t * t * t * (-10 + t * (15 - t * 6))
        pot = nnp.where(mask, pot*switch_val, pot)

        

    return pot


def evaluate_repulsion(
    dist, pair_indeces, atom_types, A, scale=1
):  # LJ without B
    force = None

    atomtype_indices = nnp.take(atom_types, pair_indeces)
    aa = A[atomtype_indices[:, 0], atomtype_indices[:, 1]]

    rinv1 = 1 / dist
    rinv6 = rinv1**6
    rinv12 = rinv6 * rinv6

    pot = (aa * rinv12) / scale
    return pot


def evaluate_repulsion_CG(
    dist, pair_indeces, atom_types, B, scale=1
):  # Repulsion like from CGNet
    force = None

    atomtype_indices = nnp.take(atom_types, pair_indeces)
    coef = B[atomtype_indices[:, 0], atomtype_indices[:, 1]]

    rinv1 = 1 / dist
    rinv6 = rinv1**6

    pot = (coef * rinv6) / scale
    return pot


def evaluate_electrostatics(
    dist,
    pair_indeces,
    atom_charges,
    scale=1,
    cutoff=None,
    rfa=False,
    solventDielectric=78.5
):
    
    # print("SHAPES")
    # print(pair_indeces.shape, "pair_indeces shape")
    # print(dist.shape, "dist shape")
    # print(atom_charges.shape, "atom_charges shape")
    # print(pair_indeces.shape, "pair_indeces shape")
    # print(pair_indeces.dtype)

    # print("force error")
    # print(pair_indeces[:,0])
    # print(atom_charges.shape, pair_indeces[:,0].shape)
    # print(nnp.take(atom_charges, pair_indeces[:,0]))
    # print("end force error")
    # force = None
    if rfa:  # Reaction field approximation for electrostatics with cutoff
        # http://docs.openmm.org/latest/userguide/theory.html#coulomb-interaction-with-cutoff
        # Ilario G. Tironi, René Sperb, Paul E. Smith, and Wilfred F. van Gunsteren. A generalized reaction field method
        # for molecular dynamics simulations. Journal of Chemical Physics, 102(13):5451–5459, 1995.
        denom = (2 * solventDielectric) + 1
        krf = (1 / cutoff**3) * (solventDielectric - 1) / denom
        crf = (1 / cutoff) * (3 * solventDielectric) / denom
        common = (
            ELEC_FACTOR
            * nnp.take(atom_charges, pair_indeces[:,0])
            * nnp.take(atom_charges, pair_indeces[:,1])
            / scale
        )
        dist2 = dist**2
        pot = common * ((1 / dist) + krf * dist2 - crf)
        
    else:
        pot = (
            ELEC_FACTOR
            * nnp.take(atom_charges, pair_indeces[:,0])
            * nnp.take(atom_charges, pair_indeces[:,1])
            / dist
            / scale
        )
       
    return pot


def evaluate_bonds(dist, bond_params):
    force = None

    k0 = bond_params[:, 0]
    d0 = bond_params[:, 1]
    x = dist - d0
    pot = k0 * (x**2)
    
    return pot


def evaluate_angles(r21, r23, angle_params):
    k0 = angle_params[:, 0]
    theta0 = angle_params[:, 1]

    dotprod = nnp.sum(r23 * r21, axis=1)
    norm23inv = 1 / nnpl.norm(r23, axis=1)
    norm21inv = 1 / nnpl.norm(r21, axis=1)

    cos_theta = dotprod * norm21inv * norm23inv
    cos_theta = nnp.clip(cos_theta, -1, 1)
    theta = nnp.arccos(cos_theta)

    delta_theta = theta - theta0
    pot = k0 * delta_theta * delta_theta

    
    return pot


def evaluate_torsion(r12, r23, r34, torsion_params):
    # Calculate dihedral angles from vectors
    crossA = nnp.cross(r12, r23, axis=1)
    crossB = nnp.cross(r23, r34, axis=1)
    crossC = nnp.cross(r23, crossA, axis=1)
    normA = nnpl.norm(crossA, axis=1)
    normB = nnpl.norm(crossB, axis=1)
    normC = nnpl.norm(crossC, axis=1)
    normcrossB = crossB / nnp.expand_dims(normB,1)
    cosPhi = nnp.sum(crossA * normcrossB, axis=1) / normA
    sinPhi = nnp.sum(crossC * normcrossB, axis=1) / normC
    phi = -nnp.arctan2(sinPhi, cosPhi)


    ntorsions = len(torsion_params[0]["idx"])
    pot = nnp.zeros(ntorsions, dtype=r12.dtype)

    # print((torsion_params[0]["idx"].shape), torsion_params[1]["idx"].shape, "len torsion params")
    

    
    # todo: reincorporate

    # print("length torsion params:", len(torsion_params))



    # print(idx.shape, pot.shape, (k0 * (1 + nnp.cos(angleDiff))).shape, "idx and pot shape")

    def amber(pot, idx, phi0, per, k0):

        angleDiff = per * phi[idx] - phi0
        pot = pot.at[idx].add(k0 * (1 + nnp.cos(angleDiff)))
        return pot

    def charmm(pot, idx, phi0, per, k0):
        # todo finish
        angleDiff = phi[idx] - phi0
        angleDiff = nnp.where(angleDiff < -pi, angleDiff, angleDiff + 2 * pi)
        angleDiff = nnp.where(angleDiff > -pi, angleDiff, angleDiff - 2 * pi)
        # angleDiff[angleDiff < -pi] = angleDiff[angleDiff < -pi] + 2 * pi
        # angleDiff[angleDiff > pi] = angleDiff[angleDiff > pi] - 2 * pi
        pot = pot.at[idx].add(k0 * angleDiff**2)
        return pot

    def part_of_torsion(tp):

        idx = tp["idx"]
        k0 = tp["params"][:, 0]
        phi0 = tp["params"][:, 1]
        per = tp["params"][:, 2]
    

        # print(nnp.all(per > 0), "bar")
        new_pot = jax.lax.cond(nnp.all(per > 0),
            lambda p : amber(p, idx, phi0, per, k0), 
            lambda p : charmm(p, idx, phi0, per, k0), pot)

        return new_pot

    return sum([
        (part_of_torsion)(tp) for tp in torsion_params
        ])

def index_matrix(mat,xi,yj):
    rows = nnp.take(mat,xi, axis=0)
    return jax.lax.scan(lambda i, row : (i+1, row[yj[i]]), 0, rows)[1]


def infsum(arr):
    return nnp.sum(nnp.where(nnp.isinf(arr), 0.0, arr ))
