import jax
from math import sqrt
import jax.numpy as nnp
import numpy as np

class Parameters:
    def __init__(
        self,
        ff,
        mol,
        terms=None,
        precision=float,
        device="cpu",
    ):
        self.A = None
        self.B = None
        self.bonds = None
        self.bond_params = None
        self.charges = None
        self.masses = None
        self.mapped_atom_types = None
        self.angles = None
        self.angle_params = None
        self.dihedrals = None
        self.dihedral_params = None
        self.idx14 = None
        self.nonbonded_14_params = None
        self.impropers = None
        self.improper_params = None

        self.natoms = mol.numAtoms
        if terms is None:
            terms = ("bonds", "angles", "dihedrals", "impropers", "1-4", "lj")
        terms = [term.lower() for term in terms]
        self.build_parameters(ff, mol, terms)
        # self.precision_(precision)
        # self.to_(device)

        self.device = device

    # def to_(self, device):
    #     self.charges = self.charges.to(device)
    #     self.masses = self.masses.to(device)
    #     if self.A is not None:
    #         self.A = self.A.to(device)
    #     if self.B is not None:
    #         self.B = self.B.to(device)
    #     if self.bonds is not None:
    #         self.bonds = self.bonds.to(device)
    #         self.bond_params = self.bond_params.to(device)
    #     if self.angles is not None:
    #         self.angles = self.angles.to(device)
    #         self.angle_params = self.angle_params.to(device)
    #     if self.dihedrals is not None:
    #         self.dihedrals = self.dihedrals.to(device)
    #         for j in range(len(self.dihedral_params)):
    #             termparams = self.dihedral_params[j]
    #             termparams["idx"] = termparams["idx"].to(device)
    #             termparams["params"] = termparams["params"].to(device)
    #     if self.idx14 is not None:
    #         self.idx14 = self.idx14.to(device)
    #         self.nonbonded_14_params = self.nonbonded_14_params.to(device)
    #     if self.impropers is not None:
    #         self.impropers = self.impropers.to(device)
    #         termparams = self.improper_params[0]
    #         termparams["idx"] = termparams["idx"].to(device)
    #         termparams["params"] = termparams["params"].to(device)
    #     if self.mapped_atom_types is not None:
    #         self.mapped_atom_types = self.mapped_atom_types.to(device)
    #     self.device = device

    # def precision_(self, precision):
    #     self.charges = self.charges.type(precision)
    #     self.masses = self.masses.type(precision)
    #     if self.A is not None:
    #         self.A = self.A.type(precision)
    #     if self.B is not None:
    #         self.B = self.B.type(precision)
    #     if self.bonds is not None:
    #         self.bond_params = self.bond_params.type(precision)
    #     if self.angles is not None:
    #         self.angle_params = self.angle_params.type(precision)
    #     if self.dihedrals is not None:
    #         for j in range(len(self.dihedral_params)):
    #             termparams = self.dihedral_params[j]
    #             termparams["params"] = termparams["params"].type(precision)
    #     if self.idx14 is not None:
    #         self.nonbonded_14_params = self.nonbonded_14_params.type(precision)
    #     if self.impropers is not None:
    #         termparams = self.improper_params[0]
    #         termparams["params"] = termparams["params"].type(precision)

    def get_exclusions(self, types=("bonds", "angles", "1-4"), fullarray=False):
        exclusions = []
        if self.bonds is not None and "bonds" in types:
            exclusions += self.bonds.tolist()
        if self.angles is not None and "angles" in types:
            npangles = self.angles
            exclusions += npangles[:, [0, 2]].tolist()
        if self.dihedrals is not None and "1-4" in types:
            # These exclusions will be covered by nonbonded_14_params
            npdihedrals = self.dihedrals
            exclusions += npdihedrals[:, [0, 3]].tolist()
        if fullarray:
            initial_fullmat = nnp.full((self.natoms, self.natoms), False, dtype=bool)
            if len(exclusions):
                exclusions = nnp.array(exclusions)
                fullmat = initial_fullmat.at[exclusions[:, 0], exclusions[:, 1]].set(True)
                fullmat = initial_fullmat.at[exclusions[:, 1], exclusions[:, 0]].set(True)
                exclusions = fullmat
        return exclusions

    def build_parameters(self, ff, mol, terms):
        uqatomtypes, indexes = np.unique(mol.atomtype, return_inverse=True)
        self.mapped_atom_types = np.array(indexes)
        # change point
        self.charges = np.array(mol.charge.astype(np.float64))
        self.masses = self.make_masses(ff, mol.atomtype)
        if "lj" in terms or "LJ" in terms:
            self.A, self.B = self.make_lj(ff, uqatomtypes)
        if "bonds" in terms and len(mol.bonds):
            uqbonds = np.unique([sorted(bb) for bb in mol.bonds], axis=0)
            self.bonds = np.array(uqbonds.astype(np.int64))
            self.bond_params = self.make_bonds(ff, uqatomtypes[indexes[uqbonds]])
        if "angles" in terms and len(mol.angles):
            uqangles = np.unique(
                [ang if ang[0] < ang[2] else ang[::-1] for ang in mol.angles], axis=0
            )
            self.angles = np.array(uqangles.astype(np.int64))
            self.angle_params = self.make_angles(ff, uqatomtypes[indexes[uqangles]])
        if "dihedrals" in terms and len(mol.dihedrals):
            uqdihedrals = np.unique(
                [dih if dih[0] < dih[3] else dih[::-1] for dih in mol.dihedrals], axis=0
            )
            self.dihedrals = np.array(uqdihedrals.astype(np.int64))
            self.dihedral_params = self.make_dihedrals(
                ff, uqatomtypes[indexes[uqdihedrals]]
            )
        if "1-4" in terms and len(mol.dihedrals):
            # Keep only dihedrals whos 1/4 atoms are not in bond+angle exclusions
            exclusions = self.get_exclusions(types=("bonds", "angles"), fullarray=True)
            keep = ~exclusions[uqdihedrals[:, 0], uqdihedrals[:, 3]]
            dih14 = uqdihedrals[keep, :]
            if len(dih14):
                # Remove duplicates (can occur if 1,4 atoms were same and 2,3 differed)
                uq14idx = np.unique(dih14[:, [0, 3]], axis=0, return_index=True)[1]
                dih14 = dih14[uq14idx]
                self.idx14 = np.array(dih14[:, [0, 3]].astype(np.int64))
                self.nonbonded_14_params = self.make_14(ff, uqatomtypes[indexes[dih14]])
        if "impropers" in terms and len(mol.impropers):
            uqimpropers = np.unique(mol.impropers, axis=0)
            # uqimpropers = self._unique_impropers(mol.impropers, mol.bonds)
            self.impropers = np.array(uqimpropers.astype(np.int64))
            self.improper_params = self.make_impropers(
                ff, uqatomtypes, indexes, uqimpropers, uqbonds
            )

    # def make_charges(self, ff, atomtypes):
    #     return np.array([ff.get_charge(at) for at in atomtypes])

    def make_masses(self, ff, atomtypes):
        masses = np.array([ff.get_mass(at) for at in atomtypes])
        masses = np.expand_dims(masses,1)  # natoms,1
        return masses

    def make_lj(self, ff, uqatomtypes):
        sigma = []
        epsilon = []
        for at in uqatomtypes:
            ss, ee = ff.get_LJ(at)
            sigma.append(ss)
            epsilon.append(ee)

        sigma = np.array(sigma, dtype=np.float64)
        epsilon = np.array(epsilon, dtype=np.float64)

        A, B = calculate_AB(sigma, epsilon)
        A = np.array(A)
        B = np.array(B)
        return A, B

    def make_bonds(self, ff, uqbondatomtypes):
        return np.array([ff.get_bond(*at) for at in uqbondatomtypes])

    def make_angles(self, ff, uqangleatomtypes):
        return np.array([ff.get_angle(*at) for at in uqangleatomtypes])

    def make_dihedrals(self, ff, uqdihedralatomtypes):
        from collections import defaultdict

        dihedrals = defaultdict(lambda: {"idx": [], "params": []})

        for i, at in enumerate(uqdihedralatomtypes):
            terms = ff.get_dihedral(*at)
            for j, term in enumerate(terms):
                dihedrals[j]["idx"].append(i)
                dihedrals[j]["params"].append(term)

        maxterms = max(dihedrals.keys()) + 1
        newdihedrals = []
        for j in range(maxterms):
            dihedrals[j]["idx"] = np.array(dihedrals[j]["idx"])
            dihedrals[j]["params"] = np.array(dihedrals[j]["params"])
            newdihedrals.append(dihedrals[j])

        return newdihedrals

    def make_impropers(self, ff, uqatomtypes, indexes, uqimpropers, bonds):
        impropers = {"idx": [], "params": []}
        graph = improper_graph(uqimpropers, bonds)

        for i, impr in enumerate(uqimpropers):
            at = uqatomtypes[indexes[impr]]
            try:
                params = ff.get_improper(*at)
            except:
                center = detect_improper_center(impr, graph)
                notcenter = sorted(np.setdiff1d(impr, center))
                order = [notcenter[0], notcenter[1], center, notcenter[2]]
                at = uqatomtypes[indexes[order]]
                params = ff.get_improper(*at)

            impropers["idx"].append(i)
            impropers["params"].append(params)

        impropers["idx"] = np.array(impropers["idx"])
        impropers["params"] = np.array(impropers["params"])
        return [impropers]

    def make_14(self, ff, uq14atomtypes):
        nonbonded_14_params = []
        for uqdih in uq14atomtypes:
            scnb, scee, lj1_s14, lj1_e14, lj4_s14, lj4_e14 = ff.get_14(*uqdih)
            # Lorentz - Berthelot combination rule
            sig = 0.5 * (lj1_s14 + lj4_s14)
            eps = sqrt(lj1_e14 * lj4_e14)
            s6 = sig**6
            s12 = s6 * s6
            A = eps * 4 * s12
            B = eps * 4 * s6
            nonbonded_14_params.append([A, B, scnb, scee])
        return np.array(nonbonded_14_params)


def calculate_AB(sigma, epsilon):
    # Lorentz - Berthelot combination rule
    sigma_table = 0.5 * (sigma + sigma[:, None])
    eps_table = np.sqrt(epsilon * epsilon[:, None])
    sigma_table_6 = sigma_table**6
    sigma_table_12 = sigma_table_6 * sigma_table_6
    A = eps_table * 4 * sigma_table_12
    B = eps_table * 4 * sigma_table_6
    del sigma_table_12, sigma_table_6, eps_table, sigma_table
    return A, B


def detect_improper_center(indexes, graph):
    for i in indexes:
        if len(np.intersect1d(list(graph.neighbors(i)), indexes)) == 3:
            return i


def improper_graph(impropers, bonds):
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(np.unique(impropers))
    g.add_edges_from([tuple(b) for b in bonds])
    return g

def set_positions(nreplicas, pos):
    if pos.shape[1] != 3:
        raise RuntimeError(
            "Positions shape must be (natoms, 3, 1) or (natoms, 3, nreplicas)"
        )

    atom_pos = nnp.transpose(pos, (2, 0, 1))
    if nreplicas > 1 and atom_pos.shape[0] != nreplicas:
        atom_pos = nnp.repeat(atom_pos[0][None, :], nreplicas, axis=0)

    # self.pos[:] = torch.tensor(
    #     atom_pos, dtype=self.pos.dtype, device=self.pos.device
    # )
    return atom_pos

# def set_velocities(vel):
#     if vel.shape != (self.nreplicas, self.natoms, 3):
#         raise RuntimeError("Velocities shape must be (nreplicas, natoms, 3)")
#     self.vel[:] = vel.clone().detach().type(self.vel.dtype).to(self.vel.device)

def set_box(nreplicas, box):
    if box.ndim == 1:
        if len(box) != 3:
            raise RuntimeError("Box must have at least 3 elements")
        box = box[:, None]

    if box.shape[0] != 3:
        raise RuntimeError("Box shape must be (3, 1) or (3, nreplicas)")

    box = np.swapaxes(box, 1, 0)

    if nreplicas > 1 and box.shape[0] != nreplicas:
        box = np.repeat(box[0][None, :], nreplicas, axis=0)

    new_box = np.zeros((nreplicas, 3, 3))
    for r in range(box.shape[0]):
        new_box[r][np.array(np.eye(3), dtype=bool)] = np.array(
            box[r], dtype=box.dtype
        )
    return new_box