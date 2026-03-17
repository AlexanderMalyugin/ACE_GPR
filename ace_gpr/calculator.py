import ase
import torch

from modules.ase_extractor import *
from modules.dataset import calc_mindist, atoms_near_carbon


class calculator:

    def __init__(self, model, mode):
        self.model = model
        self.mode = mode

    def __call__(self,
                 atoms: ase.Atoms,
                 ):

        self.model.eval()

        if self.mode == "TOTEN":

            mindist = calc_mindist(atoms)
            two = mindist * 2.1
            atom_indices = None

        if self.mode == "E_ADS":

            atoms, atom_indices = atoms_near_carbon(atoms)
            mindist = calc_mindist(atoms)
            two = mindist * 2.2

        one = mindist * 1.2
        sqrt2 = mindist * np.sqrt(2.6)
        sqrt3 = mindist * np.sqrt(3.6)

        shells = {
            "pairs": (0, one),
            "sqrt2_pairs": (one, sqrt2),
            "sqrt3_pairs": (sqrt2, sqrt3),
            "two_pairs": (sqrt3, two),
        }

        descriptor = Cluster_Expansion(atoms = atoms,
                                       max_order = 3,
                                       shells = shells,
                                       atom_indices = atom_indices
                                       )

        descriptor = torch.Tensor(descriptor.descriptor).unsqueeze(dim=0)

        energy = self.model(descriptor).mean.detach().numpy()
        stddev = self.model(descriptor).stddev.detach().numpy()

        return energy, stddev