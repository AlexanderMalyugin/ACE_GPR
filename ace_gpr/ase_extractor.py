import numpy as np
import ase
from itertools import combinations, combinations_with_replacement
from collections import defaultdict
from ase.neighborlist import neighbor_list


class Cluster_Expansion:
    def __init__(
        self,
        atoms: ase.Atoms,
        shells: dict | None = None,
        max_order: int = 2,
        atom_indices: list[int] | None = None,
    ):

        self.atoms = atoms
        self.shells = shells or {}
        self.max_order = max_order

        self.atom_indices = None if atom_indices is None else sorted(set(atom_indices))
        self.atom_index_set = None if atom_indices is None else set(atom_indices)

        self.elements_list = self.atoms.get_chemical_symbols()
        self.elements = sorted(np.unique(self.elements_list))

        self.clusters = None
        self.descriptor = None
        self.descript = None  # backward compatibility
        self.names = None

        self._validate_inputs()
        self.generate_all_descriptors()

    def _validate_inputs(self):
        """Validate constructor inputs."""
        if self.max_order not in (1, 2, 3):
            raise ValueError("max_order must be 1, 2, or 3.")

        if self.max_order >= 2 and not self.shells:
            raise ValueError("shells must be provided when max_order >= 2.")

        for shell_name, bounds in self.shells.items():
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise ValueError(
                    f"Shell '{shell_name}' must be a tuple/list of the form (rmin, rmax)."
                )

            rmin, rmax = bounds
            if rmin < 0:
                raise ValueError(f"Shell '{shell_name}' has negative rmin: {rmin}")
            if rmax <= rmin:
                raise ValueError(
                    f"Shell '{shell_name}' must satisfy rmax > rmin, got {(rmin, rmax)}"
                )

        if self.atom_indices is not None:
            n_atoms = len(self.atoms)
            for idx in self.atom_indices:
                if idx < 0 or idx >= n_atoms:
                    raise ValueError(
                        f"Atom index {idx} is out of bounds for {n_atoms} atoms."
                    )

    def chemical_labels(self, order: int):

        return [
            ''.join(comb)
            for comb in combinations_with_replacement(self.elements, order)
        ]

    def cluster_chem_key(self, cluster_indices, symbols):

        chem = sorted(symbols[i] for i in cluster_indices)
        return ''.join(chem)

    def _cluster_matches_selection(self, cluster_indices):

        if self.atom_index_set is None:
            return True
        return any(i in self.atom_index_set for i in cluster_indices)

    def _build_single_clusters(self):
        n_atoms = len(self.atoms)
        return {"singles": [[i] for i in range(n_atoms)]}

    def _build_pair_clusters(self):

        max_cutoff = max(rmax for _, rmax in self.shells.values())

        i_arr, j_arr, d_arr = neighbor_list("ijd", self.atoms, max_cutoff)

        # Collapse possibly duplicated directed pairs to unique unordered pairs
        # with the minimum observed distance.
        pair_min_dist = {}

        for i, j, d in zip(i_arr, j_arr, d_arr):
            if i == j:
                continue

            a, b = (i, j) if i < j else (j, i)
            key = (a, b)

            if key not in pair_min_dist or d < pair_min_dist[key]:
                pair_min_dist[key] = float(d)

        pair_clusters = {}
        pair_sets = {}

        for shell_name, (rmin, rmax) in self.shells.items():
            pairs_in_shell = [
                [i, j]
                for (i, j), d in pair_min_dist.items()
                if rmin <= d < rmax
            ]
            pairs_in_shell.sort(key=lambda x: (x[0], x[1]))

            pair_clusters[shell_name] = pairs_in_shell
            pair_sets[shell_name] = {tuple(p) for p in pairs_in_shell}

        return pair_clusters, pair_sets

    @staticmethod
    def pairlist_to_dict(pair_iterable):

        neighbors = defaultdict(set)
        for i, j in pair_iterable:
            neighbors[i].add(j)
            neighbors[j].add(i)
        return neighbors

    def _build_triplet_clusters(self, pair_sets):

        triplet_clusters = {}
        shell_names = list(self.shells.keys())

        for hips_name in shell_names:
            hips_pairs = pair_sets[hips_name]
            neigh = self.pairlist_to_dict(hips_pairs)

            for base_name in shell_names:
                base_pairs = pair_sets[base_name]
                triplet_name = f"trip_hips_{hips_name}_base_{base_name}"

                triplet_set = set()

                for center, nbrs in neigh.items():
                    nbrs = sorted(nbrs)
                    for j, k in combinations(nbrs, 2):
                        jk = (j, k) if j < k else (k, j)
                        if jk in base_pairs:
                            triplet = tuple(sorted((center, j, k)))
                            triplet_set.add(triplet)

                triplet_clusters[triplet_name] = [list(t) for t in sorted(triplet_set)]

        return triplet_clusters

    def build_clusters(self):

        clusters = {}

        if self.max_order >= 1:
            clusters.update(self._build_single_clusters())

        pair_sets = {}
        if self.max_order >= 2:
            pair_clusters, pair_sets = self._build_pair_clusters()
            clusters.update(pair_clusters)

        if self.max_order >= 3:
            triplet_clusters = self._build_triplet_clusters(pair_sets)
            clusters.update(triplet_clusters)

        return clusters

    def count_descriptors(self):

        descriptor = []
        names = []

        ordered_geom_types = []

        if self.max_order >= 1:
            ordered_geom_types.append("singles")

        for shell_name in self.shells.keys():
            ordered_geom_types.append(shell_name)

        if self.max_order >= 3:
            for hips_name in self.shells.keys():
                for base_name in self.shells.keys():
                    ordered_geom_types.append(f"trip_hips_{hips_name}_base_{base_name}")

        for geom_type in ordered_geom_types:
            cluster_list = self.clusters.get(geom_type, [])

            if geom_type == "singles":
                order = 1
            elif geom_type.startswith("trip"):
                order = 3
            else:
                order = 2

            labels = self.chemical_labels(order)
            counts = defaultdict(int)

            for cluster_indices in cluster_list:
                if not self._cluster_matches_selection(cluster_indices):
                    continue

                key = self.cluster_chem_key(cluster_indices, self.elements_list)
                counts[key] += 1

            for label in labels:
                descriptor.append(float(counts.get(label, 0)))
                names.append(f"{geom_type}:{label}")

        return np.asarray(descriptor, dtype=float), names

    def generate_all_descriptors(self):
        self.clusters = self.build_clusters()
        self.descriptor, self.names = self.count_descriptors()
        return 0