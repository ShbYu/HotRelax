from typing import List, Optional, Tuple

import numpy as np
from ase import Atoms
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


__all__ = [
    "get_atom_feature_names",
    "compute_voronoi_coordination_numbers",
    "compute_voronoi_volumes",
    "compute_local_densities",
    "compute_max_neighbor_distances",
    "compute_min_neighbor_distances",
    "compute_mean_neighbor_distances",
    "compute_atom_features",
]


def fill_non_finite(values: np.ndarray, fill_value: Optional[float] = None) -> np.ndarray:
    """
    Replace non-finite entries with a stable fallback value.

    Args:
        values: Input one-dimensional feature array.
        fill_value: Explicit replacement value. When None, use the median of
            finite values and fall back to 0.0 if no finite values exist.

    Returns:
        Array with all non-finite entries replaced by finite values.
    """
    values = np.asarray(values, dtype=float).copy()
    invalid_mask = ~np.isfinite(values)
    if not np.any(invalid_mask):
        return values
    if fill_value is None:
        finite_values = values[np.isfinite(values)]
        if finite_values.size > 0:
            fill_value = float(np.median(finite_values))
        else:
            fill_value = 0.0
    values[invalid_mask] = fill_value
    return values


def get_atom_feature_names() -> List[str]:
    """
    Return the default atom feature names produced by this module.

    Args:
        None.

    Returns:
        Ordered list of atom feature names.
    """
    return [
        "voronoi_coordination_number",
        "voronoi_volume",
        "local_density",
        "max_neighbor_distance",
        "min_neighbor_distance",
        "mean_neighbor_distance",
    ]


def _compute_voronoi_statistics(
    atoms: Atoms,
    tol: float = 0.5,
    cutoff: Optional[float] = None,
    use_weights: bool = True,
) -> dict:
    """
    Compute shared Voronoi-based atom statistics in one pass.

    Args:
        atoms: Periodic ASE structure.
        tol: VoronoiNN tolerance used to prune weak faces.
        cutoff: Neighbor search cutoff. When None, use the mean lattice length.
        use_weights: Whether to compute weighted coordination numbers.

    Returns:
        Dictionary containing raw per-atom Voronoi statistics.
    """
    struct = AseAtomsAdaptor.get_structure(atoms)
    if cutoff is None:
        cutoff = float(np.mean(atoms.cell.cellpar()[:3]))
    voronoi_nn = VoronoiNN(tol=tol, cutoff=cutoff)
    n_atoms = len(struct)
    coordination_numbers = np.full(n_atoms, np.nan, dtype=float)
    volumes = np.full(n_atoms, np.nan, dtype=float)
    max_neighbor_distances = np.full(n_atoms, np.nan, dtype=float)
    min_neighbor_distances = np.full(n_atoms, np.nan, dtype=float)
    mean_neighbor_distances = np.full(n_atoms, np.nan, dtype=float)
    distance_rows = []
    for index in range(n_atoms):
        current_distances = []
        try:
            neighbors = voronoi_nn.get_nn_info(struct, index)
            if use_weights:
                coordination_numbers[index] = float(
                    np.sum([neighbor.get("weight", 1.0) for neighbor in neighbors])
                )
            else:
                coordination_numbers[index] = float(len(neighbors))
            polyhedra = voronoi_nn.get_voronoi_polyhedra(struct, index)
            current_volumes = [
                face_info["volume"]
                for face_info in polyhedra.values()
                if "volume" in face_info
            ]
            if len(current_volumes) > 0:
                volumes[index] = float(np.sum(current_volumes))
                if volumes[index] <= 0.0:
                    volumes[index] = np.nan
            for neighbor in neighbors:
                current_distances.append(
                    float(
                        struct.get_distance(
                            index,
                            neighbor["site_index"],
                            jimage=neighbor["image"],
                        )
                    )
                )
        except Exception:
            current_distances = []
        current_array = np.asarray(current_distances, dtype=float)
        if current_array.size > 0:
            max_neighbor_distances[index] = float(np.max(current_array))
            min_neighbor_distances[index] = float(np.min(current_array))
            mean_neighbor_distances[index] = float(np.mean(current_array))
        distance_rows.append(current_array)
    return {
        "coordination_numbers": coordination_numbers,
        "volumes": volumes,
        "max_neighbor_distances": max_neighbor_distances,
        "min_neighbor_distances": min_neighbor_distances,
        "mean_neighbor_distances": mean_neighbor_distances,
        "distance_rows": distance_rows,
    }


def compute_neighbor_distances(
    atoms: Atoms,
    tol: float = 0.5,
    cutoff: Optional[float] = None,
    fill_value: Optional[float] = None,
) -> np.ndarray:
    """
    Compute Voronoi-neighbor distances for each atom.

    Args:
        atoms: Periodic ASE structure.
        tol: VoronoiNN tolerance used to prune weak faces.
        cutoff: Neighbor search cutoff. When None, use the mean lattice length.
        fill_value: Replacement value for atoms without valid neighbors. When
            None, use the median finite distance from the same structure.

    Returns:
        Two-dimensional array of shape [n_atoms, max_neighbors] padded with
        non-finite values for missing neighbors.
    """
    if len(atoms) == 0:
        return np.empty((0, 0), dtype=float)
    stats = _compute_voronoi_statistics(
        atoms=atoms,
        tol=tol,
        cutoff=cutoff,
        use_weights=True,
    )
    distance_rows = stats["distance_rows"]
    max_neighbors = max((row.size for row in distance_rows), default=0)
    if max_neighbors == 0:
        return np.full(
            (len(atoms), 1),
            fill_value if fill_value is not None else np.nan,
            dtype=float,
        )
    distance_matrix = np.full((len(atoms), max_neighbors), np.nan, dtype=float)
    for index, current_array in enumerate(distance_rows):
        if current_array.size > 0:
            distance_matrix[index, : current_array.size] = current_array
    if fill_value is not None:
        invalid_mask = ~np.isfinite(distance_matrix)
        distance_matrix[invalid_mask] = fill_value
    return distance_matrix


def compute_max_neighbor_distances(
    atoms: Atoms,
    tol: float = 0.5,
    cutoff: Optional[float] = None,
    fill_value: Optional[float] = None,
) -> np.ndarray:
    """
    Compute the maximum Voronoi-neighbor distance for each atom.

    Args:
        atoms: Periodic ASE structure.
        tol: VoronoiNN tolerance used to prune weak faces.
        cutoff: Neighbor search cutoff. When None, use the mean lattice length.
        fill_value: Replacement value when one site fails Voronoi analysis.

    Returns:
        One-dimensional array with shape [n_atoms].
    """
    stats = _compute_voronoi_statistics(
        atoms=atoms,
        tol=tol,
        cutoff=cutoff,
        use_weights=True,
    )
    return fill_non_finite(stats["max_neighbor_distances"], fill_value=fill_value)


def compute_min_neighbor_distances(
    atoms: Atoms,
    tol: float = 0.5,
    cutoff: Optional[float] = None,
    fill_value: Optional[float] = None,
) -> np.ndarray:
    """
    Compute the minimum Voronoi-neighbor distance for each atom.

    Args:
        atoms: Periodic ASE structure.
        tol: VoronoiNN tolerance used to prune weak faces.
        cutoff: Neighbor search cutoff. When None, use the mean lattice length.
        fill_value: Replacement value when one site fails Voronoi analysis.

    Returns:
        One-dimensional array with shape [n_atoms].
    """
    stats = _compute_voronoi_statistics(
        atoms=atoms,
        tol=tol,
        cutoff=cutoff,
        use_weights=True,
    )
    return fill_non_finite(stats["min_neighbor_distances"], fill_value=fill_value)


def compute_mean_neighbor_distances(
    atoms: Atoms,
    tol: float = 0.5,
    cutoff: Optional[float] = None,
    fill_value: Optional[float] = None,
) -> np.ndarray:
    """
    Compute the mean Voronoi-neighbor distance for each atom.

    Args:
        atoms: Periodic ASE structure.
        tol: VoronoiNN tolerance used to prune weak faces.
        cutoff: Neighbor search cutoff. When None, use the mean lattice length.
        fill_value: Replacement value when one site fails Voronoi analysis.

    Returns:
        One-dimensional array with shape [n_atoms].
    """
    stats = _compute_voronoi_statistics(
        atoms=atoms,
        tol=tol,
        cutoff=cutoff,
        use_weights=True,
    )
    return fill_non_finite(stats["mean_neighbor_distances"], fill_value=fill_value)


def compute_voronoi_coordination_numbers(
    atoms: Atoms,
    tol: float = 0.5,
    cutoff: Optional[float] = None,
    use_weights: bool = True,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Compute one Voronoi coordination number for each atom.

    Args:
        atoms: Periodic ASE structure.
        tol: VoronoiNN tolerance used to prune weak faces.
        cutoff: Neighbor search cutoff. When None, use the mean lattice length.
        use_weights: Whether to return weighted coordination numbers.
        fill_value: Replacement value when one site fails Voronoi analysis.

    Returns:
        One-dimensional array with shape [n_atoms].
    """
    stats = _compute_voronoi_statistics(
        atoms=atoms,
        tol=tol,
        cutoff=cutoff,
        use_weights=use_weights,
    )
    return fill_non_finite(stats["coordination_numbers"], fill_value=fill_value)


def compute_voronoi_volumes(
    atoms: Atoms,
    tol: float = 0.5,
    cutoff: Optional[float] = None,
    fill_value: Optional[float] = None,
) -> np.ndarray:
    """
    Compute one Voronoi cell volume for each atom.

    Args:
        atoms: Periodic ASE structure.
        tol: VoronoiNN tolerance used to prune weak faces.
        cutoff: Neighbor search cutoff. When None, use the mean lattice length.
        fill_value: Replacement value when one site fails Voronoi analysis.
            When None, use the median finite volume.

    Returns:
        One-dimensional array with shape [n_atoms].
    """
    stats = _compute_voronoi_statistics(
        atoms=atoms,
        tol=tol,
        cutoff=cutoff,
        use_weights=True,
    )
    return fill_non_finite(stats["volumes"], fill_value=fill_value)


def compute_local_densities(
    atoms: Atoms,
    tol: float = 0.5,
    cutoff: Optional[float] = None,
    use_weights: bool = True,
    eps: float = 1e-8,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Compute one local density estimate for each atom.

    Args:
        atoms: Periodic ASE structure.
        tol: VoronoiNN tolerance used to prune weak faces.
        cutoff: Neighbor search cutoff. When None, use the mean lattice length.
        use_weights: Whether to use weighted Voronoi coordination numbers.
        eps: Minimum volume used in the division denominator.
        fill_value: Replacement value when one site fails Voronoi analysis.

    Returns:
        One-dimensional array with shape [n_atoms].
    """
    stats = _compute_voronoi_statistics(
        atoms=atoms,
        tol=tol,
        cutoff=cutoff,
        use_weights=use_weights,
    )
    coordination_numbers = fill_non_finite(
        stats["coordination_numbers"],
        fill_value=fill_value,
    )
    volumes = stats["volumes"]
    densities = np.full(len(coordination_numbers), np.nan, dtype=float)
    valid_mask = np.isfinite(volumes)
    densities[valid_mask] = coordination_numbers[valid_mask] / np.clip(
        volumes[valid_mask], eps, None
    )
    return fill_non_finite(densities, fill_value=fill_value)


def compute_atom_features(
    atoms: Atoms,
    tol: float = 0.5,
    cutoff: Optional[float] = None,
    use_weights: bool = False,
    eps: float = 1e-8,
    fill_value: float = 0.0,
    volume_clip: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Compute the default Voronoi-based atom feature matrix.

    Args:
        atoms: Periodic ASE structure.
        tol: VoronoiNN tolerance used to prune weak faces.
        cutoff: Neighbor search cutoff. When None, use the mean lattice length.
        use_weights: Whether to use weighted Voronoi coordination numbers.
        eps: Minimum volume used in the local-density denominator.
        fill_value: Replacement value when one site fails Voronoi analysis.
        volume_clip: Optional lower and upper bounds applied to Voronoi volumes.

    Returns:
        Two-dimensional array with shape [n_atoms, 6].
    """
    stats = _compute_voronoi_statistics(
        atoms=atoms,
        tol=tol,
        cutoff=cutoff,
        use_weights=use_weights,
    )
    coordination_numbers = fill_non_finite(
        stats["coordination_numbers"],
        fill_value=fill_value,
    )
    volumes = fill_non_finite(stats["volumes"], fill_value=fill_value)
    if volume_clip is not None:
        volumes = np.clip(volumes, volume_clip[0], volume_clip[1])
    densities = coordination_numbers / np.clip(volumes, eps, None)
    densities = fill_non_finite(densities, fill_value=fill_value)
    max_neighbor_distances = fill_non_finite(
        stats["max_neighbor_distances"],
        fill_value=fill_value,
    )
    min_neighbor_distances = fill_non_finite(
        stats["min_neighbor_distances"],
        fill_value=fill_value,
    )
    mean_neighbor_distances = fill_non_finite(
        stats["mean_neighbor_distances"],
        fill_value=fill_value,
    )
    return np.stack(
        [
            coordination_numbers,
            volumes,
            # densities,
            # max_neighbor_distances,
            # min_neighbor_distances,
            # mean_neighbor_distances,
        ],
        axis=1,
    )
