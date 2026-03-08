import numpy as np
from collections import defaultdict


def hierarchy_energy(centers):
    """Penalise deviation from a scale ratio of ~3 between parent and child.

    Natural hierarchies (Alexander's 'levels of scale') follow approximately
    constant ratios between successive scales. The target ratio of 3 is the
    midpoint of the empirically observed range of 2-4.
    """
    E = 0
    for c in centers:
        if c.parent is None:
            continue
        parent = centers[c.parent]
        ratio = parent.scale / (c.scale + 1e-8)
        E += (np.log(ratio) - np.log(3)) ** 2
    return E


def reinforcement_energy(G):
    """Reward mutual strength between connected centres.

    E_R = -mean_ij W_ij * s_i * s_j  (normalised by N*(N-1)/2)

    Normalised so the value is O(1) regardless of centre count. Without
    normalisation E_R scales as O(N²) and the global minimum is a degenerate
    cluster where all centres coincide (maximum edges, maximum weights,
    maximum strengths after propagation).
    """
    n = G.number_of_nodes()
    if n < 2:
        return 0.0
    E = 0.0
    for i, j, data in G.edges(data=True):
        s1 = G.nodes[i]["center"].strength
        s2 = G.nodes[j]["center"].strength
        E -= data["weight"] * s1 * s2
    return E / (n * (n - 1) / 2)


def locality_energy(centers):
    """Penalise spatial overlap between centres.

    Computes mean pairwise Gaussian overlap exp(-d²/(r_i+r_j)²).
    Value is 1.0 when all centres are coincident, ~0 when well-separated.
    This prevents the degenerate minimum where all centres cluster at one point.
    """
    n = len(centers)
    if n < 2:
        return 0.0
    pos = np.array([[c.x, c.y] for c in centers])
    scales = np.array([c.scale for c in centers])
    dx = pos[:, 0:1] - pos[:, 0]
    dy = pos[:, 1:2] - pos[:, 1]
    d2 = dx**2 + dy**2
    r = scales[:, None] + scales[None, :]
    overlap = np.exp(-d2 / (r**2 + 1e-8))
    # upper triangle only (exclude self-pairs on diagonal)
    i_upper, j_upper = np.triu_indices(n, k=1)
    return float(overlap[i_upper, j_upper].mean())


def coverage_energy(centers):
    """Penalise deviation from ideal child-area coverage of a parent (~0.65).

    Coverage C_i = sum(r_child^2) / r_parent^2. Values near 0.65 indicate
    children fill their parent region without overcrowding it.
    """
    children = defaultdict(list)
    for c in centers:
        if c.parent is not None:
            children[c.parent].append(c)
    E = 0
    C_target = 0.65
    for parent_id, childs in children.items():
        parent = centers[parent_id]
        coverage = sum(ch.scale**2 for ch in childs) / (parent.scale**2 + 1e-8)
        E += (coverage - C_target) ** 2
    return E


def alignment_energy(centers):
    """Penalise children whose radial distance from parent deviates from 0.5 * r_parent.

    Alexander observed that child centres tend to lie at roughly 0.3-0.7 of the
    parent radius from the parent centre. Target is the midpoint, 0.5.
    """
    E = 0
    d_target = 0.5
    for c in centers:
        if c.parent is None:
            continue
        parent = centers[c.parent]
        d = np.sqrt((c.x - parent.x) ** 2 + (c.y - parent.y) ** 2) / (
            parent.scale + 1e-8
        )
        E += (d - d_target) ** 2
    return E


def field_energy(field):
    """Mean squared gradient: penalises abrupt transitions in the wholeness field."""
    gy, gx = np.gradient(field)
    return np.mean(gx**2 + gy**2)


def total_energy(field, centers, G):
    return (
        0.3 * hierarchy_energy(centers)
        + 0.3 * reinforcement_energy(G)
        + 0.2 * coverage_energy(centers)
        + 0.1 * alignment_energy(centers)
        + 0.1 * field_energy(field)
        + 50.0 * locality_energy(centers)
    )
