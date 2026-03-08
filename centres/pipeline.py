from .field import build_structural_field, reconstruct_field
from .graph import build_graph, propagate_strength
from .energy import total_energy
from .centers import Center
from skimage.feature import blob_log
from scipy.spatial.distance import cdist
import numpy as np


def detect_centers(field):
    """Detect multi-scale centres using Laplacian-of-Gaussian blob detection.

    A single blob_log call with log-spaced sigma values avoids duplicate
    detections that occurred when running four separate overlapping scale ranges.
    blob_log applies its own NMS internally via the overlap parameter.
    """
    blobs = blob_log(
        field, min_sigma=2, max_sigma=48, num_sigma=10, threshold=0.08, log_scale=True
    )
    centers = []
    for cid, (y, x, sigma) in enumerate(blobs):
        centers.append(
            Center(
                id=cid,
                x=float(x),
                y=float(y),
                scale=float(sigma * np.sqrt(2)),
                strength=float(field[int(y), int(x)]),
            )
        )
    return centers


def assign_hierarchy(centers):
    """Assign each centre its nearest larger centre within 3× that centre's radius.

    Containment uses dist(i, j) < 3 * scale_j rather than the strict 1×
    radius condition. The strict condition (k=1) almost never fires on real
    images: LoG blobs at different scales detect different spatial features
    whose centres are typically separated by several parent-radii. With k=3
    the assignment still requires the child to be substantially closer to its
    parent than to arbitrary large centres elsewhere, while capturing the
    real spatial nesting that Alexander's hierarchy describes.
    """
    if not centers:
        return centers
    pos = np.array([[c.x, c.y] for c in centers])
    scales = np.array([c.scale for c in centers])
    dist = cdist(pos, pos)
    for i, c in enumerate(centers):
        contained = dist[i] < 3 * scales  # dist(i,j) < 3 * scale_j
        larger = scales > c.scale
        possible = np.where(contained & larger)[0]
        if len(possible) == 0:
            continue
        j = possible[np.argmin(dist[i, possible])]
        centers[i].parent = int(j)
    return centers


def analyze(image):
    field = build_structural_field(image)
    centers = detect_centers(field)
    centers = assign_hierarchy(centers)
    G = build_graph(centers)
    G = propagate_strength(G)
    field2 = reconstruct_field(field.shape, centers)
    energy = total_energy(field2, centers, G)
    return field2, centers, G, energy


def random_centers(n, shape):
    h, w = shape
    centers = []
    for i in range(n):
        centers.append(
            Center(
                id=i,
                x=np.random.uniform(0, w),
                y=np.random.uniform(0, h),
                scale=np.random.uniform(5, 40),
                strength=np.random.uniform(0.3, 1),
            )
        )
    return centers


def evolve(
    shape=(300, 300),
    iterations=200,
    n_centers=40,
    T_start=1.0,
    T_end=0.01,
    progress=None,
    initial_centers=None,
):
    """Generative evolution via simulated annealing toward minimum structural energy.

    Each iteration perturbs all centre positions and scales, evaluates energy,
    and accepts or rejects via the Metropolis criterion. Temperature decays
    exponentially from T_start to T_end.

    initial_centers: if provided, seed the search from these centres rather
                     than from a random configuration. Useful for refining the
                     centre structure detected in an existing image.
    progress: optional callable(iteration, total, energy, temperature, accepted)
              called after each iteration.
    """
    h, w = shape
    if initial_centers is not None:
        centers = [
            Center(id=c.id, x=c.x, y=c.y, scale=c.scale, strength=c.strength)
            for c in initial_centers
        ]
    else:
        centers = random_centers(n_centers, shape)
    centers = assign_hierarchy(centers)
    G = build_graph(centers)
    G = propagate_strength(G)
    field = reconstruct_field(shape, centers)
    current_energy = total_energy(field, centers, G)

    for t in range(iterations):
        T = T_start * (T_end / T_start) ** (t / iterations)
        saved = [(c.x, c.y, c.scale, c.strength, c.parent) for c in centers]

        for c in centers:
            c.x = float(np.clip(c.x + np.random.normal(0, 2), 0, w))
            c.y = float(np.clip(c.y + np.random.normal(0, 2), 0, h))
            c.scale = float(np.clip(c.scale * np.exp(np.random.normal(0, 0.02)), 2, 80))

        centers = assign_hierarchy(centers)
        G_new = build_graph(centers)
        G_new = propagate_strength(G_new)
        field_new = reconstruct_field(shape, centers)
        new_energy = total_energy(field_new, centers, G_new)

        delta = new_energy - current_energy
        if delta < 0 or np.random.random() < np.exp(-delta / (T + 1e-10)):
            current_energy = new_energy
            G, field = G_new, field_new
            accepted = True
        else:
            for c, (x, y, sc, st, p) in zip(centers, saved):
                c.x, c.y, c.scale, c.strength, c.parent = x, y, sc, st, p
            accepted = False

        if progress is not None:
            progress(t + 1, iterations, current_energy, T, accepted)

    return field, centers
