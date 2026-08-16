from dataclasses import dataclass


@dataclass
class Center:
    """A centre: a region of space with a scale, a strength and a polarity.

    ``polarity`` is signed local contrast in [-1, 1]: the Michelson contrast
    between the mean intensity inside the centre's own radius and the mean over
    the annulus immediately outside it. Positive means the interior is *darker*
    than its surround, negative means lighter. Zero means the centre sits on no
    tonal boundary at all.

    It exists because the structural field is a distance transform from edges and
    so cannot tell figure from ground: the space *between* motifs is as much a
    local maximum as the motifs themselves, and where motifs are small and gaps
    wide it is a larger one. On a lattice of 225 identical circles the detector
    finds each circle exactly once and a further 256 centres in the gaps, with
    the gap detections responding *more* strongly to the LoG than any motif does.
    Nothing downstream could distinguish the two populations, so every measure
    that assumes "a centre is a motif" silently averaged over both. See #26.

    Which sign counts as figure is a convention, not a fact about the image — a
    carpet may as easily be light motifs on a dark ground. What the sign carries
    is that two centres of *opposite* polarity are of different kinds.
    """

    id: int
    x: float
    y: float
    scale: float
    strength: float
    parent: int = None
    polarity: float = 0.0
