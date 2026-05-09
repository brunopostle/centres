import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt

from centres.centers import Center
from centres.visualize import draw_onto_ax, visualize


def _make_field(shape=(50, 50)):
    h, w = shape
    Y, X = np.mgrid[0:h, 0:w]
    return np.exp(-((X - 25) ** 2 + (Y - 25) ** 2) / (2 * 8**2))


def _make_centers():
    return [
        Center(id=0, x=25.0, y=25.0, scale=10.0, strength=1.0),
        Center(id=1, x=15.0, y=15.0, scale=4.0, strength=0.6, parent=0),
    ]


def test_draw_onto_ax_runs_without_error():
    field = _make_field()
    centers = _make_centers()
    fig, ax = plt.subplots()
    draw_onto_ax(ax, field, centers)
    plt.close(fig)


def test_draw_onto_ax_with_image():
    field = _make_field()
    centers = _make_centers()
    image = np.full((50, 50, 3), 128, dtype=np.uint8)
    fig, ax = plt.subplots()
    draw_onto_ax(ax, field, centers, image=image)
    plt.close(fig)


def test_draw_onto_ax_clears_previous_content():
    field = _make_field()
    centers = _make_centers()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])  # add something that should be cleared
    draw_onto_ax(ax, field, centers)
    # after draw_onto_ax the axis should be off and contain imshow output
    assert not ax.get_xaxis().get_visible() or ax.axison is False
    plt.close(fig)


def test_draw_onto_ax_with_graph():
    import networkx as nx
    field = _make_field()
    centers = _make_centers()
    G = nx.Graph()
    for c in centers:
        G.add_node(c.id, center=c)
    G.add_edge(0, 1, weight=0.5)
    fig, ax = plt.subplots()
    draw_onto_ax(ax, field, centers, G=G)
    plt.close(fig)


def test_visualize_save(tmp_path):
    field = _make_field()
    centers = _make_centers()
    out = tmp_path / "out.png"
    visualize(field, centers, save_path=str(out))
    assert out.exists()
    assert out.stat().st_size > 0
