import matplotlib.pyplot as plt


def draw_onto_ax(ax, field, centers, G=None, image=None):
    """Draw field, centres, hierarchy, and graph onto an existing Axes.

    Suitable for embedding in Qt via FigureCanvasQTAgg as well as use in
    standalone figures.  The caller is responsible for calling canvas.draw()
    or plt.show() after this returns.
    """
    ax.clear()
    if image is not None:
        ax.imshow(image[:, :, ::-1])  # BGR → RGB
        ax.imshow(field, cmap="inferno", alpha=0.35)
    else:
        ax.imshow(field, cmap="inferno")

    for c in centers:
        ax.scatter(c.x, c.y, c="cyan", s=25)
        ax.add_patch(
            plt.Circle((c.x, c.y), c.scale, color="cyan", fill=False,
                       linewidth=0.6, alpha=0.6)
        )

    for c in centers:
        if c.parent is not None:
            p = centers[c.parent]
            ax.plot([c.x, p.x], [c.y, p.y], color="white", linewidth=0.8, alpha=0.5)

    if G is not None:
        for i, j, data in G.edges(data=True):
            # Use the centre stored on the node, not centers[i], to avoid
            # silent breakage if centre ids ever diverge from list indices.
            c1 = G.nodes[i]["center"]
            c2 = G.nodes[j]["center"]
            w = data["weight"]
            ax.plot(
                [c1.x, c2.x], [c1.y, c2.y],
                color="lime", linewidth=0.5 + 2 * w, alpha=0.25,
            )

    ax.axis("off")


def visualize(field, centers, G=None, image=None, save_path=None):
    """Diagnostic visualization of centers, hierarchy, and reinforcement graph."""
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_onto_ax(ax, field, centers, G, image)
    ax.set_title("Structural Centers and Reinforcement Graph")
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
