import matplotlib.pyplot as plt


def visualize(field, centers, G=None, image=None, save_path=None):
    """
    Diagnostic visualization of centers, hierarchy, and reinforcement graph.
    """

    plt.figure(figsize=(8, 8))

    # Background
    if image is not None:
        plt.imshow(image[:, :, ::-1])  # BGR → RGB
        plt.imshow(field, cmap="inferno", alpha=0.35)
    else:
        plt.imshow(field, cmap="inferno")

    # Draw centers
    for c in centers:
        plt.scatter(c.x, c.y, c="cyan", s=25)
        circle = plt.Circle(
            (c.x, c.y), c.scale, color="cyan", fill=False, linewidth=0.6, alpha=0.6
        )
        plt.gca().add_patch(circle)

    # Hierarchy (parent relationships)
    for c in centers:
        if c.parent is not None:
            p = centers[c.parent]
            plt.plot([c.x, p.x], [c.y, p.y], color="white", linewidth=0.8, alpha=0.5)

    # Reinforcement graph edges
    if G is not None:
        for i, j, data in G.edges(data=True):
            # Use the centre stored on the node, not centers[i], to avoid
            # silent breakage if centre ids ever diverge from list indices.
            c1 = G.nodes[i]["center"]
            c2 = G.nodes[j]["center"]
            w = data["weight"]
            plt.plot(
                [c1.x, c2.x],
                [c1.y, c2.y],
                color="lime",
                linewidth=0.5 + 2 * w,
                alpha=0.25,
            )

    plt.title("Structural Centers and Reinforcement Graph")
    plt.axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
