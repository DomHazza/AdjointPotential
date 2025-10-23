import matplotlib.pyplot as plt


def show_mesh(mesh, facet_tags):
    coords = mesh.geometry.x
    topology = mesh.topology
    topology.create_connectivity(topology.dim, 0)
    topology.create_connectivity(1, 0) 
    cells = topology.connectivity(topology.dim, 0).array.reshape(-1, 3)

    fig, ax_1 = plt.subplots(
        nrows=1, ncols=1, figsize=(8, 6)
    )
    ax_1.triplot(coords[:, 0], coords[:, 1], cells, 'k-', linewidth=0.3)
    __add_nodes(mesh, facet_tags, coords, ax_1)
    ax_1.set_aspect('equal')
    ax_1.set_xlabel('x')
    ax_1.set_ylabel('y')
    ax_1.set_title('Gmsh Mesh')
    ax_1.legend()
    plt.tight_layout()
    return fig




def __add_nodes(mesh, facet_tags, coords, ax_1):
    facet_to_cell = mesh.topology.connectivity(mesh.topology.dim - 1, 0)
    unique_tags = set(facet_tags.values)
    colors = plt.cm.tab10(range(len(unique_tags)))
        
    for idx, tag in enumerate(unique_tags):
        facets_with_tag = facet_tags.find(tag)
        nodes_in_tag = set()
        for facet in facets_with_tag:
            nodes_in_tag.update(facet_to_cell.links(facet))
        
        if nodes_in_tag:
            nodes_array = list(nodes_in_tag)
            ax_1.scatter(
                coords[nodes_array, 0], coords[nodes_array, 1], 
                c=[colors[idx]], label=f'Tag {tag}', s=10
            )










