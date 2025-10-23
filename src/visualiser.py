from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
import matplotlib.pyplot as plt
import numpy as np



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


def show_flow(mesh, u_n, p_):
    fig, ax = plt.subplots(figsize=(8, 8))

    length = 6
    x_min, x_max = -length, length
    y_min, y_max = -length, length
    nx, ny = 200, 200
    x_grid = np.linspace(x_min, x_max, nx)
    y_grid = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Flatten the grid points for evaluation
    points_eval = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    # Ensure the points array has the correct shape and dtype for DOLFINx
    points_eval = np.ascontiguousarray(points_eval, dtype=np.float64)

    # Find cells containing the evaluation points
    tree = bb_tree(mesh, mesh.geometry.dim)
    cell_candidates = compute_collisions_points(tree, points_eval)
    colliding_cells = compute_colliding_cells(mesh, cell_candidates, points_eval)

    # Initialize velocity components
    U = np.zeros(X.shape)
    V = np.zeros(X.shape)

    # Evaluate velocity at grid points
    for i in range(len(points_eval)):
        cells = colliding_cells.links(i)
        if len(cells) > 0:
            try:
                vel = u_n.eval(points_eval[i], cells[:1])
                U.flat[i] = vel[0]
                V.flat[i] = vel[1]
            except:
                # Point might be outside the domain
                U.flat[i] = 0
                V.flat[i] = 0

    # Evaluate pressure at grid points
    P = np.zeros(X.shape)
    for i in range(len(points_eval)):
        cells = colliding_cells.links(i)
        if len(cells) > 0:
            try:
                pres = p_.eval(points_eval[i], cells[:1])
                P.flat[i] = pres[0]
            except:
                # Point might be outside the domain
                P.flat[i] = np.nan

    # Create pressure contour plot
    contour = ax.contourf(X, Y, P, levels=20, cmap='jet', alpha=0.6)
    plt.colorbar(contour, ax=ax, label='Potential')

    # Create streamlines on top
    ax.streamplot(X, Y, U, V, density=2, color='black', linewidth=1, arrowsize=1.5)

    # Set plot properties
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Velocity Field Streamlines with Pressure Field')
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig







