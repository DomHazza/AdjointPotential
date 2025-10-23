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


def conformal_map_dzdsigma(x, y):
    sigma = x + 1j * y
    return 1 - 1 / (sigma)**2


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


def show_flow(mesh, facet_tags, u_n, phi, rho):
    fig, ax = plt.subplots(
        nrows=1, ncols=2,
        figsize=(12, 6)
    )

    length = 3
    x_min, x_max = -length, length
    y_min, y_max = -length, length
    nx, ny = 200, 200
    x_grid = np.linspace(x_min, x_max, nx)
    y_grid = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x_grid, y_grid)

    points_eval = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
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

    # Evaluate phi at grid points
    P = np.zeros(X.shape)
    for i in range(len(points_eval)):
        cells = colliding_cells.links(i)
        if len(cells) > 0:
            try:
                pres = phi.eval(points_eval[i], cells[:1])
                P.flat[i] = pres[0]
            except:
                # Point might be outside the domain
                P.flat[i] = np.nan
    
    # Evaluate density at grid points
    R = np.zeros(X.shape)
    for i in range(len(points_eval)):
        cells = colliding_cells.links(i)
        if len(cells) > 0:
            try:
                dens = rho.eval(points_eval[i], cells[:1])
                R.flat[i] = dens[0]
            except:
                # Point might be outside the domain
                R.flat[i] = np.nan

    # Create density contour plot
    contour_rho = ax[0].contourf(X, Y, R, levels=20, cmap='viridis', alpha=0.6)
    plt.colorbar(contour_rho, ax=ax[0], label='Density')
    ax[0].streamplot(X, Y, U, V, density=2, color='black', linewidth=1, arrowsize=1.5)
    ax[0].set_xlim(x_min, x_max)
    ax[0].set_ylim(y_min, y_max)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title('Density Field')
    ax[0].set_aspect('equal')

    # Create pressure contour plot
    contour = ax[1].contourf(X, Y, P, levels=20, cmap='viridis', alpha=0.6)
    plt.colorbar(contour, ax=ax[1], label='Potential')
    ax[1].streamplot(X, Y, U, V, density=2, color='black', linewidth=1, arrowsize=1.5)
    ax[1].set_xlim(x_min, x_max)
    ax[1].set_ylim(y_min, y_max)
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_title('Potential Field')
    ax[1].set_aspect('equal')

    # Plot obstacle boundary
    topology = mesh.topology
    topology.create_connectivity(1, 0)
    facet_to_vertex = topology.connectivity(1, 0)

    # Find facets tagged as obstacle (assuming tag value for obstacle)
    obstacle_facets = []
    for tag_value in set(facet_tags.values):
        if tag_value != 0:  # Assuming 0 is not obstacle, adjust as needed
            obstacle_facets.extend(facet_tags.find(tag_value))

    # Extract coordinates of obstacle boundary
    coords = mesh.geometry.x
    for facet in obstacle_facets:
        vertices = facet_to_vertex.links(facet)
        if len(vertices) == 2:  # Line segment
            x_coords = [coords[vertices[0], 0], coords[vertices[1], 0]]
            y_coords = [coords[vertices[0], 1], coords[vertices[1], 1]]
            ax[0].plot(x_coords, y_coords, 'w-', linewidth=2)
            ax[1].plot(x_coords, y_coords, 'w-', linewidth=2)

    plt.tight_layout()
    return fig



