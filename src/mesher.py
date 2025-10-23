import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import io
import gmsh

MARKERS = {
    'fluid': 1,
    'obstacle': 2,
    'border': 3
}


def build_sigma_mesh(x=0.0, y=0.0, r=1.0):
    gmsh.initialize()
    border = gmsh.model.occ.addDisk(0, 0, 0, 10, 10, tag=1)
    obstacle = gmsh.model.occ.addDisk(x, y, 0, r, r, tag=2)
    fluid_nodes = gmsh.model.occ.cut([(2, border)], [(2, obstacle)])
    gmsh.model.occ.synchronize()
    border_nodes, obstacle_nodes = __label_nodes()

    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", obstacle_nodes)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", 0.1)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", 1.0)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.1)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 3.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)


    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Netgen")
    mesh, cell_tags, facet_tags = io.gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, 0, gdim=2
    )
    gmsh.finalize()
    return mesh, cell_tags, facet_tags


def __label_nodes():
    volumes = gmsh.model.getEntities(dim=2)   
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], MARKERS['fluid'])
    gmsh.model.setPhysicalName(volumes[0][0], MARKERS['fluid'], 'fluid')   

    border_nodes, obstacle_nodes = [], []
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    for boundary in boundaries:
        if boundary[1] == MARKERS['border']:
            border_nodes.append(boundary[1])
        elif boundary[1] == MARKERS['obstacle']:
            obstacle_nodes.append(boundary[1])

    gmsh.model.addPhysicalGroup(1, border_nodes, MARKERS['border'])
    gmsh.model.setPhysicalName(1, MARKERS['border'], 'border')
    gmsh.model.addPhysicalGroup(1, obstacle_nodes, MARKERS['obstacle'])
    gmsh.model.setPhysicalName(1, MARKERS['obstacle'], 'obstacle')
    return border_nodes, obstacle_nodes