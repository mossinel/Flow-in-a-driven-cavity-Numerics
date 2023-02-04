# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 00:11:56 2022

@author: danie
"""

#### import ####
#import the modules and functions that the program uses:

# +
import numpy as np

import dolfinx
print(dolfinx.__version__, dolfinx.git_commit_hash)

import petsc4py
print(petsc4py.__version__)

import ufl
from dolfinx import cpp as _cpp
from dolfinx import fem, plot
from dolfinx.fem import (Constant, Function, FunctionSpace, dirichletbc,
                         extract_function_spaces, form,
                
                         locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.io import XDMFFile
from dolfinx.mesh import (CellType, GhostMode, create_rectangle,
                          locate_entities_boundary)
from ufl import div, dx, grad, inner, SpatialCoordinate

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.io import gmshio
import gmsh


#### MESH ####
# Create mesh

#MESH DEFINITION - CLASSIC
# msh = create_rectangle(MPI.COMM_WORLD,
#                         [np.array([0, 0]), np.array([1, 1])],
#                         [32, 32],
#                         CellType.triangle, GhostMode.none)


# MESH DEFINITION - REFINED
gmsh.initialize()
gmsh.model.add("refined")
fine_dim = 0.01
coarse_dim= 0.1
gmsh.model.geo.addPoint(0, 0, 0, coarse_dim, 1)
gmsh.model.geo.addPoint(1, 0, 0, coarse_dim, 2)
gmsh.model.geo.addPoint(0, 1, 0, fine_dim, 3)
gmsh.model.geo.addPoint(1, 1, 0, fine_dim, 4)

gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 4, 2)
gmsh.model.geo.addLine(4, 3, 3)
gmsh.model.geo.addLine(3, 1, 4)

gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
gmsh.model.geo.addPlaneSurface([1], 1)
gmsh.model.geo.synchronize()

gmsh.model.addPhysicalGroup(1, [1, 2, 4], 5)
gmsh.model.addPhysicalGroup(2, [1], name = "My surface")
gmsh.model.mesh.generate(2)

msh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0, gdim=2)
x = SpatialCoordinate(msh)

# Function to mark x = 0, x = 1 and y = 0
def noslip_boundary(x):
    return np.logical_or(np.logical_or(np.isclose(x[0], 0.0),
                                       np.isclose(x[0], 1.0)),
                         np.isclose(x[1], 0.0))

# Function to mark the lid (y = 1)
def lid(x):
    return np.isclose(x[1], 1.0)

# Lid velocity
def lid_velocity_expression(x):
    return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))
# -


#### DEFINITION OF SPACES ####

P2 = ufl.VectorElement("Lagrange", msh.ufl_cell(), 2)
P1 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1)
V, Q = FunctionSpace(msh, P2), FunctionSpace(msh, P1)

# We define boundary conditions:

# No-slip boundary condition for velocity field (`V`) on boundaries
# where x = 0, x = 1, and y = 0
noslip = np.zeros(msh.geometry.dim, dtype=PETSc.ScalarType)
facets = locate_entities_boundary(msh, 1, noslip_boundary)
bc0 = dirichletbc(noslip, locate_dofs_topological(V, 1, facets), V)

# Driving velocity condition u = (1, 0) on top boundary (y = 1)
lid_velocity = Function(V)
lid_velocity.interpolate(lid_velocity_expression)
facets = locate_entities_boundary(msh, 1, lid)
bc1 = dirichletbc(lid_velocity, locate_dofs_topological(V, 1, facets))

# Collect Dirichlet boundary conditions
bcs = [bc0, bc1]

#### PLOT OF THE MESH ####

import pyvista
cells, types, geometry = plot.create_vtk_mesh(Q)
grid = pyvista.UnstructuredGrid(cells, types, geometry)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
plotter.show(screenshot='mesh_medium.png')

#### DEFINITION OF THE BILINEAR FORM ####

# We now define the bilinear and linear forms corresponding to the weak
# mixed formulation of the Stokes equations in a blocked structure:

# Define variational problem
(u, p) = ufl.TrialFunction(V), ufl.TrialFunction(Q)
(v, q) = ufl.TestFunction(V), ufl.TestFunction(Q)
f = Constant(msh, (PETSc.ScalarType(0), PETSc.ScalarType(0)))

a = form([[inner(grad(u), grad(v)) * dx, inner(p, div(v)) * dx],
          [inner(div(u), q) * dx, None]])
L = form([inner(f, v) * dx, inner(Constant(msh, PETSc.ScalarType(0)), q) * dx])

# We will use a block-diagonal preconditioner to solve this problem:

a_p11 = form(inner(p, q) * dx)
a_p = [[a[0][0], None],
       [None, a_p11]]

#### SOLVER ####

#### Nested matrix solver ####
# procedure adapted from Stokes TH Fenicsx tutorial #

# We now assemble the bilinear form into a nested matrix `A`, and call
# the `assemble()` method to communicate shared entries in parallel.
# Rows and columns in `A` that correspond to degrees-of-freedom with
# Dirichlet boundary conditions are zeroed and a value of 1 is set on
# the diagonal.

A = fem.petsc.assemble_matrix_nest(a, bcs=bcs)
A.assemble()

# We create a nested matrix `P` to use as the preconditioner. The
# top-left block of `P` is shared with the top-left block of `A`. The
# bottom-right diagonal entry is assembled from the form `a_p11`:

P11 = fem.petsc.assemble_matrix(a_p11, [])
P = PETSc.Mat().createNest([[A.getNestSubMatrix(0, 0), None], [None, P11]])
P.assemble()

# Next, the right-hand side vector is assembled and then modified to
# account for non-homogeneous Dirichlet boundary conditions:

# +
b = fem.petsc.assemble_vector_nest(L)
fem.petsc.apply_lifting_nest(b, a, bcs=bcs)

for b_sub in b.getNestSubVecs():
    b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

# Set Dirichlet boundary condition values in the RHS
bcs0 = fem.bcs_by_block(extract_function_spaces(L), bcs)
fem.petsc.set_bc_nest(b, bcs0)
# -

# The pressure field for this problem is determined only up to a
# constant. We can supply the vector that spans the nullspace and any
# component of the solution in this direction will be eliminated during
# the iterative linear solution process.

# Create nullspace vector
null_vec = fem.petsc.create_vector_nest(L)

# Set velocity part to zero and the pressure part to a non-zero constant
null_vecs = null_vec.getNestSubVecs()
null_vecs[0].set(0.0), null_vecs[1].set(1.0)

# Normalize the vector, create a nullspace object, and attach it to the
# matrix
null_vec.normalize()
nsp = PETSc.NullSpace().create(vectors=[null_vec])
assert nsp.test(A)
A.setNullSpace(nsp)

# Krylov Subspace Solver `ksp`
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A, P)
ksp.setType("minres")
ksp.setTolerances(rtol=1e-9)
ksp.getPC().setType("fieldsplit")
ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

# Define the matrix blocks in the preconditioner with the velocity and
# pressure matrix index sets
nested_IS = P.getNestISs()
ksp.getPC().setFieldSplitIS(
    ("u", nested_IS[0][0]),
    ("p", nested_IS[0][1]))

# Set the preconditioners for each block
ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
ksp_u.setType("preonly")
ksp_u.getPC().setType("gamg")
ksp_p.setType("preonly")
ksp_p.getPC().setType("jacobi")
ksp.setFromOptions()


# Set the solution function 
u, p = Function(V), Function(Q)
x = PETSc.Vec().createNest([_cpp.la.petsc.create_vector_wrap(u.x), _cpp.la.petsc.create_vector_wrap(p.x)])
ksp.solve(b, x)

# Norms of the solution vectors are computed:
ux=u.sub(0)
uy=u.sub(1)
# print(ux.vector().get_local())  

norm_u_0 = u.x.norm()
norm_p_0 = p.x.norm()
if MPI.COMM_WORLD.rank == 0:
    print("(A) Norm of velocity coefficient vector (nested, iterative): {}".format(norm_u_0))
    print("(A) Norm of pressure coefficient vector (nested, iterative): {}".format(norm_p_0))

#plot of the nested matrix solver
################ PLOTTING THE PRESSURE ###################################################
import pyvista
cells, types, geometry = plot.create_vtk_mesh(Q)
grid = pyvista.UnstructuredGrid(cells, types, geometry)
grid.point_data["p"] = p.x.array.real
grid.set_active_scalars("p")
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=False)
plotter.view_xy()
plotter.show(screenshot='pressure.png')


#### SAVE SOLUTIONS ####
with XDMFFile(MPI.COMM_WORLD, "out_stokes/velocity_unif02.xdmf", "w") as ufile_xdmf:
    u.x.scatter_forward()
    ufile_xdmf.write_mesh(msh)
    ufile_xdmf.write_function(u)

with XDMFFile(MPI.COMM_WORLD, "out_stokes/pressure_unif02.xdmf", "w") as pfile_xdmf:
    p.x.scatter_forward()
    pfile_xdmf.write_mesh(msh)
    pfile_xdmf.write_function(p)
    
# File("u.pvd")<<u
    

#### Monolithic block iterative solver ####
# procedure adapted from Stokes TH Fenicsx tutorial #


# Next, we solve same problem, but now with monolithic (non-nested)
# matrices and iterative solvers.
A = fem.petsc.assemble_matrix_block(a, bcs=bcs)
A.assemble()
P = fem.petsc.assemble_matrix_block(a_p, bcs=bcs)
P.assemble()
b = fem.petsc.assemble_vector_block(L, a, bcs=bcs)

# Set near nullspace for pressure
null_vec = A.createVecLeft()
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
null_vec.array[offset:] = 1.0
null_vec.normalize()
nsp = PETSc.NullSpace().create(vectors=[null_vec])
assert nsp.test(A)
A.setNullSpace(nsp)

# Build IndexSets for each field (global dof indices for each field)
V_map = V.dofmap.index_map
Q_map = Q.dofmap.index_map
offset_u = V_map.local_range[0] * V.dofmap.index_map_bs + Q_map.local_range[0]
offset_p = offset_u + V_map.size_local * V.dofmap.index_map_bs
is_u = PETSc.IS().createStride(V_map.size_local * V.dofmap.index_map_bs, offset_u, 1, comm=PETSc.COMM_SELF)
is_p = PETSc.IS().createStride(Q_map.size_local, offset_p, 1, comm=PETSc.COMM_SELF)

# Create Krylov solver
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A, P)
ksp.setTolerances(rtol=1e-9)
ksp.setType("minres")
ksp.getPC().setType("fieldsplit")
ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
ksp.getPC().setFieldSplitIS(
    ("u", is_u),
    ("p", is_p))

# Configure velocity and pressure sub KSPs
ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
ksp_u.setType("preonly")
ksp_u.getPC().setType("gamg")
ksp_p.setType("preonly")
ksp_p.getPC().setType("jacobi")

# Monitor the convergence of the KSP
opts = PETSc.Options()
opts["ksp_monitor"] = None
opts["ksp_view"] = None
ksp.setFromOptions()

# Compute solution
x = A.createVecRight()
ksp.solve(b, x)

# Create Functions and scatter x solution
u, p = Function(V), Function(Q)
offset = V_map.size_local * V.dofmap.index_map_bs
u.x.array[:offset] = x.array_r[:offset]
p.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]

# print(u.x.array)
# We can calculate the $L^2$ norms of u and p as follows:
norm_u_1 = u.x.norm()
norm_p_1 = p.x.norm()
if MPI.COMM_WORLD.rank == 0:
    print("(B) Norm of velocity coefficient vector (blocked, iterative): {}".format(norm_u_1))
    print("(B) Norm of pressure coefficient vector (blocked, iterative): {}".format(norm_p_1))
assert np.isclose(norm_u_1, norm_u_0)
assert np.isclose(norm_p_1, norm_p_0)
