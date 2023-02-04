# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 00:11:56 2022

@author: danie
"""

# +
import numpy as np

import dolfinx
print(dolfinx.__version__, dolfinx.git_commit_hash)

import petsc4py
print(petsc4py.__version__)

import ufl
from dolfinx import cpp as _cpp
from dolfinx import fem, plot, geometry
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

import matplotlib.pyplot as plt
from petsc4py.PETSc import ScalarType

#definitions of the vectors for error computation
n_refinements = 4 # number of mesh refiments
maxh = np.zeros((n_refinements,1)) # maximum mesh size array
L2_error_gradu = np.zeros((n_refinements,1)) 
L2_error_p = np.zeros((n_refinements,1)) 

#MESH DEFINITION - CLASSIC
# msh = create_rectangle(MPI.COMM_WORLD,
#                         [np.array([0, 0]), np.array([1, 1])],
#                         [32, 32],
#                         CellType.triangle, GhostMode.none)

######## REFERENCE SOLUTION COMPUTATION - FINE LOCALLY REFINED MESH #############
gmsh.initialize()
gmsh.model.add("reference")
fine_dim = 0.01
coarse_dim= 0.01
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

msh_ref, cell_markers_ref, facet_markers_ref = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0, gdim=2)
x_ref = SpatialCoordinate(msh_ref)
############################################################################

def noslip_boundary(x):
    return np.logical_or(np.logical_or(np.isclose(x[0], 0.0),
                                       np.isclose(x[0], 1.0)),
                         np.isclose(x[1], 0.0))
def lid(x):
    return np.isclose(x[1], 1.0)
def lid_velocity_expression(x):
    return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))


P2 = ufl.VectorElement("Lagrange", msh_ref.ufl_cell(), 2)
P1 = ufl.FiniteElement("DG", msh_ref.ufl_cell(), 0)
V_ref, Q_ref = FunctionSpace(msh_ref, P2), FunctionSpace(msh_ref, P1)

noslip = np.zeros(msh_ref.geometry.dim, dtype=PETSc.ScalarType)
facets = locate_entities_boundary(msh_ref, 1, noslip_boundary)
bc0 = dirichletbc(noslip, locate_dofs_topological(V_ref, 1, facets), V_ref)

lid_velocity = Function(V_ref)
lid_velocity.interpolate(lid_velocity_expression)
facets = locate_entities_boundary(msh_ref, 1, lid)
bc1 = dirichletbc(lid_velocity, locate_dofs_topological(V_ref, 1, facets))

bcs = [bc0, bc1]

# Define variational problem
(u, p) = ufl.TrialFunction(V_ref), ufl.TrialFunction(Q_ref)
(v, q) = ufl.TestFunction(V_ref), ufl.TestFunction(Q_ref)
f = Constant(msh_ref, (PETSc.ScalarType(0), PETSc.ScalarType(0)))

a = form([[inner(grad(u), grad(v)) * dx, inner(p, div(v)) * dx],
          [inner(div(u), q) * dx, None]])
L = form([inner(f, v) * dx, inner(Constant(msh_ref, PETSc.ScalarType(0)), q) * dx])
# -

# We will use a block-diagonal preconditioner to solve this problem:

a_p11 = form(inner(p, q) * dx)
a_p = [[a[0][0], None],
       [None, a_p11]]


A = fem.petsc.assemble_matrix_nest(a, bcs=bcs)
A.assemble()
P11 = fem.petsc.assemble_matrix(a_p11, [])
P = PETSc.Mat().createNest([[A.getNestSubMatrix(0, 0), None], [None, P11]])
P.assemble()
b = fem.petsc.assemble_vector_nest(L)
fem.petsc.apply_lifting_nest(b, a, bcs=bcs)
for b_sub in b.getNestSubVecs():
    b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
# Set Dirichlet boundary condition values in the RHS
bcs0 = fem.bcs_by_block(extract_function_spaces(L), bcs)
fem.petsc.set_bc_nest(b, bcs0)
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
# +
ksp = PETSc.KSP().create(msh_ref.comm)
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

# Monitor the convergence of the KSP
ksp.setFromOptions()
# -

# To compute the solution, we create finite element {py:class}`Function
# <dolfinx.fem.Function>` for the velocity (on the space `V`) and
# for the pressure (on the space `Q`). The vectors for `u` and `p` are
# combined to form a nested vector and the system is solved:

u_ref, p_ref = Function(V_ref), Function(Q_ref)
x = PETSc.Vec().createNest([_cpp.la.petsc.create_vector_wrap(u_ref.x), _cpp.la.petsc.create_vector_wrap(p_ref.x)])
ksp.solve(b, x)

norm_u_0 = u_ref.x.norm()
norm_p_0 = p_ref.x.norm()
if MPI.COMM_WORLD.rank == 0:
    print("(A) Norm of velocity coefficient vector (nested, iterative): {}".format(norm_u_0))
    print("(A) Norm of pressure coefficient vector (nested, iterative): {}".format(norm_p_0))

    meshsize= 0.1
    
    #used for mesh refinement
    # meshsize= [0.03, 0.02, 0.015]
    # j=0;  

### SAVE REFERENCE SOLUTION ###

###### SAVE SOLUTIONS #############
with XDMFFile(MPI.COMM_WORLD, "out_P2P0/velocity_P2O0.xdmf", "w") as ufile_xdmf:
    u_ref.x.scatter_forward()
    ufile_xdmf.write_mesh(msh_ref)
    ufile_xdmf.write_function(u_ref)

with XDMFFile(MPI.COMM_WORLD, "out_P2P0/pressure_P2P0 .xdmf", "w") as pfile_xdmf:
    p_ref.x.scatter_forward()
    pfile_xdmf.write_mesh(msh_ref)
    pfile_xdmf.write_function(p_ref)

##################### MESH DEFINITION - UNIFORM ############################
for mesh_iterator in range(0,n_refinements) : # mesh refinements

    
    gmsh.initialize()
    gmsh.model.add("refined")
    fine_dim = meshsize
    coarse_dim= meshsize
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
    
    # We define two {py:class}`FunctionSpace <dolfinx.fem.FunctionSpace>`
    # instances with different finite elements. `P2` corresponds to a continuous
    # piecewise quadratic basis for the velocity field and `P1` to a continuous
    # piecewise linear basis for the pressure field:
    
    
    P2 = ufl.VectorElement("Lagrange", msh.ufl_cell(), 2)
    P1 = ufl.FiniteElement("DG", msh.ufl_cell(), 0)
    V, Q = FunctionSpace(msh, P2), FunctionSpace(msh, P1)
    
    # We define boundary conditions:
    
    # +
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
    
    #PLOT OF THE MESH
    # import pyvista
    # cells, types, geometry = plot.create_vtk_mesh(Q)
    # grid = pyvista.UnstructuredGrid(cells, types, geometry)
    # plotter = pyvista.Plotter()
    # plotter.add_mesh(grid, show_edges=True)
    # plotter.view_xy()
    # plotter.show(screenshot='mesh_medium.png')
    
    # We now define the bilinear and linear forms corresponding to the weak
    # mixed formulation of the Stokes equations in a blocked structure:
    
    # +
    # Define variational problem
    (u, p) = ufl.TrialFunction(V), ufl.TrialFunction(Q)
    (v, q) = ufl.TestFunction(V), ufl.TestFunction(Q)
    f = Constant(msh, (PETSc.ScalarType(0), PETSc.ScalarType(0)))
    
    a = form([[inner(grad(u), grad(v)) * dx, inner(p, div(v)) * dx],
              [inner(div(u), q) * dx, None]])
    L = form([inner(f, v) * dx, inner(Constant(msh, PETSc.ScalarType(0)), q) * dx])
    # -
    
    # We will use a block-diagonal preconditioner to solve this problem:
    
    a_p11 = form(inner(p, q) * dx)
    a_p = [[a[0][0], None],
           [None, a_p11]]
    
    # ### Nested matrix solver
    #
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
    
    # Modify ('lift') the RHS for Dirichlet boundary conditions
    fem.petsc.apply_lifting_nest(b, a, bcs=bcs)
    
    # Sum contributions from ghost entries on the owner
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
    
    # +
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
    # -
    
    # Now we create a Krylov Subspace Solver `ksp`. We configure it to use
    # the MINRES method, and a block-diagonal preconditioner using PETSc's
    # additive fieldsplit type preconditioner:
    
    # +
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
    
    # Monitor the convergence of the KSP
    ksp.setFromOptions()
    # -
    
    # To compute the solution, we create finite element {py:class}`Function
    # <dolfinx.fem.Function>` for the velocity (on the space `V`) and
    # for the pressure (on the space `Q`). The vectors for `u` and `p` are
    # combined to form a nested vector and the system is solved:
    
    uh_coarse, ph_coarse = Function(V), Function(Q)
    x = PETSc.Vec().createNest([_cpp.la.petsc.create_vector_wrap(uh_coarse.x), _cpp.la.petsc.create_vector_wrap(ph_coarse.x)])
    ksp.solve(b, x)
    
    # Norms of the solution vectors are computed:
    
    norm_u_0 = uh_coarse.x.norm()
    norm_p_0 = ph_coarse.x.norm()
    if MPI.COMM_WORLD.rank == 0:
        print("(A) Norm of velocity coefficient vector (nested, iterative): {}".format(norm_u_0))
        print("(A) Norm of pressure coefficient vector (nested, iterative): {}".format(norm_p_0))
    
    #INTERPOLATE COARSE SOLUTION TO LOCALLY REFINED MESH
    
    ################ INTERPOLATING COARSE TO FINE ##############################################
    ph_fine = fem.Function(Q_ref, dtype = ScalarType)
    ndofs_ref = len(ph_fine.x.array)

    # Create bounding box for function evaluation #CERCA DI CAPIRE MEGLIO COME FUNZIONA ESATTAMENTE QUESTO
    bb_tree = geometry.BoundingBoxTree(msh, 2)

    # Check against standard table value  
    dof_vertex_coordinates = Q_ref.tabulate_dof_coordinates()
    cell_candidates = geometry.compute_collisions(bb_tree, dof_vertex_coordinates)
    cells = geometry.compute_colliding_cells(msh, cell_candidates, dof_vertex_coordinates)

    #QUI STAI ESSENZIALMENTE INTERPOLANDO LA SOLUZIONE COARSE IN UNA MESH FINE COME QUELLA UTILIZZATA
    #PER IL RIFERIMENTO, COSI' DA POTER POI VALUTARE L'ERRORE COME LA DIFFERENZA TRA COARSE E FINE MESH 
    cell_of_dof = np.ndarray(ndofs_ref)
    for i in range(0,len(cells)) :
        cell_of_dof[i] = cells.links(i)[0]

    ph_fine.x.array[:] = (ph_coarse.eval(dof_vertex_coordinates,cell_of_dof)).flatten()

        
    ################ COMPUTING THE ERROR ###########################################################
    maxh[mesh_iterator] = meshsize

    error_p_h = fem.assemble_scalar(fem.form(inner(p_ref-ph_fine, p_ref-ph_fine)* dx))
    norm_p = fem.assemble_scalar(fem.form( p_ref**2 * dx))

    # # error over all parallel processes
    L2_error_p[mesh_iterator] = np.sqrt(msh.comm.allreduce(error_p_h/norm_p, op=MPI.SUM))

    meshsize = 0.5*meshsize
    # gmsh.model.mesh.refine() # Applies uniform refinement to the mesh

    if mesh_iterator > 0:
        print('Convergence rate in L2 norm for pressureat refinement step ',mesh_iterator,
            ' for the Taylor-Hood element is ',
            (np.log(L2_error_p[mesh_iterator]) - np.log(L2_error_p[mesh_iterator-1]) )/  \
            (np.log(maxh[mesh_iterator]) - np.log(maxh[mesh_iterator-1]) ) )
    ################################################################################################


plt.loglog(maxh[:], L2_error_p[:], label='$||p - p_h||$', marker='.', markersize='8.0', linestyle='-')
plt.loglog(maxh[:], maxh[:], '--', label= 'order 1')
gmsh.finalize()
plt.xlabel('$h$')
plt.ylabel('Error L2 norm')
plt.grid(True)
plt.suptitle('Stokes Problem, $\mathbb{P}^2$-$\mathbb{P}^0$ element')
plt.legend()
plt.show()
plt.savefig('P2P0_cconvergence.png')

print('Computation is over! Success!')