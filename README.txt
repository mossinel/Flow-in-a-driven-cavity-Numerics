##### NUMERICS FOR FLUIDS STRUCTURE AND ELECTROMAGNETS #####

The folder contains code and report for the Project 1 - Lid Cavity Problem.

# Solver #

Contains code for Taylor Hood with:
- locally refined meshes generated with gmesh
- globally refined meshes generated with standard procedure 
- nested iterative solver
- block iterative solver

Contains code for Mortar Method implementation:
- mesh generation in gmesh
- multiphenicsx for sub domains management

Output includes velocity and pressure for different refinements.

# Convergence #

Contains code for convergence study for P1-PMINI, P1-P2 and P2-P0

# Burggraff Reference Comparison # 

Contains code for comparison between the results of a locally refined mesh evaluated at the mesh vertical centerline and considering only horizontal velocity (data extracted using ParaView plot over line filter and SpreadSheet View) and the manually extracted reference points cited in the references of the project.

# Plots #

Contains selected plots included in the report