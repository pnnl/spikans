# OpenFOAM Case: Lid-Driven Cavity with Cylinder

This is an OpenFOAM case for simulating 2D lid-driven cavity flow with a cylindrical obstacle.

## Case Parameters
- Reynolds number: Re = 100
- Cavity dimensions: 1m x 1m
- Cylinder center: (0.5, 0.5)
- Cylinder radius: 0.2m
- Lid velocity: 1 m/s
- Kinematic viscosity: ν = 0.01 m²/s

## Directory Structure
```
openfoam_case/
├── 0/                  # Initial conditions
│   ├── U              # Velocity field
│   └── p              # Pressure field
├── constant/          # Physical properties
│   └── transportProperties
├── system/            # Numerical settings
│   ├── blockMeshDict
│   ├── controlDict
│   ├── fvSchemes
│   ├── fvSolution
│   └── snappyHexMeshDict
├── Allrun            # Run script
└── Allclean         # Clean script
```

## How to Run

1. Make sure OpenFOAM is loaded:
   ```bash
   source $HOME/OpenFOAM/OpenFOAM-v2312/etc/bashrc
   ```

2. Clean any previous runs:
   ```bash
   ./Allclean
   ```

3. Run the case:
   ```bash
   ./Allrun
   ```

## Manual Steps (if Allrun doesn't work)

1. Generate the base mesh:
   ```bash
   blockMesh
   ```

2. Create the mesh with cylinder cutout:
   ```bash
   snappyHexMesh -overwrite
   ```

3. Check mesh quality:
   ```bash
   checkMesh
   ```

4. Run the solver:
   ```bash
   simpleFoam
   ```

## Post-Processing

Use ParaView to visualize the results:
```bash
paraFoam
```

Key results to examine:
- Velocity magnitude contours
- Streamlines
- Pressure distribution
- Vorticity field

## Notes
- The mesh uses 100x100 base resolution with refinement near the cylinder
- The cylinder is defined analytically using `searchableCylinder` in snappyHexMeshDict (no STL file needed)
- The solver runs for 1000 iterations (steady-state convergence)
- **Only the final converged solution is saved** (at iteration 1000)
- The case uses simpleFoam (steady-state incompressible laminar solver)
- You'll only see folders: 0 (initial) and 1000 (final steady-state solution)