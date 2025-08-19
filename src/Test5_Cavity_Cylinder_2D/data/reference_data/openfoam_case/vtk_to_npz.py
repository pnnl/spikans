#!/usr/bin/env python3
"""
Convert OpenFOAM VTK output to NPZ format for comparison with PINN results.
"""

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import os

def read_vtm_data(vtm_file):
    """
    Read VTM file and extract mesh coordinates and field data.
    """
    # Create reader for VTM file
    reader = vtk.vtkXMLMultiBlockDataReader()
    reader.SetFileName(vtm_file)
    reader.Update()
    
    # Get the output
    output = reader.GetOutput()
    
    # Initialize lists to store data
    all_points = []
    all_u = []
    all_v = []
    all_p = []
    
    # Iterate through blocks
    for i in range(output.GetNumberOfBlocks()):
        block = output.GetBlock(i)
        
        if block and block.GetNumberOfPoints() > 0:
            # Get points
            points = vtk_to_numpy(block.GetPoints().GetData())
            all_points.append(points)
            
            # Get velocity field
            if block.GetPointData().GetArray('U'):
                U_array = vtk_to_numpy(block.GetPointData().GetArray('U'))
                all_u.append(U_array[:, 0])  # x-component
                all_v.append(U_array[:, 1])  # y-component
            
            # Get pressure field
            if block.GetPointData().GetArray('p'):
                p_array = vtk_to_numpy(block.GetPointData().GetArray('p'))
                all_p.append(p_array)
    
    # Concatenate all data
    points = np.vstack(all_points)
    u = np.concatenate(all_u)
    v = np.concatenate(all_v)
    p = np.concatenate(all_p)
    
    # Calculate velocity magnitude
    umag = np.sqrt(u**2 + v**2)
    
    return points, u, v, p, umag

def create_structured_grid(points, u, v, p, umag, nx=200, ny=200):
    """
    Interpolate scattered data onto a structured grid for easier comparison with PINN.
    """
    from scipy.interpolate import griddata
    
    # Extract x, y coordinates (ignore z)
    x = points[:, 0]
    y = points[:, 1]
    
    # Create structured grid
    xi = np.linspace(0, 1, nx)
    yi = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(xi, yi)
    
    # Interpolate fields onto structured grid
    U = griddata((x, y), u, (X, Y), method='linear')
    V = griddata((x, y), v, (X, Y), method='linear')
    P = griddata((x, y), p, (X, Y), method='linear')
    UMAG = griddata((x, y), umag, (X, Y), method='linear')
    
    return X, Y, U, V, P, UMAG

def main():
    # Input VTM file
    vtm_file = "output_test.vtm"
    
    if not os.path.exists(vtm_file):
        print(f"Error: {vtm_file} not found!")
        return
    
    print(f"Reading VTM file: {vtm_file}")
    
    # Read VTK data
    points, u, v, p, umag = read_vtm_data(vtm_file)
    
    print(f"Number of points: {len(points)}")
    print(f"u range: [{u.min():.4f}, {u.max():.4f}]")
    print(f"v range: [{v.min():.4f}, {v.max():.4f}]")
    print(f"p range: [{p.min():.4f}, {p.max():.4f}]")
    print(f"umag range: [{umag.min():.4f}, {umag.max():.4f}]")
    
    # Create structured grid
    print("\nInterpolating to structured grid...")
    X, Y, U, V, P, UMAG = create_structured_grid(points, u, v, p, umag)
    
    # Create output directory
    output_dir = "../data/reference_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as NPZ
    output_file = os.path.join(output_dir, "openfoam_cavity_cylinder_Re100.npz")
    
    # Save data in a format similar to PINN output
    data_dict = {
        'X_mesh': X,
        'Y_mesh': Y,
        'U': U,
        'V': V,
        'P': P,
        'UMAG': UMAG,
        'Re': 100.0,
        'cylinder_center': (0.5, 0.5),
        'cylinder_radius': 0.2,
        'points_raw': points,  # Raw point cloud data
        'u_raw': u,
        'v_raw': v,
        'p_raw': p,
        'umag_raw': umag
    }
    
    np.savez(output_file, **data_dict)
    print(f"\nData saved to: {output_file}")
    
    # Print info about the saved data
    print(f"\nStructured grid size: {X.shape}")
    print(f"Fields included: X_mesh, Y_mesh, U, V, P, UMAG")
    print(f"Raw data also included with '_raw' suffix")

if __name__ == "__main__":
    main()