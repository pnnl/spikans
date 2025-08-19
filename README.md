# SPIKANs: Separable Physics-Informed Kolmogorov-Arnold Networks

## Overview

**SPIKANs (Separable Physics-Informed Kolmogorov-Arnold Networks)** is a novel neural network architecture that combines the power of Physics-Informed Neural Networks (PINNs) with the efficiency of separable representations and the interpretability of Kolmogorov-Arnold Networks (KANs).

### Key Innovation

Traditional PIKANs (Physics-Informed Kolmogorov-Arnold Networks) suffer from the **curse of dimensionality** - requiring O(N^d) collocation points for d-dimensional problems. SPIKANs alleviates this by decomposing multi-dimensional PDEs into **separable components**, reducing computational complexity from **O(N^d) to O(N)** while maintaining accuracy.

### Performance Gains

- $O(N^d) \rightarrow O(Nd)$ speedup over traditional PIKANs
- Superior scalability to high-dimensional problems  
- Comparable or better accuracy with fewer parameters
- Reduced memory footprint

## Architecture

Instead of one KAN processing all dimensions simultaneously:
```
Traditional PIKAN: u(x,y,t) → Single KAN
```

SPIKANs use separable representation. For example, for 2D+1 problems:
```
SPIKAN: u(x,y,t) ≈ Σᵣ fₓ(x) ⊗ fᵧ(y) ⊗ fₜ(t)
```

## Installation

### Prerequisites
- Python 3.11.5+
- CUDA-compatible GPU (recommended)

### Setup
```bash
# Clone the repository
git clone https://github.com/pnnl/spikans
cd spikans

# Install dependencies
pip install -r requirements.txt
```


## Test Cases & Examples

The repository includes 5 test cases demonstrating SPIKAN capabilities:

### Test 1: 2D Helmholtz Equation
**Location**: `src/Test1_Helmholtz_2D/`

**Governing Equation:**
$$\nabla^2 u + \kappa^2 u = q(x,y) \quad \text{on } \Omega = [-1,1]^2$$
$$u = 0 \quad \text{on } \partial\Omega $$

Elliptic PDE benchmark with manufactured solution $u(x,y) = \sin(\pi x)\sin(4\pi y)$
- **Parameters**: $\kappa = 1$
- **Grid Sizes**: $100^2$, $200^2$
- **Files**: `helmoltz_spikan.ipynb`, `helmoltz_pikan.ipynb`
- **Validation**: Against manufactured solution

### Test 2: 2D Navier-Stokes (Lid-Driven Cavity)
**Location**: `src/Test2_NavierStokes_2D/`

**Governing Equations:**
$$\nabla \cdot \mathbf{u} = 0$$
$$\mathbf{u} \cdot \nabla \mathbf{u} + \nabla p = \frac{1}{Re}\nabla^2 \mathbf{u}$$

Fluid dynamics benchmark with velocity and pressure outputs
- **Boundary Conditions**: Moving lid ($u=1$ at top), no-slip walls
- **Reynolds Numbers**: $Re=100, 400$  
- **Grid Sizes**: $50^2$, $100^2$
- **Validation**: Against high-resolution finite volume reference data

### Test 3: 1D+1 Allen-Cahn Equation  
**Location**: `src/Test3_AllenCahn_2D/`

**Governing Equation:**
$$\frac{\partial u}{\partial t} - D\frac{\partial^2 u}{\partial x^2} + 5(u^3 - u) = 0$$

Challenging phase-field dynamics problem
- **Parameters**: $D = 1 \times 10^{-4}$, domain: $x \in [-1,1]$, $t \in [0,1]$
- **Initial Condition**: $u(x,0) = x^2\cos(\pi x)$
- **Boundary Conditions**: $u(-1,t) = u(1,t) = -1$
- **Validation**: Against Fourier pseudospectral reference solution

### Test 4: 2D+1 Klein-Gordon Equation
**Location**: `src/Test4_KleinGordon_3D/`

**Governing Equation:**
$$\frac{\partial^2 u}{\partial t^2} - \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) + u^2 = h(x,y,t)$$

Time-dependent wave equation with nonlinearity
- **Domain**: $[0,1]^2 \times [0,10]$
- **Manufactured Solution**: $u(x,y,t) = (x+y)\cos(t) + xy \sin(t)$
- **Forcing Term**: $h(x,y,t) = u^2 - u$
- **Grid Sizes**: $50^3$, $100^3$, $150^3$, $200^3$
- **Validation**: Against manufactured solution 

### Test 5: 2D Cavity Flow with Immersed Cylinder
**Location**: `src/Test5_Cavity_Cylinder_2D/`

**Governing Equations:** Same as in Test 2, but the presence of an immersed cylinder at the center of the cavity illustrates how SPIKANs are able to handle complex geometries with  using masking for interior points and point-wise evaluation on the cylinder boundary.
- **Geometry**: Square cavity $[0,1]^2$ with circular cylinder (center: $(0.5,0.5)$, radius: $0.2$)
- **Boundary Conditions**: No-slip on cylinder surface, lid-driven cavity
- **Reynolds Numbers**: $Re=100, 1000$
- **Validation**: Against finite volume reference solution
- **Files**: `cavity_cylinder_spikan.py`, `cavity_cylinder_pikan.py`


## Quick Start

### 2D Helmholtz example of SPIKAN in a Jupyter notebook:

```python
# Example: 2D Helmholtz equation
cd src/Test1_Helmholtz_2D/
jupyter notebook helmoltz_spikan.ipynb
```

### Command-Line Usage (Advanced Tests)

```bash
# Cavity with cylinder - parameter sweep
cd src/Test5_Cavity_Cylinder_2D/
python cavity_cylinder_spikan.py --Re 100 --nx 400 --ny 400 --epochs 200000
```

### Key Parameters
- `--nx, --ny`: Grid resolution  
- `--epochs`: Training iterations
- `--Re`: Reynolds number
- `--r`: Rank (latent dimension)
- `--k`: B-spline degree

## Core Implementation

### Base Classes
- **`KAN`**: Foundation KAN implementation (in `KAN.py`)
- **`SF_KAN_Separable`**: Separable extension implemented in each test case

### Key Files
- **`KAN.py`**: Core KAN implementation with B-spline layers
- **`KANLayer.py`**: Individual KAN layer implementation  
- **`KANWrapper.py`**: Network wrapper classes
- **`splines.py`**: B-spline basis functions
- **`general.py`**: Utility functions

### Separable Forward Pass
```python
# Compute predictions from each dimension
preds_x = model_x.apply(variables_x, x[:, None])  # Shape: (nx, out_size*r)
preds_y = model_y.apply(variables_y, y[:, None])  # Shape: (ny, out_size*r)

# Reshape and combine via tensor product
preds_x = preds_x.reshape(-1, out_size, r)
preds_y = preds_y.reshape(-1, out_size, r)
preds = jnp.einsum('ijk,ljk->ilj', preds_x, preds_y)  # Separable combination
```

## Performance Comparison

![Computational Time Comparison](comparison_computational_time.png)

SPIKANs demonstrate consistent speedups across all test cases, with computational time improvements ranging from $8\times$ to $287\times$ compared to traditional PIKANs while maintaining or improving solution accuracy.


## Academic Contributions
- Novel separable KAN architecture for physics-informed learning
- Reduction of computational complexity of Physics-informed KANs from $O(N^d)$ to $O(N)$ while preserving or improving accuracy
- Comprehensive benchmark suite across different PDE types
- Extension to complex geometries via point masking


## Citation

If you use this code in your research, please cite:

```bibtex
@article{jacob2025spikans,
  title={SPIKANs: Separable Physics-Informed Kolmogorov-Arnold Networks},
  author={Jacob, B. and Howard, A. A. and Stinis, P.},
  journal={Machine Learning: Science and Technology},
  year={2025}
}
```

---

## DISCLAIMER

This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or
any information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.
 
Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.
 
                 PACIFIC NORTHWEST NATIONAL LABORATORY
                              operated by
                                BATTELLE
                                for the
                   UNITED STATES DEPARTMENT OF ENERGY
                    under Contract DE-AC05-76RL01830

## LICENSE

Copyright Battelle Memorial Institute 2025
 
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 
1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
 
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
 
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
**Contact**: bruno.jacob@pnnl.gov, amanda.howard@pnnl.gov, panagiotis.stinis@pnnl.gov
