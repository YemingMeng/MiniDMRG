# MiniDMRG

MiniDMRG is a lightweight, educational MPS/MPO DMRG implementation in Python (using NumPy), achieved in under 1000 lines of code and aimed at clarity by abstracting low-level tensor manipulations.

## Features

* **`TensorNode` Class (`tensor.py`)**
    * A fundamental building block representing a single tensor, built upon NumPy arrays.
    * Supports labeling and relabeling of tensor legs (indices), allowing for intuitive tensor network operations.
    * Abstracts away the underlying tensor contraction details, simplifying the implementation of higher-level algorithms.

* **`TensorChain` Class (`tensor.py`)**
    * Represents a chain of connected `TensorNode` objects for MPS.
    * Provides convenient methods for converting the chain into left or right canonical forms (`left_canonical`, `right_canonical`).

* **Intuitive Chain Construction**
    * The `<<` operator is overloaded to easily connect `TensorNode` instances sequentially into a `TensorChain`.
    * This is achieved by automatically identifying common leg labels between adjacent nodes for contraction.

* **MPO Construction (`mpo.py`)**
    * `OperatorNode`: A specialized `TensorNode` for representing MPO tensors.
    * `LocalHamiltonianChain`: A helper class to easily construct MPOs for 1D Hamiltonians with nearest-neighbor and on-site terms (e.g., spin chains). Define operators (`define_operator`), add interactions (`add_neighbor_interaction`, `add_onsite_interaction`), and generate the MPO chain (`gen_data`).

* **Finite DMRG Algorithm (`dmrg.py`)**
    * Implementation of the standard finite-system DMRG algorithm for finding the ground state energy and MPS representation of a given Hamiltonian (MPO).
    * Uses single-site updates (`right_sweep`, `left_sweep`).
    * Employs `scipy.sparse.linalg.eigsh` for efficient eigenvalue calculation of the effective Hamiltonian.

* **Lazy Environment Calculation (`dmrg.py`)**
    * The left and right environment tensors needed during DMRG sweeps are calculated recursively and cached (`ltensor`, `rtensor`). This avoids redundant calculations, especially when sweeping back and forth.

## Requirements

* Python 3.x
* NumPy
* SciPy (`scipy.sparse.linalg.eigsh` is used in the DMRG solver)

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/YemingMeng/MiniDMRG.git
    cd MiniDMRG
    ```
2.  Install the required libraries:
    ```bash
    pip install numpy scipy
    ```

## Usage

The `example.py` file provides several examples demonstrating how to use the different components of the library. To run all examples:

```bash
python example.py
```

Below are snippets illustrating each main feature:

**1. Basic Tensor Operations (`example1`)**
   Creating `TensorNode`, labeling legs, contracting tensors (`*`), and connecting nodes into a `TensorChain` (`<<`).

   ```python
# --- example1: Basic tensor node and tensor chain usage ---
from tensor import Leg, TensorNode, TensorChain, gen_random_node
import numpy as np

# Define legs with dimensions and unique labels
legs1 = [Leg(dimension=2, label=0),
        Leg(dimension=2, label=1),
        Leg(dimension=4, label=2)]
legs2 = [Leg(dimension=4, label=2), # Common label '2' for contraction
        Leg(dimension=8, label=4)]

# Create random tensor nodes
t1 = gen_random_node(legs1)
t2 = gen_random_node(legs2)

# Contract tensors along the common leg '2'
t3 = t1 * t2
print('Contracted tensor legs:', t3.legs)

# Connect nodes into a TensorChain using '<<'
# This automatically identifies the common leg '2' for the connection
tc1 = t1 << t2
print('TensorChain legs:', tc1.legs) # Legs of the simplified chain

# Verify data consistency between contraction and chain simplification
print('Data consistent:', np.allclose(t3.data, tc1.simplify.data))
   ```

**2. MPS Creation (`example2`)**
   Generating a random Matrix Product State (MPS) for a given number of sites, local Hilbert space dimension, and bond dimension truncation.

   ```python
# --- example2: Basic MPS usage ---
from mps import random_tensor_chain

lhd = 2       # Local Hilbert space dimension (e.g., spin-1/2)
nsite = 20    # Number of sites in the chain
bond_dim = 40 # Maximum bond dimension

# Generate a random MPS TensorChain
mps_chain = random_tensor_chain(lhd, nsite, truncation=bond_dim)

# Optionally, convert to right canonical form
mps_chain.right_canonical()

print('MPS Tensor dimensions:', [t.dims for t in mps_chain])
   ```

**3. MPO Definition (`example3`)**
   Manually defining `OperatorNode` instances for constructing a Matrix Product Operator (MPO). This example shows nodes for a specific Heisenberg MPO representation.

   ```python
# --- example3: Basic MPO definition ---
from mpo import OperatorNode
import numpy as np

# Define necessary operators (example: Spin-1/2 operators)
s0 = np.eye(2)
sp = np.asarray([[0, 1], [0, 0]])
sm = np.asarray([[0, 0], [1, 0]])
sz = np.asarray([[0.5, 0], [0, -0.5]])
o = np.zeros((2, 2))

# Define MPO tensor data (example for a specific Heisenberg construction)
# Left boundary tensor
M1 = [[h/2*(sp+sm), j/2*sm, j/2*sp, jz*sz, s0]]
N1 = OperatorNode(M1) # Legs: (right:5, up:2, down:2)

# Bulk tensor
M2 = [[s0, o, o, o, o],
      [sp, o, o, o, o],
      [sm, o, o, o, o],
      [sz, o, o, o, o],
      [h/2*(sp+sm), j/2*sm, j/2*sp, jz*sz, s0]]
N2 = OperatorNode(M2) # Legs: (left:5, up:2, down:2)

# Right boundary tensor
M3 = [[s0],
      [sp],
      [sm],
      [sz],
      [h/2*(sp+sm)]]
N3 = OperatorNode(M3) # Legs: (left:5, right:5, up:2, down:2)

print('Bulk MPO tensor shape:', np.shape(N2.data))
print('Bulk MPO tensor legs:', N2.legs)
   ```
   *(Note: This example manually constructs MPO tensors. `example4` shows an elegant way using `LocalHamiltonianChain`.)*

**4. MPO Generation (`example4`)**
   Using the `LocalHamiltonianChain` helper class to automatically build the MPO for a Hamiltonian with nearest-neighbor and on-site terms.

   ```python
# --- example4: Another way to generate MPO ---
from mpo import LocalHamiltonianChain
import numpy as np

# Define parameters (e.g., Heisenberg model)
j = 1.0
jz = 1.0
h = 1.0 # Example: Transverse field Sx = (sp + sm)/2
L = 20  # Chain length

# Define local operators needed
sp = np.asarray([[0, 1], [0, 0]])
sm = np.asarray([[0, 0], [1, 0]])
sz = np.asarray([[0.5, 0], [0, -0.5]])

# Initialize the MPO class
H = LocalHamiltonianChain()

# Define operators used in the Hamiltonian
H.define_operator('sp', sp)
H.define_operator('sm', sm)
H.define_operator('sz', sz)

# Add nearest-neighbor interactions (term by term)
# J/2 (sp_i sm_{i+1} + sm_i sp_{i+1}) + Jz sz_i sz_{i+1}
H.add_neighbor_interaction('sp', 'sm', j / 2.0)
H.add_neighbor_interaction('sm', 'sp', j / 2.0)
H.add_neighbor_interaction('sz', 'sz', jz)

# Add on-site interactions (example: h Sx = h/2*(sp + sm))
H.add_onsite_interaction('sp', h / 2.0)
H.add_onsite_interaction('sm', h / 2.0)

# Generate the MPO TensorChain for the specified length L
H.gen_data(L=L)

print('Generated MPO length:', len(H))
print('Shape of first MPO tensor:', np.shape(H[0].data))
print('Legs of first MPO tensor:', H[0].legs)
   ```

**5. DMRG Simulation (`example5`)**
   Setting up and running the single-site DMRG calculation to find the ground state of the Heisenberg model.

   ```python
# --- example5: Variational MPS algorithm (finite DMRG) for Heisenberg XXZ model ---
from dmrg import Dmrg
from mps import random_tensor_chain
from mpo import LocalHamiltonianChain
import numpy as np

# --- Set Parameters ---
j = 1.0   # XX coupling
jz = 1.0  # Z coupling
h = 0.0   # On-site field (set to 0 for standard Heisenberg)
L = 20    # System size
bond_dim = 40 # MPS bond dimension for truncation
lhd = 2   # Local Hilbert space dimension (spin-1/2)

# --- Generate MPO ---
sp = np.asarray([[0, 1], [0, 0]])
sm = np.asarray([[0, 0], [1, 0]])
sz = np.asarray([[0.5, 0], [0, -0.5]])

H = LocalHamiltonianChain()
H.define_operator('sp', sp)
H.define_operator('sm', sm)
H.define_operator('sz', sz)
H.add_neighbor_interaction('sp', 'sm', j / 2.0)
H.add_neighbor_interaction('sm', 'sp', j / 2.0)
H.add_neighbor_interaction('sz', 'sz', jz)
H.add_onsite_interaction('sp', 1/2*h)
H.add_onsite_interaction('sm', 1/2*h)
H.gen_data(L=L)

# --- Generate Initial Random MPS ---
initial_mps = random_tensor_chain(lhd, L, truncation=bond_dim)

# --- Setup and Run DMRG ---
# eps: convergence tolerance for energy per site
# visual=True: prints canonical form during sweeps
dmrg_solver = Dmrg(mps=initial_mps, mpo=H, eps=1e-6, visual=True)
dmrg_solver.start() # Run the DMRG sweeps until convergence

# --- Print Result ---
print(f'Final ground state energy per site = {dmrg_solver.energy:.12f}')
# Final MPS is stored in dmrg_solver._mps
   ```

## Code Structure

* `tensor.py`: Defines `Leg`, `TensorNode`, and `TensorChain` base classes.
* `mps.py`: Contains functions for generating MPS (e.g., `random_tensor_chain`).
* `mpo.py`: Defines `OperatorNode` and `LocalHamiltonianChain` for MPO construction.
* `linalg.py`: Wrappers for linear algebra functions (QR, LQ, SVD) used internally.
* `dmrg.py`: Implements the `Dmrg` class containing the core algorithm.
* `example.py`: Demonstrates usage of the library components.

## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
