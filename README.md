# QUBO Project

This repository contains the implementation of algorithms and techniques for solving Quadratic Unconstrained Binary Optimization (QUBO) problems. The project includes classical and quantum-inspired approaches for optimization.

## Project Structure

- main.py: The main entry point for the project, orchestrating the execution of various components.

- ClassicSA.py: Implements a classical Simulated Annealing algorithm for solving QUBO problems. Simulated Annealing is a probabilistic technique for approximating the global optimum of a given function.

- Lqa.py: Implementations of a Local Quantum Annealing approach. Adjustments and modification of code by https://doi.org/10.1103/PhysRevApplied.18.03401

- QML.py: Implements Quantum Machine Learning techniques.

## Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- Required libraries specified in the code

You can install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone this repository:

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Add your data to the appropriate directory. QUBO problems used for this code can be found in https://github.com/rliang/qubo-benchmark-instances

3. Run the main script:

    ```bash
    python main.py
    ```

4. Modify parameters in the respective Python files to experiment with different configurations or algorithms.

## Contributions

Feel free to submit issues or pull requests for bug fixes, new features, or improvements.
