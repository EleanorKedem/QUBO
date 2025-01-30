import ClassicSA as sa
import QML as qml
import Lqa as lqa
import numpy as np
import cProfile
import pstats
import matplotlib.pyplot as plot


def test():
    """
    Function to test the QUBO optimization process with a predefined matrix and solution vector.

    This function evaluates the performance of the QUBO solver by comparing the energy of
    the obtained solution with the optimal solution and calculates the loss. It also prints
    the convergence iteration and the final solution.

    Steps:
    - Define a QUBO matrix and a solution vector.
    - Compute the optimal solution energy using the matrix and solution vector.
    - Use the QUBOSolver (or alternative solvers) to minimize the QUBO problem.
    - Calculate the loss and print key performance metrics.
    """

    # Define a sample QUBO problem (5x5 matrix and solution vector)
    matrix = [[ 1, -2,  3, -4,  2],
              [ -2,-1,  2, -3,  1],
              [ 3,  2,  1, -2,  2],
              [ -4,-3, -2, -1,  1],
              [ 2,  1,  2,  1,  1]]
    solVect = [1, 1, 0, 1, 0]

    # Compute the optimal solution energy using the QUBO matrix
    optSolution = (np.dot(np.matmul(matrix + np.diag(np.diag(matrix)), solVect), solVect)) / 2

    # Initialize the QUBO solver (selecting one of the available solvers)
    machine = sa.QUBOSolver(matrix)  # lqa.Lqa(matrix)  #qml.QMLsolver(matrix)

    # Minimize the QUBO problem and retrieve the results
    solution, energy, iteration = machine.minimise(optSolution)

    # Calculate the relative loss compared to the optimal solution
    loss = (energy - optSolution) / optSolution

    # Print performance metrics
    print("Optimal solution %f " % optSolution)  # Expected energy of the optimal solution
    print("Loss %f" % loss)  # Relative loss in energy
    print("Convergence iteration %d, with best solution %d" % (iteration, energy))  # Convergence stats
    print(solution)  # Final solution vector
    print((np.dot(np.matmul(matrix + np.diag(np.diag(matrix)), solution), solution)) / 2)  # Validate solution energy

# Function to calculate the Euclidean distance between two points
def calcDistance(x1, y1, x2, y2):
    """
    Calculate the Euclidean distance between two points.

    Args:
        x1, y1 (float): Coordinates of the first point.
        x2, y2 (float): Coordinates of the second point.

    Returns:
        float: Euclidean distance between the two points.
    """
    dist = np.sqrt(pow((x1-x2),2) + pow((y1-y2),2))
    return dist

# Function to create the QUBO matrix from a file
def create_QUBO_matrix(file_name):
    """
    Create a QUBO matrix from a file.

    The file should contain the size of the matrix on the first line,
    followed by rows of entries (i, j, coefficient).

    Args:
        file_name (str): Path to the input file containing the QUBO matrix data.

    Returns:
        np.ndarray: Symmetric QUBO matrix constructed from the file.
    """
    with open(file_name, 'r') as file:
        lines = file.readlines()

        # Read the size of the QUBO matrix from the first line
        size = int(lines[0].strip())

        # Initialize the QUBO matrix with zeros
        QUBO_matrix = np.zeros((size, size), dtype=int)

        # Process each line after the first
        for line in lines[1:]:
            i, j, coefficient = map(int, line.split())
            QUBO_matrix[i, j] = coefficient
            if i != j:
                QUBO_matrix[j, i] = coefficient  # Ensure the matrix is symmetric

        return QUBO_matrix

# Function to benchmark QUBO solutions
def QUBO_benchmark():
    """
    Benchmark the performance of the QUBO solver across multiple cases.

    Iterates through a list of predefined benchmark cases, reads the QUBO matrix
    for each case, and evaluates the solver's performance. Results include loss values,
    convergence iterations, and other metrics.
    """
    print("Begin QUBO using quantum inspired annealing - QUBO benchmark cases")

    data = [(-2098, 'bqp50.1'),(-3702, 'bqp50.2'),(-4626, 'bqp50.3'),(-3544, 'bqp50.4'),(-4012, 'bqp50.5'),(-3693, 'bqp50.6'),(-4520, 'bqp50.7'),(-4216, 'bqp50.8'),(-3780, 'bqp50.9'),(-3507, 'bqp50.10'),
            (-7970, 'bqp100.1'),(-11036, 'bqp100.2'),(-12723, 'bqp100.3'),(-10368, 'bqp100.4'),(-9083, 'bqp100.5'),(-10210, 'bqp100.6'),(-10125, 'bqp100.7'),(-11435, 'bqp100.8'),(-11455, 'bqp100.9'),(-12565, 'bqp100.10'),
            (-45607, 'bqp250.1'),(-44810, 'bqp250.2'),(-49037, 'bqp250.3'),(-41274, 'bqp250.4'),(-47961, 'bqp250.5'),(-41014, 'bqp250.6'),(-46757, 'bqp250.7'),(-35726, 'bqp250.8'),(-48916, 'bqp250.9'),(-40442, 'bqp250.10'),
            (-116586, 'bqp500.1'),(-128223, 'bqp500.2'),(-130812, 'bqp500.3'),(-130097, 'bqp500.4'),(-125487, 'bqp500.5'),(-121772, 'bqp500.6'),(-122201, 'bqp500.7'),(-123559, 'bqp500.8'),(-120798, 'bqp500.9'),(-130619, 'bqp500.10'),
            (-371438, 'bqp1000.1'),(-354932, 'bqp1000.2'),(-371236, 'bqp1000.3'),(-370675, 'bqp1000.4'),(-352760, 'bqp1000.5'),(-359629, 'bqp1000.6'),(-371193, 'bqp1000.7'),(-351994, 'bqp1000.8'),(-349337, 'bqp1000.9'),(-351415, 'bqp1000.10'),
            (-1515944, 'bqp2500.1'),(-1471392, 'bqp2500.2'),(-1414192, 'bqp2500.3'),(-1507701, 'bqp2500.4'),(-1491816, 'bqp2500.5'),(-1469162, 'bqp2500.6'),(-1479040, 'bqp2500.7'),(-1484199, 'bqp2500.8'),(-1482413, 'bqp2500.9'),(-1483355, 'bqp2500.10'),
            (-3931583, 'p3000.1'),(-5193073, 'p3000.2'),(-5111533, 'p3000.3'),(-5761822, 'p3000.4'),(-5675625, 'p3000.5'),
            (-6181830, 'p4000.1'),(-7801355, 'p4000.2'), (-7741685, 'p4000.3'), (-8711822, 'p4000.4'), (-8908979, 'p4000.5'),
            (-8559680, 'p5000.1'), (-10836019, 'p5000.2'), (-10489137, 'p5000.3'), (-12252318, 'p5000.4'), (-12731803, 'p5000.5'),
            (-11384976, 'p6000.1'), (-14333855, 'p6000.2'), (-16132915, 'p6000.3'),
            (-14478676, 'p7000.1'), (-18249948, 'p7000.2'), (-20446407, 'p7000.3'),
            (-285149199, 'torusg3-15'), (-41684814, 'torusg3-8'), (-3014, 'toruspm3-15-50'), (-458, 'toruspm3-8-50')]

    loss_values = []
    convergence = []
    graph = []
    n = 0
    sum_iteration = sum_loss = 0

    for file in range(len(data)):
        n += 1
        optSol = data[file][0]
        print("problem %d, optimal solution %d" % (n, optSol))

        matrix = create_QUBO_matrix(data[file][1])

        # Initialize the QUBO solver (selecting one of the available solvers)
        machine = sa.QUBOSolver(matrix) #qml.QMLsolver(matrix) #lqa.Lqa(matrix)

        # Initialize the profiler for stats
        # profiler = cProfile.Profile()
        # profiler.enable()

        # Minimize the QUBO problem and retrieve the results
        solution, energy, iteration, errors = machine.minimise(optSol)
        # profiler.disable()

        # Print out the stats
        # stats = pstats.Stats(profiler)
        # stats.sort_stats('cumtime').print_stats(10)  # Sort by cumulative time and show top 10 functions

        # Calculate the relative loss compared to the optimal solution
        loss = (energy - optSol) / optSol

        # Collect results for statistics across different problems
        loss_values.append(loss)
        convergence.append(iteration)
        graph.append(errors)

        sum_iteration += iteration
        sum_loss += loss

        # Print performance metrics
        print("Loss %f" % loss)  # Relative loss in energy
        print("Convergence iteration %d, with best energy %d" % (iteration, energy))  # Convergence stats
        print("Average iteration %d, average loss %f" % ((sum_iteration / n), (sum_loss / n)))  # Statistics across different problems

# Main entry point for the script
def main():
    """
    Main function to execute the QUBO benchmark and other evaluations.
    """
    QUBO_benchmark()

if __name__ == "__main__":
    main()
