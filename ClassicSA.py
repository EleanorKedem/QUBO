import random
import math
import numpy as np
import matplotlib.pyplot as plot

class QUBOSolver:
    """
    A solver for Quadratic Unconstrained Binary Optimization (QUBO) problems
    using a simulated annealing approach.
    """

    def __init__(self, qubo_matrix, initial_temperature=10000, cooling_rate=0.9999, max_iterations=100000):
        """
        Initialize the QUBOSolver with problem parameters.

        Args:
            qubo_matrix (np.ndarray): QUBO coefficient matrix.
            initial_temperature (float, optional): Initial temperature for annealing. Default is 10000.
            cooling_rate (float, optional): Cooling rate for annealing. Default is 0.9999.
            max_iterations (int, optional): Maximum number of iterations. Default is 100000.
        """
        self.qubo_matrix = qubo_matrix
        self.num_variables = len(qubo_matrix)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = self.num_variables*2000  # Adaptive iteration limit as a parameter of the problem size

    def calculate_energy(self, solution):
        """
        Calculate the energy of a given solution based on the QUBO matrix.

        Args:
            solution (list): Binary vector representing the solution.

        Returns:
            float: Energy of the solution.
        """
        solVect = np.array(solution)
        return (np.dot(np.matmul(self.qubo_matrix + np.diag(np.diag(self.qubo_matrix)), solVect), solVect)) / 2

    def get_neighbor(self, solution):
        """
        Generate a neighboring solution by flipping a random bit in the current solution.

        Args:
            solution (list): Current binary solution vector.

        Returns:
            list: New solution vector with one bit flipped.
        """
        neighbor = solution[:]
        index_to_flip = random.randint(0, self.num_variables - 1)
        neighbor[index_to_flip] = 1 - neighbor[index_to_flip]  # Flip 0 to 1 or 1 to 0
        return neighbor

    def minimise(self, optSol):
        """
        Perform simulated annealing to minimize the QUBO problem.

        Args:
            optSol (float): Optimal solution energy for comparison.

        Returns:
            tuple: Final solution, its energy, convergence step, and error values over iterations.
        """
        current_solution = [random.randint(0, 1) for _ in range(self.num_variables)]
        current_energy = self.calculate_energy(current_solution)

        temperature = self.initial_temperature

        # Threshold for convergence
        convergence_threshold = -0.1  # set an appropriate threshold
        convergence_step = self.max_iterations  # to record the step at which convergence happens
        is_converged = False
        converge_counter = 0

        errors = []

        for iteration in range(self.max_iterations):
            # Generate a neighboring solution by flipping a random bit
            new_solution = self.get_neighbor(current_solution)
            new_energy = self.calculate_energy(new_solution)
            prev_energy = current_energy

            # Check if the new solution is better
            # Accept the new solution if it has lower energy or probabilistically based on temperature
            if new_energy < current_energy or random.random() < math.exp((current_energy - new_energy) / temperature):
                current_solution = new_solution
                current_energy = new_energy

            # Check for convergence
            if not is_converged:
                energy_change = prev_energy - current_energy
                if energy_change > convergence_threshold:
                    if converge_counter == 0:
                        convergence_step = iteration  # Record first convergence step
                    converge_counter += 1
                    if converge_counter > 1000:
                        is_converged = True

                else:
                    converge_counter = 0
                    convergence_step = iteration

            # if is_converged:
            #     break

            # Reduce temperature based on cooling schedule
            temperature *= self.cooling_rate
            # Record the relative error compared to the optimal solution
            errors.append((current_energy - optSol) / optSol)

        # Plot the convergence graph to visualize optimization progress
        x_axis = range(0, len(errors))
        plot.title("Convergence - QUBO")
        plot.xlabel('iterations')
        plot.ylabel('loss')
        plot.plot(x_axis, errors)
        plot.show()

        # Return the final solution, its energy, the convergence step, and errors
        return current_solution, current_energy, convergence_step, errors