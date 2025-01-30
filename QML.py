import numpy as np
import matplotlib.pyplot as plot


class QMLsolver:
    """
    Quantum-inspired Machine Learning (QML) solver for Quadratic Unconstrained Binary Optimization (QUBO) problems.

    Uses simulated annealing with tunable parameters for optimization.
    """

    def __init__(self, qubo_matrix, initial_temperature=100000, cooling_rate=0.99999, max_iterations=1000000):
        """
        Initialize the QMLsolver with problem parameters.

        Args:
            qubo_matrix (np.ndarray): QUBO coefficient matrix.
            initial_temperature (float, optional): Initial temperature for annealing. Default is 100000.
            cooling_rate (float, optional): Cooling rate for annealing. Default is 0.99999.
            max_iterations (int, optional): Maximum number of iterations. Default is 1000000.
        """
        self.num_variables = qubo_matrix.shape[0]
        self.qubo_matrix = qubo_matrix
        self.initial_temperature = self.num_variables*10000  # Temperature scaling based on problem size
        self.cooling_rate = 1 - 10/self.initial_temperature  # Adaptive cooling rate as a parameter of the initial temperature
        self.max_iterations = self.num_variables*20000  # Adaptive iteration limit as a parameter of the problem size

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


    def error(self, solution, optSol):
        """
        Compute the difference between the current solution's energy and the optimal solution.

        Args:
            solution (list): Current solution.
            optSol (float): Optimal solution energy.

        Returns:
            float: Energy difference.
        """
        energy = self.calculate_energy(solution)
        return energy - optSol

    def adjacent(self, solution, n_swaps, rnd):
        """
        Generate a neighboring solution by randomly flipping bits.

        Args:
            solution (list): Current binary solution vector.
            n_swaps (int): Number of bits to flip.
            rnd (np.random.RandomState): Random generator instance.

        Returns:
            list: New solution vector with flipped bits.
        """
        n = self.num_variables
        neighbor = np.copy(solution)
        for ns in range(n_swaps):
            i = rnd.randint(0, n-1)
            neighbor[i] = 1 - neighbor[i]  # Flip bit - 0 to 1 or 1 to 0
        return neighbor

    def my_kendall_tau_dist(self, p1, p2):
        """
        Compute the Kendall tau distance between two sequences.

        Args:
            p1 (list): First sequence.
            p2 (list): Second sequence.

        Returns:
            tuple: Raw distance (number of pair misorderings) and normalized distance.
        """
        n = len(p1)
        index_of = [None] * n  # Lookup table for p2 positions
        for i in range(n):
            v = p2[i]
            index_of[v] = i

        d = 0  # Count misordered pairs
        for i in range(n):
            for j in range(i+1, n):
                if index_of[p1[i]] > index_of[p1[j]]:
                    d += 1
        normer = n * (n - 1) / 2.0
        nd = d / normer  # normalized distance
        return (d, nd)

    def minimise(self, optSol, pctTunnel=0.15):
        """
        Perform simulated annealing to minimise the QUBO problem.

        Args:
            optSol (float): Optimal solution energy for comparison.
            pctTunnel (float, optional): Probability of tunneling move. Default is 0.15.

        Returns:
            tuple: Best solution, its energy, convergence step, and error values over iterations.
        """
        # print("number of iterations %d, initial temperature %d, cooling rate %f" % (
        # self.max_iterations, self.initial_temperature, self.cooling_rate))

        rnd = np.random.RandomState(6)  # Random seed for reproducibility
        currTemp = self.initial_temperature
        soln = [rnd.randint(0, 1) for _ in range(self.num_variables)]
        # print("Initial guess: ")
        # print(soln)

        err = self.error(soln, optSol)
        iteration = 0
        interval = (int)(self.max_iterations / 10)

        bestSoln = np.copy(soln)
        bestErr = err

        errors = []
        # converge = 0
        # Threshold for convergence
        convergence_threshold = 0.1 * optSol
        convergence_step = self.max_iterations  # to record the step at which convergence happens
        is_converged = False
        converge_counter = 0

        while iteration < self.max_iterations:# and err > 0.0:
            # Determine the number of bit flips based on iteration progress
            pct_iters_left = (self.max_iterations - iteration) / (self.max_iterations * 1.0)
            p = rnd.random()  # [0.0, 1.0]
            if p < pctTunnel:            # tunnel
                numSwaps = (int)(pct_iters_left * self.num_variables)
                if numSwaps < 1:
                     numSwaps = 1
            else: # no tunneling
                numSwaps = 1

            adjSol = self.adjacent(soln, numSwaps, rnd)
            adjErr = self.error(adjSol, optSol)

            # Update the best known solution
            if adjErr < bestErr:
                bestSoln = np.copy(adjSol)
                bestErr = adjErr

            errDifference = err - adjErr

            # Accept the new solution if it's better, or probabilistically based on temperature
            if adjErr < err:  # better route so accept
                soln = adjSol
                err = adjErr
            else: # adjacent is worse
                accept_p = np.exp((err - adjErr) / currTemp)
                p = rnd.random()
                if p < accept_p:  # accept anyway
                    soln = adjSol
                    err = adjErr
                # else don't accept worse route

            # Print statistics
            if iteration % interval == 0:
                (dist, nd) = self.my_kendall_tau_dist(soln, adjSol)
                print("iteration = %6d | " % iteration, end="")
                print("dist curr to candidate = %8.4f | " % nd, end="")
                print("curr_temp = %12.4f | " % currTemp, end="")
                print("error = %6.1f " % bestErr)

            # Reduce temperature following the cooling schedule
            if currTemp < 0.00001:
                currTemp = 0.00001
            else:
                currTemp *= self.cooling_rate

            # Check for convergence
            if not is_converged:
                if errDifference < convergence_threshold:
                    if converge_counter == 0:
                        convergence_step = iteration
                    converge_counter += 1
                    if converge_counter > 100:
                        is_converged = True

                else:
                    converge_counter = 0
                    convergence_step = iteration

            if is_converged:
                break

            errors.append(err/optSol)
            iteration += 1

        # Plot convergence results
        energy = self.calculate_energy(bestSoln)
        print(bestSoln)
        plot.title("Convergence - QUBO")
        x_axis = range(0, iteration)
        plot.xlabel('iterations')
        plot.ylabel('loss')
        plot.plot(x_axis, errors)
        plot.show()

        return bestSoln, energy, convergence_step, errors

