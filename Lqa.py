import torch
import torch.nn as nn
import time
from math import pi
import matplotlib.pyplot as plot


class Lqa(nn.Module):
    """
    Quantum-inspired solver for Quadratic Unconstrained Binary Optimization (QUBO) problems
    using Local Quantum Annealing (LQA).
    """

    def __init__(self, couplings):
        """
        Initialize the LQA solver with the problem couplings.

        Args:
            couplings (numpy.ndarray): Square symmetric matrix encoding the problem Hamiltonian.
        """
        super(Lqa, self).__init__()

        # Convert the coupling matrix to a PyTorch tensor
        self.orig_couplings = torch.from_numpy(couplings).float()
        self.normal_factor = torch.max(torch.abs(self.orig_couplings))  # Normalise factor
        self.couplings = self.orig_couplings / self.normal_factor
        self.n = couplings.shape[0]  # Number of variables

        # Compute internal coupling interactions excluding self-loops
        self.int_couplings = self.couplings - torch.diag_embed(torch.diag(self.couplings))
        self.bias = torch.sum(self.int_couplings, dim=1) / 4 + torch.diagonal(self.couplings) / 2

        # Initialize variables for energy tracking
        self.energy = 0.
        self.config = torch.zeros([self.n, 1])
        self.min_en = 9999.
        self.min_config = torch.zeros([self.n, 1])
        self.weights = torch.zeros([self.n])

    def schedule(self, i, N):
        """
        Compute the annealing schedule based on iteration progress.

        Args:
            i (int): Current iteration.
            N (int): Total number of iterations.

        Returns:
            float: Annealing schedule parameter.
        """
        return i / N


    def energy_ising(self, config):
        """
        Compute the Ising energy of a given configuration.

        Args:
            config (torch.Tensor): Binary solution vector.

        Returns:
            torch.Tensor: Computed Ising energy.
        """
        return (torch.dot(torch.matmul(self.int_couplings, config), config)) / 8 + torch.dot(self.bias, config)

    def problem_energy(self, config):
        """
        Compute the total energy of a given configuration.

        Args:
            config (torch.Tensor): Binary solution vector.

        Returns:
            torch.Tensor: Computed configuration energy.
        """
        return (torch.dot(torch.matmul(self.orig_couplings + torch.diag_embed(torch.diag(self.orig_couplings)), config), config)) / 2

    def energy_full(self, t, g):
        """
         Compute the full cost function value combining problem Hamiltonian and driver Hamiltonian.

         Args:
             t (float): Annealing schedule parameter.
             g (float): Scaling factor for quantum tunneling.

         Returns:
             tuple: (Problem energy, total energy including driver Hamiltonian contribution).
         """
        config = torch.tanh(self.weights)*pi/2  # Cost function value. Map weights to configuration space
        ez = self.energy_ising(torch.sin(config))  # Compute problem Hamiltonian energy
        ex = torch.cos(config).sum()  # Compute driver Hamiltonian energy
        return ez, (t*ez*g- (1-t)*ex)


    def minimise(self,
                 optSol,
                 step=1,  # learning rate
                 N=1000,  # no of iterations
                 g=1.,
                 f=1.):
        """
        Perform quantum-inspired annealing to minimize the QUBO problem.

        Args:
            optSol (float): Optimal solution energy for comparison.
            step (float, optional): Learning rate for optimization. Default is 1.
            N (int, optional): Number of iterations. Default is 1000.
            g (float, optional): Balances the influence between the problem specific and the driver
            Hamiltonians. Default is 1.
            f (float, optional): Scaling factor for weight initialization. Default is 1.

        Returns:
            tuple: Best solution, its energy, convergence step, and error values over iterations.
        """
        # Initialize weights randomly
        self.weights = (2 * torch.rand([self.n]) - 1) * f
        self.weights.requires_grad=True
        time0 = time.time()
        optimizer = torch.optim.Adam([self.weights],lr=step)
        errors = []


        # Convergence tracking variables
        convergence_threshold = -0.1  # Energy change threshold for convergence - set an appropriate threshold
        convergence_step = N  # Iteration step at which convergence occurs
        is_converged = False
        converge_counter = 0

        # Compute initial solution and its energy
        solution = (torch.sign(self.weights.detach()) + 1) / 2
        sol_value = float(self.problem_energy(solution))  # * self.normal_factor

        for i in range(N):
            t = self.schedule(i, N)  # Compute annealing schedule parameter
            sol, energy = self.energy_full(t,g)  # Compute full energy function

            optimizer.zero_grad()
            energy.backward()  # Compute gradients
            optimizer.step()  # Update weights based on gradient descent

            # Convert weight-based solution to binary format
            solution = (torch.sign(self.weights.detach()) + 1)/2
            prev_sol = sol_value
            sol_value = float(self.problem_energy(solution)) #* self.normal_factor  # Compute new solution energy
            loss = sol_value - optSol  # Compute loss relative to optimal solution
            errors.append(loss/optSol)  # Store relative error

            # Check for convergence
            if not is_converged:
                energy_change = prev_sol - sol_value
                if energy_change > convergence_threshold:
                    if converge_counter == 0:
                        convergence_step = i  # Record first convergence step
                    converge_counter += 1
                    if converge_counter > 1000:
                        is_converged = True  # Mark convergence if threshold is exceeded consistently

                else:
                    converge_counter = 0  # Reset counter if energy change is negligible
                    convergence_step = i

            if is_converged:
                break  # Stop optimization once convergence is reached

        # Store final configuration and energy
        self.opt_time = time.time() - time0
        self.config = (torch.sign(self.weights.detach()) + 1) / 2
        self.energy = float(self.problem_energy(self.config)) #* self.normal_factor

        # Plot convergence graph
        x_axis = range(0, len(errors))
        plot.title("Convergence - QUBO")
        plot.xlabel('iterations')
        plot.ylabel('loss')
        plot.plot(x_axis, errors)
        plot.show()

        return self.config, self.energy, convergence_step, errors