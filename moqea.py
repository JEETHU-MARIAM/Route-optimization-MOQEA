import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.compiler import transpile
from math import ceil, log2
import pandas as pd


class MOQERouteOptimizer:
    def __init__(self, distances, traffic_conditions, population_size=10):
        self.distances = distances
        self.traffic_conditions = traffic_conditions
        self.num_routes = len(distances)
        self.num_qubits = ceil(log2(self.num_routes))
        self.population_size = population_size
        self.population = []
        self.pareto_front = []
        self.pareto_index=[]

    def normalize(self, data):
        """Normalize data to [0, 1] range."""
        data = np.array(data)
        if data.max() == data.min():
            return np.zeros_like(data)
        return (data - data.min()) / (data.max() - data.min())

    def initialize_population(self):
        """Initialize the quantum population."""
        self.population = [
            format(np.random.randint(0, 2**self.num_qubits), f'0{self.num_qubits}b')
            for _ in range(self.population_size)
        ]

    def fitness(self, route_idx):
        """Calculate fitness values for both objectives."""
        route_idx = int(route_idx, 2) % self.num_routes  # Ensure index is within bounds
        
        distance = self.distances_normalized[route_idx]
        traffic = self.traffic_normalized[route_idx]
        return (distance, traffic)

    def mutate(self, individual):
        """Perform mutation by flipping random bits."""
        if self.num_qubits > 0:
            mutation_probability = 2 / self.num_qubits  # Increased mutation probability
            mutated = ''.join(
                str((int(bit) + 1) % 2) if np.random.random() < mutation_probability else bit
                for bit in individual
            )
            return mutated
        else:
            raise ValueError("Number of qubits should be greater than 0")

    def pareto_selection(self, population):
        """Select non-dominated solutions based on Pareto dominance."""
        pareto_front = []
        for i in population:
            dominated = False
            for j in population:
                if i != j:
                    fit_i = np.array(self.fitness(i))
                    fit_j = np.array(self.fitness(j))
                    if np.all(fit_j <= fit_i) and np.any(fit_j < fit_i):
                        dominated = True
                        break
            if not dominated:
                pareto_front.append(i)
        return pareto_front

    def update_pareto_front(self):
        """Update and maintain a diverse Pareto front."""
        combined_population = list(set(self.pareto_front + self.population))
        self.pareto_front = self.pareto_selection(combined_population)

    def evolve_population(self):
        """Evolve the population using mutation only (crossover is disabled)."""
        new_population = []
        while len(new_population) < self.population_size:
            parent = np.random.choice(self.population)
            child = self.mutate(parent)
            new_population.append(child)
        self.population = new_population[:self.population_size]

    def optimize(self, generations=10):
        """Optimize using MOQEA with enhanced diversity and Pareto management."""
        self.distances_normalized = self.normalize(self.distances)
        self.traffic_normalized = self.normalize(self.traffic_conditions)

        self.initialize_population()
        for generation in range(generations):
            print(f"Generation {generation + 1}")
            self.evolve_population()
            self.update_pareto_front()
            self.display_results_per_generation(generation, self.pareto_front)

        return self.pareto_front

    def display_results_per_generation(self, generation, refined_pareto):
        """Display optimization results per generation."""
        print(f"\nGeneration {generation + 1} Pareto-Optimal Routes:")
        for route in refined_pareto:
            idx = int(route, 2)
            print(f"Route {idx}: Distance = {self.distances[idx]} km, "
                  f"Traffic Condition = {self.traffic_conditions[idx]}")

    def display_results(self, refined_pareto):
        """Display optimization results."""
        print("\nAvailable Routes:")
        for i, (d, t) in enumerate(zip(self.distances, self.traffic_conditions)):
            print(f"Route {i}: Distance = {d} km, Traffic Condition = {t}")

        print("\nPareto-Optimal Routes:")
        for route in refined_pareto:
            idx = int(route, 2)
            self.pareto_index.append(idx)
            print(f"Route {idx}: Distance = {self.distances[idx]} km, "
                  f"Traffic Condition = {self.traffic_conditions[idx]}")
        print(self.pareto_index)
        return self.pareto_index

    def plot_pareto_front(self, refined_pareto):
        """Plot Pareto front visualization."""
        pareto_points = [(self.distances[int(i, 2)], self.traffic_conditions[int(i, 2)])
                         for i in refined_pareto]
        all_points = list(zip(self.distances, self.traffic_conditions))

        plt.figure(figsize=(10, 6))
        plt.scatter(*zip(*all_points), color='blue', label='All Routes')
        plt.scatter(*zip(*pareto_points), color='red', label='Pareto Front')
        plt.xlabel('Distance (km)')
        plt.ylabel('Traffic Condition (1-10)')
        plt.title('Pareto Front for Route Optimization')
        plt.legend()
        plt.grid(True)
        plt.show()


# Example usage
if __name__ == "__main__":
    # Load distances from the text file
    distances = []
    with open('distances.txt', 'r') as file:
        distances = [float(line.strip()) for line in file]

    traffic_conditions = []
    with open('costs.txt', 'r') as file:
        traffic_conditions = [float(line.strip()) for line in file]

    # Initialize the optimizer
    optimizer = MOQERouteOptimizer(distances, traffic_conditions)
    refined_pareto = optimizer.optimize(generations=10)
    
    # Display results
    pareto_index2=optimizer.display_results(refined_pareto)
    optimizer.plot_pareto_front(refined_pareto)
    with open('pareto_index.txt', 'w') as file:
        for p in pareto_index2:
            file.write(f"{p}\n")
