import random
import numpy as np
import pandas as pd

class GeneticAlgorithm:
    def __init__(self, input_vector, n, tournament_size, distance_matrix, mutation_point, crossover_type, mutation_type):
        # Initialize the genetic algorithm with parameters
        self.mutation_point = mutation_point
        self.mutation_type = mutation_type
        self.input_vector = input_vector
        self.n = n
        self.tournament_size = tournament_size
        self.distance_matrix = distance_matrix
        self.locations = len(input_vector)
        self.crossover_type = crossover_type
        self.candidates = self.generate_permutations()

    def calculate_total_cost(self, input_vector):
        # Calculate the total cost of a given input vector
        total_cost = 0
        for i in range(self.locations - 1):
            source = input_vector[i]
            destination = input_vector[i + 1]
            total_cost += self.distance_matrix[source][destination]

        total_cost += self.distance_matrix[input_vector[-1]][input_vector[0]]
        return total_cost

    def generate_permutations(self):
        # Generate a population of candidate solutions
        unique_items = list(range(self.locations))
        permutations = []
        while len(permutations) < self.n:
            random.shuffle(unique_items)
            permutation = list(unique_items)
            cost = self.calculate_total_cost(permutation)
            permutations.append((permutation, cost))

        candidates = pd.DataFrame(permutations, columns=['Candidate_Solution', 'Cost'])
        return candidates

    def tournament_selection(self, tournament_size):
        # Perform tournament selection to choose the best candidate
        selected_indices = random.sample(range(len(self.candidates)), tournament_size)
        selected_candidates = self.candidates.loc[selected_indices].sort_values('Cost', ascending=True).reset_index(drop=True)
        return selected_candidates['Candidate_Solution'][0]

    def repair_offspring(self, offspring, candidate):
        # Repair duplicated genes in offspring by replacing them with unused genes from the candidate
        seen = set()
        repaired_offspring = []

        for chromosome in offspring:
            if chromosome not in seen:
                seen.add(chromosome)
                repaired_offspring.append(chromosome)
            else:
                for replacement in candidate:
                    if replacement not in seen:
                        seen.add(replacement)
                        repaired_offspring.append(replacement)
                        break
        return repaired_offspring

    def single_point_swap_mutation(self, offspring):
        # Perform single-point swap mutation on the offspring
        mutation_point1, mutation_point2 = random.sample(range(len(offspring)), 2)
        offspring[mutation_point1], offspring[mutation_point2] = offspring[mutation_point2], offspring[mutation_point1]
        return offspring

    def single_point_crossover(self, candidate1, candidate2):
        # Perform single-point crossover between two candidates
        crossover_point = random.randint(1, len(candidate1) - 1)
        offspring1 = candidate1[:crossover_point] + candidate2[crossover_point:]
        offspring2 = candidate2[:crossover_point] + candidate1[crossover_point:]
        offspring1 = self.repair_offspring(offspring1, candidate1)
        offspring2 = self.repair_offspring(offspring2, candidate2)
        return offspring1, offspring2

    def ordered_crossover(self, parent1, parent2):
        # Perform ordered crossover between two parents
        crossover_points = sorted(random.sample(range(len(parent1)), 2))
        offspring1 = [None] * len(parent1)
        offspring2 = [None] * len(parent1)
        offspring1[crossover_points[0]:crossover_points[1] + 1] = parent1[crossover_points[0]:crossover_points[1] + 1]
        offspring2[crossover_points[0]:crossover_points[1] + 1] = parent2[crossover_points[0]:crossover_points[1] + 1]
        remaining_elements1 = [gene for gene in parent2 if gene not in offspring1]
        remaining_elements2 = [gene for gene in parent1 if gene not in offspring2]
        offspring_index1 = crossover_points[1] + 1
        offspring_index2 = crossover_points[1] + 1

        for gene1, gene2 in zip(remaining_elements1, remaining_elements2):
            if offspring1[offspring_index1] is None:
                offspring1[offspring_index1] = gene1
            if offspring2[offspring_index2] is None:
                offspring2[offspring_index2] = gene2
            offspring_index1 = (offspring_index1 + 1) % len(parent1)
            offspring_index2 = (offspring_index2 + 1) % len(parent1)

        return offspring1, offspring2

    def inverse_mutation(self, offspring):
        # Perform inverse mutation on the offspring
        mutation_point1, mutation_point2 = sorted(random.sample(range(len(offspring)), 2))
        reversed_segment = offspring[mutation_point1:mutation_point2 + 1][::-1]
        offspring[mutation_point1:mutation_point2 + 1] = reversed_segment
        return offspring

    def perform_crossover(self, candidate1, candidate2):
        # Perform crossover based on the specified type
        if self.crossover_type == 'fix':
            return self.single_point_crossover(candidate1, candidate2)
        elif self.crossover_type == 'ordered':
            return self.ordered_crossover(candidate1, candidate2)
        else:
            raise ValueError("Invalid crossover type")

    def perform_mutation(self, offspring, mutation_type='single'):
        # Perform mutation based on the specified type
        if mutation_type == 'single':
            return self.single_point_swap_mutation(offspring)
        elif mutation_type == 'inversion':
            return self.inverse_mutation(offspring)
        elif mutation_type == 'multi':
            return self.multi_point_swap_mutation(offspring, self.mutation_point)
        else:
            raise ValueError("Invalid mutation type")

    def replace_highest_cost_solution(self, x):
        # Replace the highest-cost solution with a new solution
        highest_cost_index = self.candidates['Cost'].idxmax()
        self.candidates.at[highest_cost_index, 'Candidate_Solution'] = x
        self.candidates.at[highest_cost_index, 'Cost'] = self.calculate_total_cost(x)

    def multi_point_swap_mutation(self, offspring, n):
        # Perform multi-point swap mutation on the offspring
        mutation_points = random.sample(range(self.locations), n)
        for i in range(0, n, 2):
            if i + 1 < n:
                offspring[mutation_points[i]], offspring[mutation_points[i + 1]] = offspring[mutation_points[i + 1]], offspring[mutation_points[i]]
        return offspring
