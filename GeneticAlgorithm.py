import random
import numpy as np
import pandas as pd

class GeneticAlgorithm:
    def __init__(self, input_vector, n, tournament_size, distance_matrix, mutation_point):
        self.mutation_point = mutation_point
        self.input_vector = input_vector
        self.n = n
        self.tournament_size = tournament_size
        self.distance_matrix = distance_matrix
        self.locations = len(input_vector)
        self.candidates = self.generate_permutations()

    def calculate_total_cost(self, input_vector):
        total_cost = 0
        for i in range(self.locations - 1):
            source = input_vector[i]
            destination = input_vector[i + 1]
            total_cost += self.distance_matrix[source][destination]

        total_cost += self.distance_matrix[input_vector[-1]][input_vector[0]]
        return total_cost

    def generate_permutations(self):
        unique_items = list(range(self.locations))
        permutations = []
        while len(permutations) < self.n:
            random.shuffle(unique_items)
            permutation = list(unique_items)
            cost = self.calculate_total_cost(permutation)
            permutations.append((permutation, cost))

        candidates = pd.DataFrame(permutations, columns=['Candidate_Solution', 'Cost'])
        return candidates

    def tournament_selection(self, tournament_size=2):
        selected_indices = random.sample(range(len(self.candidates)), tournament_size)
        selected_candidates = self.candidates.loc[selected_indices]
        return selected_candidates['Candidate_Solution'].tolist()

    def repair_offspring(self, offspring, candidate):
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
        mutation_point1, mutation_point2 = random.sample(range(len(offspring)), 2)
        offspring[mutation_point1], offspring[mutation_point2] = offspring[mutation_point2], offspring[mutation_point1]
        return offspring

    def single_point_crossover(self, candidate1, candidate2):

        crossover_point = random.randint(1, len(candidate1) - 1)
        offspring1 = candidate1[:crossover_point] + candidate2[crossover_point:]
        offspring2 = candidate2[:crossover_point] + candidate1[crossover_point:]
        offspring1 = self.repair_offspring(offspring1, candidate1)
        offspring2 = self.repair_offspring(offspring2, candidate2)
        return offspring1, offspring2

    def replace_highest_cost_solution(self, x):
        highest_cost_index = self.candidates['Cost'].idxmax()
        self.candidates.at[highest_cost_index, 'Candidate_Solution'] = x
        self.candidates.at[highest_cost_index, 'Cost'] = self.calculate_total_cost(x)

    def multi_point_swap_mutation(self, offspring, n):
        mutation_points = random.sample(range(self.locations), n)
        for i in range(0, n, 2):
            if i + 1 < n:
                offspring[mutation_points[i]], offspring[mutation_points[i + 1]] = offspring[mutation_points[i + 1]], offspring[mutation_points[i]]
        return offspring
