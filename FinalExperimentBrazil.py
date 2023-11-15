import pandas as pd
import time
from GeneticAlgorithm import GeneticAlgorithm
from ReadXml import read_xml
from tqdm import tqdm
from statistics import stdev, mean
from itertools import product

# Single-time setup
xml_file_path = 'brazil58.xml'
distance_matrix = read_xml(xml_file_path)
input_string = [i for i in range(distance_matrix.shape[0])]

def run_experiment(input_string, n, tournament_size, iterations, distance_matrix, mutation_point, crossover_type, mutation_type, mutation):
    # Create a log file for each experiment
    log_file = open(f'output_csv_burma/experiment_results_pop_size_{n}_tournament_size_{tournament_size}_iterations_{iterations}_crossover_type_{crossover_type}_mutation_type_{mutation_type}_time_taken{start_time-end_time}.txt', "w")
    
    # Initialize GeneticAlgorithm instance
    ga = GeneticAlgorithm(input_string, n, tournament_size, distance_matrix, mutation_point, crossover_type, mutation_type)
    print('candidates', ga.candidates.sort_values('Cost'))
    
    # Initialize counters and result lists
    counter = 0
    iteration_list = []
    min_cost_list = []
    average_cost_list = []
    standard_deviation_list = []

    experiment_results = pd.DataFrame(columns=['Iteration', 'MinCost'])
    
    try:
        for i in tqdm(range(iterations)):
            start_time = time.time()

            # Select individuals=tournament size, then get the min cost
            selected_permutations_1 = ga.tournament_selection(tournament_size)
            selected_permutations_2 = ga.tournament_selection(tournament_size)

            log_file.write(f"selected permutations for iteration {i} {selected_permutations_1}\n")
            log_file.write(f"selected permutations for iteration {i} {selected_permutations_2}\n")

            log_file.flush()
            candidate1 = selected_permutations_1
            candidate2 = selected_permutations_2
            log_file.write(f"candidates for iteration {i} {candidate1, candidate2}\n")
            log_file.flush()

            # Perform crossover with the specified mutation type
            offspring1, offspring2 = ga.perform_crossover(candidate1, candidate2)

            if offspring1 == candidate1 or offspring2 == candidate2:
                counter += 1
                log_file.write(f"offsprings same as parent for iteration {i} {offspring1, offspring2}\n")
                log_file.flush()

            log_file.write(f"offsprings for iteration {i} {offspring1, offspring2}\n")
            log_file.flush()

            # Perform mutation if required
            if mutation:
                mutated_offspring1 = ga.perform_mutation(offspring1, mutation_type)
                mutated_offspring2 = ga.perform_mutation(offspring2, mutation_type)
                log_file.write(f"mutated offspring for iteration {i} {mutated_offspring1, mutated_offspring2}\n")
                log_file.flush()

                # Replace the highest cost solutions with mutated offspring
                ga.replace_highest_cost_solution(mutated_offspring1)
                ga.replace_highest_cost_solution(mutated_offspring2)

            # Replace the highest cost solutions with crossover offspring
            ga.replace_highest_cost_solution(offspring1)
            ga.replace_highest_cost_solution(offspring2)

            log_file.write(f"{i} {min(ga.candidates.Cost.to_list())}\n")

            # Calculate and store metrics
            min_cost = min(ga.candidates.Cost.to_list())
            average_cost = mean(ga.candidates.Cost.to_list())
            standard_deviation = stdev(ga.candidates.Cost.to_list())

            iteration_list.append(i)
            min_cost_list.append(min_cost)
            average_cost_list.append(average_cost)
            standard_deviation_list.append(standard_deviation)

            log_file.flush()  # Flush the buffer to ensure immediate write
            end_time = time.time()
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    log_file.close()
    
    # Create a DataFrame for experiment results
    experiment_results = pd.DataFrame({'Iteration': iteration_list, 'MinCost': min_cost_list, 'AvgCost': average_cost_list, 'StdDevCost': standard_deviation_list})

    print('minimum of experiment', min(ga.candidates.Cost.to_list()))
    
    # Save results to a CSV file
    experiment_results.to_csv(f'output_csv_brazil/experiment_results_pop_size_{n}_tournament_size_{tournament_size}_iterations_{iterations}_crossover_type_{crossover_type}_mutation_type_{mutation_type}_mutationPoint_{mutation_point}_mutationTrue_{mutation}_time_taken{start_time-end_time}.csv', index=False)
    
    print(counter)

# Define the parameter ranges
n_values = [100]
tournament_size_values = [12]
iterations_values = [10000]
mutation_point_values = [2, 5, 10]
crossover_type_values = ['fix']
mutation_type_values = ['single', 'multi', 'inversion']
mutation_true_values = [True, False]

# Generate all possible parameter combinations
parameter_combinations = product(n_values, tournament_size_values, iterations_values, mutation_point_values, crossover_type_values, mutation_type_values, mutation_true_values)

# Run experiments for each parameter combination
for n, tournament_size, iterations, mutation_point, crossover_type, mutation_type, mutation in parameter_combinations:
    if tournament_size > n or ((mutation_type == 'single' or mutation_type == 'inversion') and (mutation_point > 1)):
        continue
    print('experiment>>>>>>', n, tournament_size, iterations, mutation_point, crossover_type, mutation_type, mutation)
    run_experiment(input_string=input_string, n=n, tournament_size=tournament_size, iterations=iterations, distance_matrix=distance_matrix, mutation_point=mutation_point, crossover_type=crossover_type, mutation_type=mutation_type, mutation=mutation)
