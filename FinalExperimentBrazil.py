import pandas as pd
import time
from GeneticAlgorithm import GeneticAlgorithm
from ReadXml import read_xml
from tqdm import tqdm


# single time setup
xml_file_path = '/Users/devarshigoswami/Desktop/brazil58.xml'
distance_matrix = read_xml(xml_file_path)
print(distance_matrix)
print(distance_matrix.shape)
input_string = [i for i in range(distance_matrix.shape[0])]


def run_experiment(input_string,n,tournament_size,iterations,distance_matrix,mutation_point,crossover_type,mutation):
    #create a log file for each experiment
    log_file = open(f"logs_brazil/genetic_algorithm_log.txt_{n}_{iterations}_{tournament_size}.txt", "w")
    ga = GeneticAlgorithm(input_string, n, tournament_size, distance_matrix,mutation_point,crossover_type)
    print('candidates', ga.candidates.sort_values('Cost'))
    counter=0
    iteration_list = []
    min_cost_list = []

    experiment_results = pd.DataFrame(columns=['Iteration', 'MinCost'])
    try:
        for i in tqdm(range(iterations)):
            start_time=time.time()

            # select individuals=tournament size , then get the min cost
            selected_permutations_1 = ga.tournament_selection(tournament_size)
            selected_permutations_2 = ga.tournament_selection(tournament_size)

            log_file.write(f"selected permuations for iteration {i} {selected_permutations_1}\n")
            log_file.write(f"selected permuations for iteration {i} {selected_permutations_2}\n")

            log_file.flush()
            candidate1 = selected_permutations_1
            candidate2 = selected_permutations_2
            log_file.write(f"candidates for iteration {i} {candidate1,candidate2}\n")
            log_file.flush()
            offspring1, offspring2 = ga.single_point_crossover(candidate1, candidate2)
            if offspring1==candidate1 or offspring2==candidate2:
                counter+=1
                log_file.write(f"offsprings same as parent for iteration {i} {offspring1,offspring2}\n")
                log_file.flush()
            log_file.write(f"offsprings for iteration {i} {offspring1,offspring2}\n")
            log_file.flush()
            if mutation==True:
                mutated_offspring1 = ga.single_point_swap_mutation(offspring1)
                mutated_offspring2 = ga.single_point_swap_mutation(offspring2)
                log_file.write(f"mutated offspring for iteration {i} {mutated_offspring1,mutated_offspring2}\n")
                log_file.flush()
                ga.replace_highest_cost_solution(mutated_offspring1)
                ga.replace_highest_cost_solution(mutated_offspring2)
            ga.replace_highest_cost_solution(offspring1)
            ga.replace_highest_cost_solution(offspring2)
            log_file.write(f"{i} {min(ga.candidates.Cost.to_list())}\n")
            min_cost=min(ga.candidates.Cost.to_list())
            iteration_list.append(i)
            min_cost_list.append(min_cost)
            log_file.flush()  # Flush the buffer to ensure immediate write
            end_time=time.time()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    log_file.close()
    experiment_results = pd.DataFrame({'Iteration': iteration_list, 'MinCost': min_cost_list})

    print('minimum of experiment',min(ga.candidates.Cost.to_list()))
    experiment_results.to_csv(f'output_csv_brazil/experiment_results_pop_size_{n}_tournament_size_{tournament_size}_iterations_{iterations}_crossover_type_{crossover_type}_time_taken{start_time-end_time}.csv', index=False)
    print(counter)


# run_experiment(input_string=input_string,n=50,tournament_size=2,iterations=50,distance_matrix=distance_matrix,mutation_point=1,crossover_type='fix')




from itertools import product

# Define the parameter ranges
n_values = [50, 100, 1000, 5000, 10000]
tournament_size_values = [2, 10, 20, 100, 1000]
iterations_values = [10000, 20000, 30000]
mutation_point_values = [1, 2, 5, 7]
crossover_type_values = ['fix', 'ordered']
mutation_true_values= [True,False]
# Generate all possible parameter combinations
parameter_combinations = product(n_values, tournament_size_values, iterations_values, mutation_point_values, crossover_type_values,mutation_true_values)

# Run experiments for each parameter combination
for n, tournament_size, iterations, mutation_point, crossover_type ,mutation in parameter_combinations:
    print('experiment>>>>>>',n, tournament_size, iterations, mutation_point, crossover_type, mutation)
    run_experiment(input_string=input_string, n=n, tournament_size=tournament_size, iterations=iterations, distance_matrix=distance_matrix, mutation_point=mutation_point, crossover_type=crossover_type,mutation=mutation)