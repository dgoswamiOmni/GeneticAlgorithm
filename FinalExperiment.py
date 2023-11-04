import pandas as pd
from GeneticAlgorithm import GeneticAlgorithm
from ReadXml import read_xml
import statistics
from tqdm import tqdm

# Read the XML file and parse it into a DataFrame
xml_file_path = '/Users/devarshigoswami/Desktop/brazil58.xml'

df = read_xml(xml_file_path)

# Now you have the parsed DataFrame to work with
print(df)

n = 50  # The number of permutations to generate
tournament_size = 2  # The size of the tournament for selection
input_string = [i for i in range(58)]

log_file = open("genetic_algorithm_log.txt", "w")

# Initialize the GeneticAlgorithm object
ga = GeneticAlgorithm(input_string, n, tournament_size, df,1)
print('candidates', ga.candidates.sort_values('Cost'))
log_file.write(f"{ga.candidates.sort_values('Cost')}\n")
log_file.flush()
# Call methods on the object for your experiment
counter=0
try:
    for i in tqdm(range(10000)):
        selected_permutations = ga.tournament_selection()
        log_file.write(f"selected permuations for iteration {i} {selected_permutations}\n")
        log_file.flush()
        candidate1 = selected_permutations[0]
        candidate2 = selected_permutations[1]
        log_file.write(f"candidates for iteration {i} {candidate1,candidate2}\n")
        log_file.flush()
        offspring1, offspring2 = ga.single_point_crossover(candidate1, candidate2)
        if offspring1==candidate1 or offspring2==candidate2:
            counter+=1
            log_file.write(f"offsprings same as parent for iteration {i} {offspring1,offspring2}\n")
            log_file.flush()
        log_file.write(f"offsprings for iteration {i} {offspring1,offspring2}\n")
        log_file.flush()
        mutated_offspring1 = ga.single_point_swap_mutation(offspring1)
        mutated_offspring2 = ga.single_point_swap_mutation(offspring2)
        log_file.write(f"mutated offspring for iteration {i} {mutated_offspring1,mutated_offspring2}\n")
        log_file.flush()
        ga.replace_highest_cost_solution(mutated_offspring1)
        ga.replace_highest_cost_solution(mutated_offspring2)
        log_file.write(f"{i} {min(ga.candidates.Cost.to_list())}\n")
        log_file.flush()  # Flush the buffer to ensure immediate write
except Exception as e:
    print(f"An error occurred: {str(e)}")

log_file.close()

print(min(ga.candidates.Cost.to_list()))
print(counter)