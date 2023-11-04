import pandas as pd
from GeneticAlgorithm import GeneticAlgorithm
from ReadXml import read_xml


# Read the XML file and parse it into a DataFrame
xml_file_path = '/Users/devarshigoswami/Desktop/brazil58.xml'

df = read_xml(xml_file_path)

# Now you have the parsed DataFrame to work with
print(df)

n = 10  # The number of permutations to generate
tournament_size = 2  # The size of the tournament for selection
# input_string=''.join(df.Source.unique())
input_string=[1,2,3,5]
# Initialize the GeneticAlgorithm object
ga = GeneticAlgorithm(input_string, n, tournament_size, df)
print('candidates',ga.candidates.sort_values('Cost'))
# Call methods on the object for your experiment
selected_permutations = ga.tournament_selection()
print('tournament selection done',selected_permutations)

candidate1 = selected_permutations[0]
candidate2 = selected_permutations[1]
offspring1, offspring2 = ga.single_point_crossover(candidate1, candidate2)
print('single point crossover done',offspring1,offspring2)
mutated_offspring1 = ga.single_point_swap_mutation(offspring1)
mutated_offspring2 = ga.single_point_swap_mutation(offspring2)
print('mutation done',mutated_offspring1,mutated_offspring2)
ga.replace_highest_cost_solution(mutated_offspring2)

# Access the candidates DataFrame if needed
print(ga.candidates)
