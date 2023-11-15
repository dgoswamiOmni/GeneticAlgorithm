
## Genetic Algorithms Project

This project implements a genetic algorithm for solving optimization problems, with a focus on solving a specific problem using a genetic algorithm approach. The problem involves finding an optimal solution for a given graph representation.

### Libraries Used

1. **xml.etree.ElementTree**: This library is used for parsing XML files to extract information about the graph representation.

2. **numpy (np)**: NumPy is utilized for efficient manipulation of arrays and matrices, particularly in constructing and working with the distance matrix.

3. **pandas (pd)**: Pandas is employed for handling and analyzing data through the use of DataFrames. It plays a crucial role in storing and processing the results of the genetic algorithm.

4. **time**: The time module is used for measuring the execution time of the genetic algorithm experiments.

5. **GeneticAlgorithm.py**: A custom Python file containing the implementation of the GeneticAlgorithm class, responsible for initializing and managing the genetic algorithm.

6. **ReadXml.py**: A Python file providing the `read_xml` function, which parses an XML file representing a graph and returns a distance matrix.

7. **tqdm**: This library is used to display a progress bar for iterations, providing a visual indication of the algorithm's progress.

8. **statistics (stdev, mean)**: These functions from the statistics module are utilized for calculating the standard deviation and mean of the costs during the genetic algorithm experiments.

9. **itertools.product**: This function generates Cartesian product of input iterables, and in this project, it is used to create combinations of different parameters for running experiments.

10. **matplotlib.pyplot (plt)**: Matplotlib is used for creating convergence plots to visualize the performance of the genetic algorithm over iterations.

### Python Files

1. **GeneticAlgorithm.py**: Contains the implementation of the `GeneticAlgorithm` class, responsible for managing the genetic algorithm operations such as selection, crossover, mutation, and replacement.

2. **ReadXml.py**: Provides the `read_xml` function, which reads an XML file containing graph information and returns a distance matrix.

3. **Main.py (or your main script)**: The main script where the genetic algorithm experiments are run, and convergence plots are generated.

### GeneticAlgorithm Class and Its Class Variables

The `GeneticAlgorithm` class is designed to encapsulate the genetic algorithm operations. Key class variables include:

- **input_string**: A representation of the graph vertices.
- **n**: Population size, representing the number of individuals in each generation.
- **tournament_size**: The size of the tournament during selection.
- **distance_matrix**: The matrix representing distances between graph vertices.
- **mutation_point**: The parameter determining the mutation point during mutation.
- **crossover_type**: The type of crossover operation used (e.g., 'fix').
- **mutation_type**: The type of mutation operation used (e.g., 'single', 'multi', 'inversion').
- **candidates**: A DataFrame storing the current population of individuals, including their costs.

### Experiments Using itertools.product

The experiments are conducted by iterating over different combinations of parameters using `itertools.product`. This allows for systematic exploration of the parameter space, varying population size, tournament size, iterations, mutation point, crossover type, and mutation type.

### Logs Creation

Logs are created using a log file for each experiment. Information such as selected permutations, candidates, offspring, and other relevant details are recorded in the log file. This logging process aids in understanding the behavior of the algorithm during each iteration.

### CSV Creation and Analysis

Results of each experiment, including iteration number, minimum cost, average cost, and standard deviation, are stored in a CSV file. This facilitates further analysis of the algorithm's performance and allows for easy comparison between different parameter settings.

Convergence plots are generated using matplotlib to visualize the algorithm's behavior over iterations for varying parameters.

For any questions or clarifications, please refer to the code documentation or contact the project contributors.


### Please run the FinalExperimentBrazil.py and FinalExperimentBurma.py to run the genetic algorithm
