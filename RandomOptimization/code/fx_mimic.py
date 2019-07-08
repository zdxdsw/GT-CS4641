import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_tree
import random

'''
target function:
8 * math.sin(0.06 * x) + 8 * math.cos(0.14 * x) + 8 * math.exp(math.cos(0.2*x))

'''

class OptProb:
    """Base class for optimisation problems.

    Parameters
    ----------
    length: int
        Number of elements in state vector.

    maximize: bool, default: True
        Whether to maximize the fitness function.
        Set :code:`False` for minimization problem.
    """

    def __init__(self, length, maximize=True):

        self.length = length
        self.state = [random.randint(0, 1) for i in range(self.length)]
        self.neighbors = []
        self.fitness = 0
        self.population = []
        self.pop_fitness = []
        self.mate_probs = []

        if maximize:
            self.maximize = 1
        else:
            self.maximize = -1

    def best_child(self):
        """Return the best state in the current population.

        Returns
        -------
        best: array
            State vector defining best child.
        """
        best = self.population[np.argmax(self.pop_fitness)]

        return best

    def best_neighbor(self):
        """Return the best neighbor of current state.

        Returns
        -------
        best: array
            State vector defining best neighbor.
        """
        fitness_list = []

        for neigh in self.neighbors:
            fitness = self.eval_fitness(neigh)
            fitness_list.append(fitness)

        best = self.neighbors[np.argmax(fitness_list)]

        return best

    def eval_fitness(self, state):
        """Evaluate the fitness of a state vector.

        Parameters
        ----------
        state: array
            State vector for evaluation.

        Returns
        -------
        fitness: float
            Value of fitness function.
        """
        x = binary2decimal(state, self.length)
        if (x<-63) or (x>64):
            raise Exception("state length must match problem length")
        
        fitness = self.maximize*(8 * math.sin(0.06 * x) + 8 * math.cos(0.14 * x) + 8 * math.exp(math.cos(0.2*x)))

        return fitness

    def eval_mate_probs(self):
        """
        Calculate the probability of each member of the population reproducing.
        """
        pop_fitness = np.copy(self.pop_fitness)

        # Set -1*inf values to 0 to avoid dividing by sum of infinity.
        # This forces mate_probs for these pop members to 0.
        pop_fitness[pop_fitness == -1.0*np.inf] = 0

        if np.sum(pop_fitness) == 0:
            self.mate_probs = np.ones(len(pop_fitness)) \
                              / len(pop_fitness)
        else:
            self.mate_probs = pop_fitness/np.sum(pop_fitness)

    def get_fitness(self):
        """ Return the fitness of the current state vector."""
        return self.fitness

    def get_mate_probs(self):
        """ Return the population mate probabilities."""
        return self.mate_probs

    def get_maximize(self):
        """ Return the maximization multiplier."""
        return self.maximize

    def get_pop_fitness(self):
        """ Return the current population fitness array."""
        return self.pop_fitness

    def get_population(self):
        """ Return the current population."""
        return self.population

    def get_state(self):
        """ Return the current state vector."""
        return self.state

    def set_population(self, new_population):
        """ Change the current population to a specified new population and get
        the fitness of all members.

        Parameters
        ----------
        new_population: array
            Numpy array containing new population.
        """
        self.population = new_population

        # Calculate fitness
        pop_fitness = []

        for i in range(len(self.population)):
            fitness = self.eval_fitness(self.population[i])
            pop_fitness.append(fitness)

        self.pop_fitness = np.array(pop_fitness)

    def set_state(self, new_state):
        """
        Change the current state vector to a specified value
        and get its fitness.

        Parameters
        ----------
        new_state: array
            New state vector value.
        """
        self.state = new_state
        new_fitness = self.eval_fitness(self.state)
        self.fitness = new_fitness

def binary2decimal(binary, chromosome_length):
    t = 0
    for j in range(len(binary)):
        t += binary[j] * 2 ** j
    t -= 63
    return t

class fxOpt(OptProb):
    """Class for defining discrete-state optimization problems.

    Parameters
    ----------
    maximize: bool, default: True
        Whether to maximize the fitness function.
        Set :code:`False` for minimization problem.

    max_val: int, default: 2
        Number of unique values that each element in the state vector
        can take. Assumes values are integers in the range 0 to
        (max_val - 1), inclusive.
    """

    def __init__(self, length, maximize=True, max_val=2):

        OptProb.__init__(self, length, maximize)

        if max_val < 0:
            raise Exception("""max_val must be a positive integer.""")
        elif not isinstance(max_val, int):
            if max_val.is_integer():
                self.max_val = int(max_val)
            else:
                raise Exception("""max_val must be a positive integer.""")
        else:
            self.max_val = max_val
        self.length = length
        self.keep_sample = []
        self.node_probs = np.zeros([self.length, self.max_val, self.max_val])
        self.parent_nodes = []
        self.sample_order = []
        self.prob_type = 'discrete'
    
    def eval_node_probs(self):
        """Update probability density estimates.
        """
        # Create mutual info matrix
        mutual_info = np.zeros([self.length, self.length])
        for i in range(self.length - 1):
            for j in range(i + 1, self.length):
                mutual_info[i, j] = -1 * mutual_info_score(
                    self.keep_sample[:, i],
                    self.keep_sample[:, j])

        # Find minimum spanning tree of mutual info matrix
        mst = minimum_spanning_tree(csr_matrix(mutual_info))

        # Convert minimum spanning tree to depth first tree with node 0 as root
        dft = depth_first_tree(csr_matrix(mst.toarray()), 0, directed=False)
        dft = np.round(dft.toarray(), 10)

        # Determine parent of each node
        parent = np.argmin(dft[:, 1:], axis=0)

        # Get probs
        probs = np.zeros([self.length, self.max_val, self.max_val])

        probs[0, :] = np.histogram(self.keep_sample[:, 0],
                                   np.arange(self.max_val + 1),
                                   density=True)[0]

        for i in range(1, self.length):
            for j in range(self.max_val):
                subset = self.keep_sample[np.where(
                    self.keep_sample[:, parent[i - 1]] == j)[0]]

                if not len(subset):
                    probs[i, j] = 1/self.max_val
                else:
                    probs[i, j] = np.histogram(subset[:, i],
                                               np.arange(self.max_val + 1),
                                               density=True)[0]

        # Update probs and parent
        self.node_probs = probs
        self.parent_nodes = parent

    def find_neighbors(self):
        """Find all neighbors of the current state.
        """
        self.neighbors = []

        if self.max_val == 2:
            for i in range(self.length):
                neighbor = np.copy(self.state)
                neighbor[i] = np.abs(neighbor[i] - 1)
                self.neighbors.append(neighbor)

        else:
            for i in range(self.length):
                vals = list(np.arange(self.max_val))
                vals.remove(self.state[i])

                for j in vals:
                    neighbor = np.copy(self.state)
                    neighbor[i] = j
                    self.neighbors.append(neighbor)

    def find_sample_order(self):
        """Determine order in which to generate sample vector elements.
        """
        sample_order = []
        last = [0]
        parent = np.array(self.parent_nodes)

        while len(sample_order) < self.length:
            inds = []

            # If last nodes list is empty, select random node than has not
            # previously been selected
            if len(last) == 0:
                inds = [np.random.choice(list(set(np.arange(self.length)) -
                                              set(sample_order)))]
            else:
                for i in last:
                    inds += list(np.where(parent == i)[0] + 1)

            sample_order += last
            last = inds

        self.sample_order = sample_order

    def find_top_pct(self, keep_pct):
        """Select samples with fitness in the top keep_pct percentile.

        Parameters
        ----------
        keep_pct: float
            Proportion of samples to keep.
        """
        if (keep_pct < 0) or (keep_pct > 1):
            raise Exception("""keep_pct must be between 0 and 1.""")

        # Determine threshold
        theta = np.percentile(self.pop_fitness, 100*(1 - keep_pct))

        # Determine samples for keeping
        keep_inds = np.where(self.pop_fitness >= theta)[0]

        # Determine sample for keeping
        self.keep_sample = self.population[keep_inds]

    def get_keep_sample(self):
        """ Return the keep sample.

        Returns
        -------
        self.keep_sample: array
            Numpy array containing samples with fitness in the top keep_pct
            percentile.
        """
        return self.keep_sample

    def get_prob_type(self):
        """ Return the problem type.

        Returns
        -------
        self.prob_type: string
            Returns problem type.
        """
        return self.prob_type

    def random(self):
        """Return a random state vector.

        Returns
        -------
        state: array
            Randomly generated state vector.
        """
        state = np.random.randint(0, self.max_val, self.length)

        return state

    def random_neighbor(self):
        """Return random neighbor of current state vector.

        Returns
        -------
        neighbor: array
            State vector of random neighbor.
        """
        neighbor = np.copy(self.state)
        i = np.random.randint(0, self.length)

        if self.max_val == 2:
            neighbor[i] = np.abs(neighbor[i] - 1)

        else:
            vals = list(np.arange(self.max_val))
            vals.remove(neighbor[i])
            neighbor[i] = vals[np.random.randint(0, self.max_val-1)]

        return neighbor

    def random_pop(self, pop_size):
        """Create a population of random state vectors."""
        if pop_size <= 0:
            raise Exception("""pop_size must be a positive integer.""")
        elif not isinstance(pop_size, int):
            if pop_size.is_integer():
                pop_size = int(pop_size)
            else:
                raise Exception("""pop_size must be a positive integer.""")

        population = []
        pop_fitness = []

        for _ in range(pop_size):
            state = [random.randint(0, 1) for i in range(self.length)]
            fitness = self.eval_fitness(state)

            population.append(state)
            pop_fitness.append(fitness)

        self.population = np.array(population)
        self.pop_fitness = np.array(pop_fitness)

    def reproduce(self, parent_1, parent_2, mutation_prob=0.1):
        """Create child state vector from two parent state vectors.

        Parameters
        ----------
        parent_1: array
            State vector for parent 1.
        parent_2: array
            State vector for parent 2.
        mutation_prob: float
            Probability of a mutation at each state element during
            reproduction.

        Returns
        -------
        child: array
            Child state vector produced from parents 1 and 2.
        """
        if len(parent_1) != self.length or len(parent_2) != self.length:
            raise Exception("""Lengths of parents must match problem length""")

        if (mutation_prob < 0) or (mutation_prob > 1):
            raise Exception("""mutation_prob must be between 0 and 1.""")

        # Reproduce parents
        _n = np.random.randint(self.length - 1)
        child = np.array([0]*self.length)
        child[0:_n+1] = parent_1[0:_n+1]
        child[_n+1:] = parent_2[_n+1:]

        # Mutate child
        rand = np.random.uniform(size=self.length)
        mutate = np.where(rand < mutation_prob)[0]

        if self.max_val == 2:
            for i in mutate:
                child[i] = np.abs(child[i] - 1)

        else:
            for i in mutate:
                vals = list(np.arange(self.max_val))
                vals.remove(child[i])
                child[i] = vals[np.random.randint(0, self.max_val-1)]

        return child

    def reset(self):
        """Set the current state vector to a random value and get its fitness.
        """
        self.state = [random.randint(0, 1) for i in range(self.length)]
        self.fitness = self.eval_fitness(self.state)

    def sample_pop(self, sample_size):
        """Generate new sample from probability density.

        Parameters
        ----------
        sample_size: int
            Size of sample to be generated.

        Returns
        -------
        new_sample: array
            Numpy array containing new sample.
        """
        if sample_size <= 0:
            raise Exception("""sample_size must be a positive integer.""")
        elif not isinstance(sample_size, int):
            if sample_size.is_integer():
                sample_size = int(sample_size)
            else:
                raise Exception("""sample_size must be a positive integer.""")

        # Initialize new sample matrix
        new_sample = np.zeros([sample_size, self.length])

        # Get value of first element in new samples
        new_sample[:, 0] = np.random.choice(self.max_val, sample_size,
                                            p=self.node_probs[0, 0])

        # Get sample order
        self.find_sample_order()
        sample_order = self.sample_order[1:]

        # Get values for remaining elements in new samples
        for i in sample_order:
            par_ind = self.parent_nodes[i-1]

            for j in range(self.max_val):
                inds = np.where(new_sample[:, par_ind] == j)[0]
                new_sample[inds, i] = np.random.choice(self.max_val,
                                                       len(inds),
                                                       p=self.node_probs[i, j])

        return new_sample

import mlrose
import numpy as np
import math
import matplotlib.pyplot as plt
problem = fxOpt(length=7, maximize=True, max_val=2)
optimum_list = []
correct = 0
n=100
for i in range(n):
    best_x, best_y, next_fitness_list= mlrose.mimic(problem, pop_size=50, keep_pct=0.8, max_attempts=50, max_iters=100)
    print(best_x)
    optimum_list.append([binary2decimal(best_x, 7), best_y])
    if (binary2decimal(best_x, 7) == 0):
        correct += 1
print ("accuracy = ", correct, " / ", n)
X = [i for i in range(len(next_fitness_list))]
Y = [next_fitness_list[i] for i in range(len(next_fitness_list))]
plt.plot(X, Y)
plt.show()
for i in range(n):
        plt.scatter(optimum_list[i][0], optimum_list[i][1], c='r', s=10)
X1 = [i for i in range(-63, 65, 1)]
Y1 = [8 * math.sin(0.06 * x) + 8 * math.cos(0.14 * x) + 8 * math.exp(math.cos(0.2*x)) for x in X1]
plt.plot(X1, Y1)
plt.show()


