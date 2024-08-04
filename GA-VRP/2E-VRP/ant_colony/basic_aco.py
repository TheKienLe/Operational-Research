import numpy as np
import random
from vrp_base import VrptwGraph, PathMessage
from vrp_aco_figure import VrptwAcoFigure
from ant import Ant
from threading import Thread
from queue import Queue
import time

class BasicACO:
    def __init__(self, graph: VrptwGraph, ants_num=10, max_iter=200, beta=2, q0=0.1,
                 whether_or_not_to_show_figure=True):
        """
        Initializes the BasicACO class with the given parameters.

        Parameters:
        graph (VrptwGraph): The VRPTW graph.
        ants_num (int): The number of ants. Default is 10.
        max_iter (int): The maximum number of iterations. Default is 200.
        beta (float): The importance of heuristic information. Default is 2.
        q0 (float): The probability of choosing the next node directly with the highest probability. Default is 0.1.
        whether_or_not_to_show_figure (bool): Whether to show the figure. Default is True.
        """
        super()
        self.graph = graph  # The graph representing the problem
        self.ants_num = ants_num  # Number of ants
        self.max_iter = max_iter  # Maximum number of iterations
        self.max_load = graph.vehicle_capacity  # Maximum load for vehicles
        self.beta = beta  # Importance of heuristic information
        self.q0 = q0  # Probability of choosing the next node directly
        self.best_path_distance = None  # Best path distance found
        self.best_path = None  # Best path found
        self.best_vehicle_num = None  # Best number of vehicles used
        self.whether_or_not_to_show_figure = whether_or_not_to_show_figure  # Flag to show figure

    def run_basic_aco(self):
        """
        Runs the Basic ACO algorithm, optionally showing a figure of the progress.

        Example:
        --------
        >>> from vrptw_base import VrptwGraph
        >>> graph = VrptwGraph()
        >>> aco = BasicACO(graph)
        >>> aco.run_basic_aco()
        """
         # Queue for path messages to the figure
        path_queue_for_figure = Queue()
        # Start a thread to run the ACO algorithm
        basic_aco_thread = Thread(target=self._basic_aco, args=(path_queue_for_figure,))
        basic_aco_thread.start()

        # Show figure if enabled
        if self.whether_or_not_to_show_figure:
            figure = VrptwAcoFigure(self.graph.nodes, path_queue_for_figure)
            figure.run()
        basic_aco_thread.join()

        # Signal end of path messages
        if self.whether_or_not_to_show_figure:
            path_queue_for_figure.put(PathMessage(None, None))

    def _basic_aco(self, path_queue_for_figure: Queue):
        """
        The core of the Basic ACO algorithm.

        Parameters:
        path_queue_for_figure (Queue): Queue for communicating paths to the figure.

        Example:
        --------
        >>> from queue import Queue
        >>> path_queue = Queue()
        >>> from vrptw_base import VrptwGraph
        >>> graph = VrptwGraph()
        >>> aco = BasicACO(graph)
        >>> aco._basic_aco(path_queue)
        """
        start_time_total = time.time()  # Record start time
        start_iteration = 0  # Initial iteration

        for iter in range(self.max_iter):
            # Create ants
            ants = list(Ant(self.graph) for _ in range(self.ants_num))  
            
            for k in range(self.ants_num):
                
                while not ants[k].index_to_visit_empty():
                    # Select next node
                    next_index = self.select_next_index(ants[k])  
                    
                    # Check constraints
                    if not ants[k].check_condition(next_index):  
                        next_index = self.select_next_index(ants[k])
                        if not ants[k].check_condition(next_index):
                            next_index = 0  # Return to depot

                    # Move to next node
                    ants[k].move_to_next_index(next_index)  
                    
                    # Local pheromone update
                    self.graph.local_update_pheromone(ants[k].current_index, next_index)  
                
                # Return to depot
                ants[k].move_to_next_index(0)  

                # Local pheromone update
                self.graph.local_update_pheromone(ants[k].current_index, 0)  

            # Calculate path distances
            paths_distance = np.array([ant.total_travel_distance for ant in ants])  
            
            # Find best path
            best_index = np.argmin(paths_distance)  
            if self.best_path is None or paths_distance[best_index] < self.best_path_distance:
                # Update best path
                self.best_path = ants[int(best_index)].travel_path  
                self.best_path_distance = paths_distance[best_index]
                
                # Update best vehicle number
                self.best_vehicle_num = self.best_path.count(0) - 1  
                start_iteration = iter

                if self.whether_or_not_to_show_figure:
                    path_queue_for_figure.put(PathMessage(self.best_path, self.best_path_distance))  # Send path to figure

                print('\n')
                print('[iteration %d]: find a improved path, its distance is %f' % (iter, self.best_path_distance))
                print('it takes %0.3f second multiple_ant_colony_system running' % (time.time() - start_time_total))

    def select_next_index(self, ant):
        """
        Selects the next node for the ant to visit based on the transition probabilities.

        Parameters:
        ant (Ant): The ant for which the next node is being selected.

        Returns:
        int: The index of the next node to visit.

        Example:
        --------
        >>> import numpy as np
        >>> from vrptw_base import VrptwGraph, PathMessage
        >>> from ant import Ant
        >>> 
        >>> # Mock classes for the example
        >>> class MockGraph:
        ...     def __init__(self):
        ...         self.pheromone_mat = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [0.2, 0.1, 0.3]])
        ...         self.heuristic_info_mat = np.array([[1, 2, 3], [3, 2, 1], [2, 1, 3]])
        ... 
        >>> class MockAnt:
        ...     def __init__(self):
        ...         self.current_index = 0
        ...         self.index_to_visit = [1, 2]
        ... 
        >>> graph = MockGraph()
        >>> ant = MockAnt()
        >>> aco = BasicACO(graph)
        >>> np.random.seed(42)  # For reproducibility
        >>> random.seed(42)  # For reproducibility
        >>> next_index = aco.select_next_index(ant)
        >>> print(next_index)
        2
        """
        # Current node
        current_index = ant.current_index  
        
        # Nodes left to visit
        index_to_visit = ant.index_to_visit  

        # Calculate transition probabilities
        transition_prob = self.graph.pheromone_mat[current_index][index_to_visit] * \
            np.power(self.graph.heuristic_info_mat[current_index][index_to_visit], self.beta)
        transition_prob = transition_prob / np.sum(transition_prob)

        if np.random.rand() < self.q0:
            # Choose node with highest probability
            max_prob_index = np.argmax(transition_prob)  
            next_index = index_to_visit[max_prob_index]
        else:
            # Roulette wheel selection
            next_index = BasicACO.stochastic_accept(index_to_visit, transition_prob)  
        return next_index

    @staticmethod
    def stochastic_accept(index_to_visit, transition_prob):
        """
        Performs roulette wheel selection to choose the next node.

        Parameters:
        index_to_visit (list or tuple): A list of indices to choose from.
        transition_prob (np.array): The transition probabilities for each index.

        Returns:
        int: The selected index.

        Example:
        --------
        >>> import numpy as np
        >>> import random
        >>> np.random.seed(42)  # For reproducibility
        >>> random.seed(42)  # For reproducibility
        >>> indices = [1, 2, 3]
        >>> probs = np.array([0.2, 0.5, 0.3])
        >>> selected_index = BasicACO.stochastic_accept(indices, probs)
        >>> print(selected_index)
        2
        """
        # Number of nodes
        N = len(index_to_visit)

        # Sum of probabilities
        sum_tran_prob = np.sum(transition_prob)  

        # Normalize probabilities
        norm_transition_prob = transition_prob / sum_tran_prob  

        while True:
            # Select a node randomly
            ind = int(N * random.random())  

            # Accept based on probability
            if random.random() <= norm_transition_prob[ind]:  
                return index_to_visit[ind]