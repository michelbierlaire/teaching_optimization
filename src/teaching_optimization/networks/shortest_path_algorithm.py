""" Generic shortest path algorithm.

Michel Bierlaire
Sun Jul 7 17:39:24 2024
"""

from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from networkx import DiGraph
import logging

from teaching_optimization.networks import draw_network

logger = logging.getLogger(__name__)


class ShortestPathAlgorithm:
    """Class implementing the shortest path algorithm"""

    def __init__(
        self, the_network: DiGraph, the_cost_name: str, the_origin: Any
    ) -> None:
        self.network = the_network
        self.origin = the_origin
        self.cost_name = the_cost_name

        self.the_arcs = the_network.edges(data=True)
        # Identify the lowest cost
        arc_with_lowest_cost = min(self.the_arcs, key=lambda x: x[2][self.cost_name])
        lowest_cost = arc_with_lowest_cost[2][self.cost_name]
        self.dijkstra = False
        self.lower_bound = None
        if lowest_cost < 0:
            self.lower_bound = (the_network.number_of_edges() - 1) * lowest_cost
            logger.warning(f'Arcs with negative costs have been detected.')
        else:
            self.dijkstra = True
            logger.warning(
                f'No arc with negative cost has been detected. Dijkstra algorithm is used.'
            )
        self.labels: dict[Any:float] | None = None
        self.predecessors: dict[Any:Any] | None = None
        self.iterations: pd.DataFrame | None = None

    def shortest_path_algorithm(self) -> bool:
        """
        :return: True if the algorithm has found the shortest paths.
        """

        self.labels = {name: np.inf for name in self.network.nodes}
        self.predecessors = {name: None for name in self.network.nodes}
        self.labels[self.origin] = 0

        nodes_to_be_treated = {self.origin}

        iteration_number = 0

        reporting_iteration: list[dict[str:Any]] = list()

        while nodes_to_be_treated:
            row = {
                'Iteration': iteration_number,
                'Set': str(nodes_to_be_treated),
            }
            if self.dijkstra:
                # The node to be treated is the one with the minimum label
                current_node = min(
                    nodes_to_be_treated, key=lambda a_node: self.labels[a_node]
                )
                nodes_to_be_treated.remove(current_node)
            else:
                # Any node can be treated.
                current_node = nodes_to_be_treated.pop()
            row['Node'] = current_node
            for node, label in self.labels.items():
                row[node] = label
            reporting_iteration.append(row)
            outgoing_arcs = self.network.out_edges(current_node, data=True)
            for arc in outgoing_arcs:
                upstream_node = arc[0]
                downstream_node = arc[1]
                cost = arc[2][self.cost_name]
                if self.labels[downstream_node] > self.labels[upstream_node] + cost:
                    self.labels[downstream_node] = self.labels[upstream_node] + cost
                    self.predecessors[downstream_node] = upstream_node
                    if not self.dijkstra:
                        # We need to verify that there is not negative cost cycle.
                        if (
                            self.labels[downstream_node] < 0
                            and self.labels[downstream_node] < self.lower_bound
                        ):
                            print('The network contains a cycle with negative cost.')
                            self.iterations = pd.DataFrame(reporting_iteration)
                            self.predecessors = None
                            self.labels = None
                            return False
                    nodes_to_be_treated.add(downstream_node)
            iteration_number += 1

        row = {'Iteration': iteration_number, 'Set': '{}', 'Node': ''}
        for node, label in self.labels.items():
            row[node] = label
        reporting_iteration.append(row)
        self.iterations = pd.DataFrame(reporting_iteration)
        return True

    def recursive_shortest_path(self, node: Any) -> str:
        """Print the shortest path to a given node, recursively"""

        if self.predecessors is None:
            logger.warning('Shortest paths have not been established.')
            return str()

        if self.predecessors[node] is None:
            return str(node)

        return f'{self.recursive_shortest_path(node=self.predecessors[node])} -> {str(node)}'

    def list_of_shortest_paths(self) -> list[str]:
        """Print the list of shortest paths from the origin to each node"""
        return [self.recursive_shortest_path(node=node) for node in self.network.nodes]

    def plot_shortest_path_tree(self) -> None:
        """Plot the shortest path tree"""
        shortest_path_arcs = [
            (upstream, downstream)
            for downstream, upstream in self.predecessors.items()
            if upstream is not None
        ]

        shortest_path_tree: DiGraph = DiGraph()
        shortest_path_tree.add_nodes_from(self.network.nodes(data=True))
        shortest_path_tree.add_edges_from(shortest_path_arcs)
        fig, ax = plt.subplots(figsize=(8, 6))
        draw_network(the_network=shortest_path_tree, ax=ax)
        plt.show()
