""" Bellman subnetwork and shortest path trees

Michel Bierlaire
Sun Jul 7 18:20:50 2024
"""

import itertools
import logging
from typing import Any

import numpy as np
from networkx import DiGraph

logger = logging.getLogger(__name__)


class BellmanSubnetworks:
    """Class in charge of generating the Bellman subnetworks"""

    def __init__(self, a_network: DiGraph, cost_name: str, label_name: str):
        self.network = a_network
        self.cost_name = cost_name
        self.label_name = label_name

    def bellman_arcs(self, a_node: Any) -> list[tuple[Any, Any]]:
        """Identifies the Bellman arcs of a node

        :param a_node: the node under interest
        :return: a list of arcs
        """

        optimal_arcs = [
            (upstream, a_node)
            for upstream, _ in self.network.in_edges(a_node)
            if np.isclose(
                self.network.nodes[upstream][self.label_name]
                + self.network[upstream][a_node][self.cost_name],
                self.network.nodes[a_node][self.label_name],
            )
        ]
        return optimal_arcs

    def create_bellman_subnetworks(self, maximum_size=1000) -> list[DiGraph]:
        """Create the list, up to a maximum size to avoid exponential explosion"""

        the_bellman_arcs = {
            node: self.bellman_arcs(
                a_node=node,
            )
            for node in self.network.nodes
        }

        # Consider all possible combinations
        all_lists = [a_list for a_list in the_bellman_arcs.values() if a_list]
        all_combinations = list(itertools.product(*all_lists))

        all_subnetworks = []
        # Let's create the Bellman's subnetworks
        for one_instance in itertools.product(*all_lists):

            # Create a directed graph
            a_graph: DiGraph = DiGraph()

            # Add nodes to the graph
            a_graph.add_nodes_from(self.network.nodes(data=True))

            # Add arcs to the graph
            a_graph.add_edges_from(one_instance)

            all_subnetworks.append(a_graph)
            if len(all_subnetworks) >= maximum_size:
                logger.warning(
                    f'Maximum number of Bellman subnetworks reached: {maximum_size}'
                )
                return all_subnetworks

        return all_subnetworks
