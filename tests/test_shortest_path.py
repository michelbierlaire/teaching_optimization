"""File test_shortest_path.py

Michel Bierlaire
Sun Jul 7 17:55:05 2024

Tests for the shortest path algorithm
"""

import unittest

import numpy as np
import pandas as pd
from networkx import DiGraph

from teaching_optimization.networks.shortest_path_algorithm import (
    ShortestPathAlgorithm,
)


class TestShortestPathAlgorithmUnbounded(unittest.TestCase):

    def setUp(self):
        self.positions = {
            'a': (0, 0),
            'b': (2, 1.5),
            'c': (2, -1.5),
            'd': (4, 1.5),
            'e': (4, -1.5),
            'f': (6, 0),
        }

        self.nodes = list(self.positions.keys())

        self.arcs = [
            ('a', 'b', -1),
            ('a', 'c', 3),
            ('b', 'd', 7),
            ('b', 'e', 5),
            ('c', 'b', -9),
            ('e', 'c', -7),
            ('d', 'e', 4),
            ('d', 'f', 3),
            ('e', 'f', -2),
        ]

        self.the_network = DiGraph()
        for node in self.nodes:
            self.the_network.add_node(node, pos=self.positions[node])
        self.the_network.add_weighted_edges_from(self.arcs, weight='cost')

    def test_shortest_path_algorithm(self):
        expected_labels = None

        expected_first_iteration = pd.DataFrame(
            [
                {
                    'Iteration': 0,
                    'Set': "{'a'}",
                    'Node': 'a',
                    'a': 0,
                    'b': np.inf,
                    'c': np.inf,
                    'd': np.inf,
                    'e': np.inf,
                    'f': np.inf,
                },
            ]
        )

        the_algorithm = ShortestPathAlgorithm(
            the_network=self.the_network, the_cost_name='cost', the_origin='a'
        )
        the_algorithm.shortest_path_algorithm()

        self.assertEqual(the_algorithm.labels, expected_labels)
        pd.testing.assert_frame_equal(
            the_algorithm.iterations.iloc[:1], expected_first_iteration
        )


class TestShortestPathAlgorithmBounded(unittest.TestCase):

    def setUp(self):
        positions = {
            'a': (0, 1),
            'b': (1, 2),
            'c': (1, 0),
            'd': (2, 2),
            'e': (3, 1),
            'f': (2, 0),
            'g': (4, 2),
            'h': (4, 0),
            'i': (5, 1),
        }

        nodes = list(positions.keys())

        arcs = [
            ('a', 'b', 10),
            ('a', 'c', 12),
            ('b', 'd', -12),
            ('b', 'f', 4),
            ('c', 'd', 7),
            ('c', 'b', 8),
            ('c', 'f', 6),
            ('d', 'g', 16),
            ('d', 'e', -3),
            ('f', 'd', 7),
            ('e', 'g', 7),
            ('e', 'i', -1),
            ('e', 'h', 6),
            ('e', 'f', -4),
            ('f', 'h', 15),
            ('g', 'i', 8),
            ('h', 'i', 5),
        ]

        self.the_network: DiGraph = DiGraph()
        for node in nodes:
            self.the_network.add_node(node, pos=positions[node])
        self.the_network.add_weighted_edges_from(arcs, weight='cost')

    def test_shortest_path_algorithm(self):
        expected_labels = {
            'a': 0,
            'b': 10,
            'c': 12,
            'd': -2,
            'e': -5,
            'f': -9,
            'g': 2,
            'h': 1,
            'i': -6,
        }

        first_iteration = pd.DataFrame(
            [
                {
                    'Iteration': 0,
                    'Set': "{'a'}",
                    'Node': 'a',
                    'a': 0,
                    'b': np.inf,
                    'c': np.inf,
                    'd': np.inf,
                    'e': np.inf,
                    'f': np.inf,
                    'g': np.inf,
                    'h': np.inf,
                    'i': np.inf,
                },
            ]
        )

        the_algorithm = ShortestPathAlgorithm(
            the_network=self.the_network, the_cost_name='cost', the_origin='a'
        )
        the_algorithm.shortest_path_algorithm()

        self.assertEqual(the_algorithm.labels, expected_labels)
        pd.testing.assert_frame_equal(
            the_algorithm.iterations.iloc[:1], first_iteration
        )

        the_shortest_paths = the_algorithm.list_of_shortest_paths()
        expected_list = [
            'a',
            'a -> b',
            'a -> c',
            'a -> b -> d',
            'a -> b -> d -> e',
            'a -> b -> d -> e -> f',
            'a -> b -> d -> e -> g',
            'a -> b -> d -> e -> h',
            'a -> b -> d -> e -> i',
        ]

        self.assertListEqual(expected_list, the_shortest_paths)


if __name__ == '__main__':
    unittest.main()
