"""File test_bellman.py

Michel Bierlaire
Sun Jul 7 18:56:03 2024

Tests for the Bellman subnetwork
"""

import unittest

from networkx import DiGraph, set_node_attributes
from teaching_optimization.networks.bellman import BellmanSubnetworks


class TestShortestPathAlgorithmUnbounded(unittest.TestCase):

    def setUp(self):

        nodes = ['a', 'b', 'c', 'd']

        arcs = [
            ('a', 'b', 1),
            ('a', 'c', 1),
            ('b', 'd', 1),
            ('c', 'd', 1),
            ('b', 'c', 0),
        ]

        labels = {'a': 0, 'b': 1, 'c': 1, 'd': 2}

        self.the_network: DiGraph = DiGraph()
        for node in nodes:
            self.the_network.add_node(node)
        set_node_attributes(self.the_network, labels, name='label')
        self.the_network.add_weighted_edges_from(arcs, weight='cost')

        self.bellman = BellmanSubnetworks(
            a_network=self.the_network, cost_name='cost', label_name='label'
        )

    def test_bellman_arcs(self):
        the_bellman_arcs = {
            node: self.bellman.bellman_arcs(
                a_node=node,
            )
            for node in self.the_network.nodes
        }
        expected_result = {
            'a': [],
            'b': [('a', 'b')],
            'c': [('a', 'c'), ('b', 'c')],
            'd': [('b', 'd'), ('c', 'd')],
        }
        self.assertDictEqual(expected_result, the_bellman_arcs)

    def test_bellman_subnetwork(self):
        subnetworks = self.bellman.create_bellman_subnetworks()
        self.assertEqual(len(subnetworks), 4)


if __name__ == '__main__':
    unittest.main()
