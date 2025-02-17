import pandas as pd
from torch.utils.data import Dataset
import networkx as nx
import igraph as ig


class PrimeKGDataset(Dataset):
    def __init__(self, path):
        self.path = path
        df = pd.read_csv(path)
        self.nodes, self.edges, self.edge_index = self._preprocess(df)
        self.graph = self._create_graph()

    def _preprocess(self, df):
        nodes = pd.concat(
            [
                df.get(["x_id", "x_type", "x_name", "x_source"]).rename(
                    columns={
                        "x_id": "node_id",
                        "x_type": "node_type",
                        "x_name": "node_name",
                        "x_source": "node_source",
                    }
                ),
                df.get(["y_id", "y_type", "y_name", "y_source"]).rename(
                    columns={
                        "y_id": "node_id",
                        "y_type": "node_type",
                        "y_name": "node_name",
                        "y_source": "node_source",
                    }
                ),
            ]
        )

        nodes = (
            nodes.drop_duplicates()
            .reset_index()
            .drop("index", axis=1)
            .reset_index()
            .rename(columns={"index": "node_idx"})
        )

        edges = pd.merge(
            df,
            nodes,
            "left",
            left_on=["x_id", "x_type", "x_name", "x_source"],
            right_on=["node_id", "node_type", "node_name", "node_source"],
        )
        edges = edges.rename(columns={"node_idx": "x_idx"})
        edges = pd.merge(
            edges,
            nodes,
            "left",
            left_on=["y_id", "y_type", "y_name", "y_source"],
            right_on=["node_id", "node_type", "node_name", "node_source"],
        )
        edges = edges.rename(columns={"node_idx": "y_idx"})

        edge_index = edges.get(["x_idx", "y_idx"]).values.T
        return nodes, edges, edge_index

    def _create_graph(self):
        graph = ig.Graph()
        graph.add_vertices(list(range(self.nodes.shape[0])))
        graph.add_edges([tuple(x) for x in self.edge_index.T])
        G = nx.Graph()
        G = graph.to_networkx()
        return G

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        node = self.nodes.iloc[idx]
        return (node, self.get_edges(idx))

    def get_edges(self, node_idx):
        return self.edges[
            (self.edges["x_idx"] == node_idx) | (self.edges["y_idx"] == node_idx)
        ]

    def get_edge_index(self):
        return self.edge_index
