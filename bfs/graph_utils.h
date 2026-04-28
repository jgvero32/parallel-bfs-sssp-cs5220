#include <chrono>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <vector>
#include <sstream>

struct Graph {
    uint32_t num_nodes;
    uint32_t num_edges;
    // CSR storage format
    std::vector<int> col_ind; // each element is a column ID (to)
    std::vector<int> row_ptr; // elements are [start:stop] in col_ind

    Graph() {
        num_nodes = 0;
        num_edges = 0;
    }
};

/**
 * This function reads the file and stores every node edge into an adjacency
 * list The nodeIds in the dataset files are not contiguous, so the adjacency
 * list size needs to be of max_num_nodes (max nodeID + 1) row_ptr is
 * [start:stop] in the col_ind array col_ind contains all the columnIDs, which
 * are just the "to" nodeIds
 *
 * 0 1 2 3 4
 * 0 1   1
 * 1     1
 * 2
 * 3     1 1
 * 4
 *
 * row_ptr = {0, 2, 3, 5}
 * column_indices {0, 2, 2, 3, 4}
 *
 * transpose: transposing the matrix A is necessary for parallel BFS, because
 * the new frontier f_{i+1} is produced by A^T*f_{i}
 */
Graph load_graph(const std::string &filename, bool transpose) {
    std::ifstream file(filename); // this opens the file
    if (!file.is_open())
        throw std::runtime_error("Error opening file: " + filename);

    std::string line;
    std::vector<std::vector<int>> adj;
    int max_node_id = -1;

    while (std::getline(file, line)) {        // this reads a line of the file
        if (line.empty() || line[0] == '#') { // ignore the first 4 lines
            continue;
        }

        std::istringstream ss(line); // this turns file text into ints
        int from, to;
        if (!(ss >> from >> to)) { // grab the from and to node numbers
            continue;
        }

        max_node_id = std::max(max_node_id, std::max(from, to));
        if (max_node_id + 1 >
            adj.size()) { // IDs in the dataset are sparse, so we need an adj
                          // size of up to max_node_id + 1
            adj.resize(max_node_id + 1);
        }

        if (transpose) {
            adj[to].push_back(from);
        } else {
            adj[from].push_back(to);
        }
    }

    Graph g;
    g.num_nodes = adj.size();
    g.row_ptr.resize(adj.size() + 1, 0); // row_ptr vector is the # nodes + 1

    for (int i = 0; i < adj.size(); i++) {
        g.row_ptr[i + 1] =
            g.row_ptr[i] + adj[i].size(); // row_ptr is [start:stop] -> we want
                                          // to do stop = start + node's # edges
                                          // (aka number of elements in a row)
    }

    g.num_edges = g.row_ptr[adj.size()];
    g.col_ind.resize(
        g.num_edges); // col_ind is the number of edges (# elements in all rows)

    int i = 0;
    for (int from = 0; from < adj.size(); from++) {
        for (int to : adj[from]) {
            g.col_ind[i++] = to; // col_ind consists of column indices which are
                                 // the "to" indices
        }
    }

    return g;
}

/**
 * The function prints the distances of every node from the source node.
 * nodes_to_print is used when the graph is too large to print every value.
 */
void print_distances(const std::vector<int> dists, size_t nodes_to_print = 30) {
    int end = std::min(nodes_to_print, dists.size());

    std::cout << "[";
    for (int i = 0; i < end - 1; ++i) {
        std::cout << dists[i] << ", ";
    }
    std::cout << dists[end - 1] << "]\n";
}

int nodes_visited(const std::vector<int> dists) {
    int visited_ct = 0;
    for (int i = 0; i < dists.size(); ++i) {
        if (dists[i] >= 0) {
            visited_ct++;
        }
    }
    return visited_ct;
}
