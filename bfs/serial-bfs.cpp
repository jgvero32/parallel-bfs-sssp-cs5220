/*
Serial BFS for roadNet-CA
How to run:
    cd bfs
    Compile: g++ -O2 -std=c++23 -o bfs_serial serial-bfs.cpp
    Run:     ./bfs_serial ../datasets/roadNet-CA.txt <source_node>
        ex:  ./bfs_serial ../datasets/roadNet-CA.txt 0
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <string>
#include <chrono>
#include <stdexcept>


struct Graph {
    int num_nodes;
    int num_edges;
    std::vector<int> col_ind; // each element is a column ID (to)
    std::vector<int> row_ptr; // elements are [start:stop] in col_ind

    Graph() {
        num_nodes = 0;
        num_edges = 0;
    }
};


// This function reads the file and stores every node edge into an adjacency list
// The nodeIds in the dataset files are not contiguous, so the adjacency list size needs to be of max_num_nodes (max nodeID + 1)
// row_ptr is [start:stop] in the col_ind array
// col_ind contains all the columnIDs, which are just the "to" nodeIds
    //   0 1 2 3 4
    // 0 1   1
    // 1     1
    // 2
    // 3       1 1
    // 4   
    
    // row_ptr = {0, 2, 3, 5}
    // column_indices {0, 2, 2, 3, 4}
Graph load_graph(const std::string &filename) {
    std::ifstream file(filename); // this opens the file
    if (!file.is_open())
        throw std::runtime_error("Error opening file: " + filename);
 
    std::string line;
    std::vector<std::vector<int>> adj;
    int max_node_id = -1;

    while (std::getline(file, line)) { // this reads a line of the file
        if (line.empty() || line[0] == '#') { // ignore the first 4 lines
            continue;
        }

        std::istringstream ss(line); // this turns file text into ints
        int from, to;
        if (!(ss >> from >> to)) { // grab the from and to node numbers
            continue;
        }

        max_node_id = std::max(max_node_id, std::max(from, to));
        if (max_node_id + 1 > adj.size()) { // IDs in the dataset are sparse, so we need an adj size of up to max_node_id + 1
            adj.resize(max_node_id + 1);
        }

        adj[from].push_back(to);
    }
 
    Graph g;
    g.num_nodes = adj.size();
    g.row_ptr.resize(adj.size() + 1, 0); // row_ptr vector is the # nodes + 1
 
    for (int i = 0; i < adj.size(); i++) {
        g.row_ptr[i + 1] = g.row_ptr[i] + adj[i].size(); // row_ptr is [start:stop] -> we want to do stop = start + node's # edges (aka number of elements in a row)
    }

    g.num_edges = g.row_ptr[adj.size()];
    g.col_ind.resize(g.num_edges); // col_ind is the number of edges (# elements in all rows)

    int i = 0;
    for (int from = 0; from < adj.size(); from++) {
        for (int to : adj[from]) {
            g.col_ind[i++] = to; // col_ind consists of column indices which are the "to" indices 
        }
    }
 
    return g;
}

// BFS for single connected component
std::vector<int> bfs(const Graph &g, int src) {
    std::vector<bool> visited(g.num_nodes, false);
    std::vector<int> res;
    std::queue<int> queue;

    visited[src] = true;
    queue.push(src);

    while (!queue.empty()) {
        int curr = queue.front();
        queue.pop();
        res.push_back(curr);

        // visit all the unvisited neighbours of curr node
        for (long index_for_col_ind = g.row_ptr[curr]; index_for_col_ind < g.row_ptr[curr + 1]; index_for_col_ind++) {
            int neighbor = g.col_ind[index_for_col_ind];
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                queue.push(neighbor);
            }
        }
    }
    
    return res;
}

std::vector<int> parallel_bfs(const Graph &g, int src) {
    // TODO: write parallel version
    std::vector<int> res;
    return res;
}

int main(int argc, char *argv[]) { // argv looks like {./bfs_serial, ../datasets/roadNet-CA.txt, 0}
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <graph_file> <source_node>" << std::endl;
        return 1;
    }

    const std::string dataset_file_name = argv[1];
    int source = std::stoi(argv[2]);

    // load da graph
    std::cout << "Loading graph from: " << dataset_file_name << std::endl;
    Graph g;
    try {
        g = load_graph(dataset_file_name);
    } catch (const std::exception &e) {
        std::cerr << "Error loading the graph??" << std::endl;
        return 1;
    }

    if (source < 0 || source >= g.num_nodes) {
        std::cerr << "Source node " << source << " out of range [0, " << g.num_nodes << ")\n";
        return 1;
    }

    std::cout << "Running serial BFS from source node: " << source << std::endl;
    auto t0 = std::chrono::steady_clock::now();

    std::vector<int> res = bfs(g, source);

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();


    std::cout << "\n----- Serial BFS Results -----\n";
    std::cout << "  Source node      : " << source        << std::endl;
    std::cout << "  Nodes visited    : " << res.size() << std::endl;
    std::cout << "  Node ID space    : " << g.num_nodes   << " (max_node_id + 1)" << std::endl;
    std::cout << "  Elapsed time     : " << elapsed << " seconds\n";

    return 0;
}