/*
Serial DJ for roadNet-CA
How to run:
   Compile: g++ -O2 -std=c++23 -o dj_serial serial-dj.cpp
   Run:     ./dj_serial ../datasets/roadNet-CA.txt <source_node>
       ex:  ./dj_serial ../datasets/roadNet-CA.txt 0
*/

#include "graph_utils.h"
#include <limits>
#include <random>

// SSSP for single connected component
vector<double> dijkstra(const Graph &g, int src)
{
    vector<double> distances(g.num_nodes, INF);
    priority_queue<pair<double, int>, vector<pair<double, int>>, greater<>> pqueue;

    distances[src] = 0;
    pqueue.push({0, src});

    while (!pqueue.empty())
    {
        auto [dist, node] = pqueue.top();
        pqueue.pop();

        if (dist > distances[node])
            continue;

        // relax edges for all neighbors
        for (long index_for_col_ind = g.row_ptr[node]; index_for_col_ind < g.row_ptr[node + 1]; index_for_col_ind++)
        {
            int neighbor = g.col_ind[index_for_col_ind];
            int weight = g.data[index_for_col_ind];

            double d_prime = distances[node] + weight;
            if (d_prime < distances[neighbor])
            {
                distances[neighbor] = d_prime;
                pqueue.push({d_prime, neighbor});
            }
        }
    }

    return distances;
}

vector<double> parallel_dijkstra(const Graph &g, int src)
{
    // TODO: write parallel version
    vector<double> distances(g.num_nodes, INF);
    return distances;
}

int main(int argc, char *argv[])
{ // argv looks like {./dj_serial ../datasets/roadNet-CA.txt, 0}
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " <graph_file> <source_node>" << endl;
        return 1;
    }

    const string dataset_file_name = argv[1];
    int source = stoi(argv[2]);

    // load da graph
    cout << "Loading graph from: " << dataset_file_name << endl;
    Graph g;
    try
    {
        g = load_graph(dataset_file_name);
        mt19937 rng(42);
        uniform_int_distribution<int> weight_dist(1, 100);
        for (int &w : g.data)
            w = weight_dist(rng);
    }
    catch (const exception &e)
    {
        cerr << "Error loading the graph??" << endl;
        return 1;
    }

    if (source < 0 || source >= g.num_nodes)
    {
        cerr << "Source node " << source << " out of range [0, " << g.num_nodes << ")\n";
        return 1;
    }

    cout << "Running serial dijkstra from source node: " << source << endl;
    auto t0 = chrono::steady_clock::now();

    vector<double> res = dijkstra(g, source);

    auto t1 = chrono::steady_clock::now();
    double elapsed = chrono::duration<double>(t1 - t0).count();

    print_distances(res);

    cout << "\n----- Serial dijkstra Results -----\n";
    cout << "  Source node      : " << source << endl;
    cout << "  Nodes visited    : " << nodes_visited(res) << endl;
    cout << "  Node ID space    : " << g.num_nodes << " (max_node_id + 1)" << endl;
    cout << "  Elapsed time     : " << elapsed << " seconds\n";

    return 0;
}
