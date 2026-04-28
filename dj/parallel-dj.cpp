/*
In progress...

Parallel SSSP for roadNet-CA
How to run:
   Compile: g++ -O2 -std=c++23 -o dj_parallel parallel-dj.cpp
   Run:     ./dj_parallel ../datasets/roadNet-CA.txt <source_node>
       ex:  ./dj_parallel ../datasets/roadNet-CA.txt 0
*/

#include "graph_utils.h"
#include <unordered_set>
#include <chrono>
#include <unordered_map>
#include <mpi.h>

// Delta-stepping SSSP
// Buckets: bucket[i] holds nodes with tentative distance in [i*delta, (i+1)*delta)
// Light edges: weight <= delta (processed within a bucket)
// Heavy edges: weight > delta  (processed after bucket is done)
vector<double> parallel_dijktras(const Graph &g, int src, int delta)
{
    vector<double> distances(g.num_nodes, INF);
    unordered_map<int, unordered_set<int>> buckets;

    distances[src] = 0;
    buckets[0].insert(src);

    while (!buckets.empty())
    {
        int b = buckets.begin()->first;

        // determine smallest non-empty bucket
        for (auto &[new_b, _] : buckets)
        {
            if (new_b < b)
                b = new_b;
        }
        // cout << "processing bucket " << b << endl;

        // process
        unordered_set<int> processed_nodes;
        while (!buckets[b].empty())
        {
            // take snapshot the current bucket
            vector<int> snapshot(buckets[b].begin(), buckets[b].end());
            buckets[b].clear();

// for each node in the bucket snapshot
#pragma omp parallel for
            for (int node : snapshot)
            {
                processed_nodes.insert(node);
                // cout << "node " << node << "'s neighbors -----------------\n";

                // go through node's neighbors and relax edges
                for (long index_for_col_ind = g.row_ptr[node]; index_for_col_ind < g.row_ptr[node + 1]; index_for_col_ind++)
                {
                    int neighbor = g.col_ind[index_for_col_ind];
                    int weight = g.data[index_for_col_ind];

                    // cout << "light edge with " << neighbor << " ? ";

                    // skip heavy edges
                    if (weight > delta)
                        continue;

                    // cout << "yes\n";

                    double d_prime = distances[node] + weight;
                    if (d_prime < distances[neighbor])
                    {
                        // remove from old bucket if neighbor has been seen before
                        if (distances[neighbor] != INF)
                        {
                            buckets[(int)distances[neighbor] / delta].erase(neighbor);
                        }
                        distances[neighbor] = d_prime;
                        buckets[(int)d_prime / delta].insert(neighbor);
                    }
                }
            }
        }

        // remove empty bucket
        buckets.erase(b);

// cout << "relaxing heavy edges" << endl;

// relax heavy edges of nodes from bucket (weight > delta)
#pragma omp parallel for
        for (int node : processed_nodes)
        {
            if (distances[node] == INF || ((int)distances[node] / delta) != b)
                continue;

            for (long index_for_col_ind = g.row_ptr[node]; index_for_col_ind < g.row_ptr[node + 1]; index_for_col_ind++)
            {
                int neighbor = g.col_ind[index_for_col_ind];
                int weight = g.data[index_for_col_ind];

                // cout << "heavy edge with " << neighbor << " ? ";

                // skip lights edges
                if (weight <= delta)
                    continue;

                // cout << "yes\n";

                double d_prime = distances[node] + weight;
                if (d_prime < distances[neighbor])
                {
                    // remove from old bucket if neighbor has been seen before
                    if (distances[neighbor] != INF)
                    {
                        buckets[(int)distances[neighbor] / delta].erase(neighbor);
                    }
                    distances[neighbor] = d_prime;
                    buckets[(int)d_prime / delta].insert(neighbor);
                }
            }
        }
    }

    return distances;
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " <graph_file> <source_node> [delta]" << endl;
        return 1;
    }

    const string dataset_file_name = argv[1];
    int source = stoi(argv[2]);
    int delta = (argc >= 4) ? stoi(argv[3]) : 3; // default delta is 3

    // load da graph
    cout << "Loading graph from: " << dataset_file_name << endl;
    Graph g;
    try
    {
        g = load_graph(dataset_file_name);
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

    // int num_procs, rank;
    // MPI_Init(&argc, &argv);
    // MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    cout << "Running delta-stepping SSSP from source node: " << source << " (delta=" << delta << ")\n";
    auto t0 = chrono::steady_clock::now();

    vector<double> res = parallel_dijktras(g, source, delta);

    auto t1 = chrono::steady_clock::now();
    double elapsed = chrono::duration<double>(t1 - t0).count();

    print_distances(res);

    cout << "\n----- Parallel SSSP Results -----\n";
    cout << "  Source node      : " << source << endl;
    cout << "  Nodes visited    : " << nodes_visited(res) << endl;
    cout << "  Node ID space    : " << g.num_nodes << " (max_node_id + 1)" << endl;
    cout << "  Delta            : " << delta << "\n";
    cout << "  Elapsed time     : " << elapsed << " seconds\n";

    // MPI_Finalize();

    return 0;
}

int main(int argc, char *argv[])
{
    int num_procs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Finalize();

    return 0;
}