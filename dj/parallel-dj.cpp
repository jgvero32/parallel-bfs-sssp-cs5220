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
#include <omp.h>
#include <random>
#include <climits>

// Delta-stepping SSSP
// Buckets: bucket[i] holds nodes with tentative distance in [i*delta, (i+1)*delta)
// Light edges: weight <= delta (processed within a bucket)
// Heavy edges: weight > delta  (processed after bucket is done)

vector<double> parallel_dijktras(const Graph &g, int src, int delta, int nthreads)
{
    vector<double> distances(g.num_nodes, INF);
    vector<bool> in_frontier(g.num_nodes, false);
    vector<int> frontier;

    distances[src] = 0.0;
    in_frontier[src] = true;
    frontier.push_back(src);

    double t_parallel = 0, t_merge = 0;

    while (!frontier.empty())
    {
        // threshold = min_dist(frontier) + delta
        double min_dist = INF;
#pragma omp parallel for num_threads(nthreads) reduction(min : min_dist)
        for (int i = 0; i < (int)frontier.size(); i++)
            min_dist = min(min_dist, distances[frontier[i]]);

        double threshold = min_dist + delta;

        // extract current batch (like closest bucket) and build next frontier
        vector<int> batch;
        vector<int> next_frontier;

        for (int node : frontier)
        {
            if (distances[node] <= threshold)
            {
                batch.push_back(node);
            }
            else
            {
                next_frontier.push_back(node);
            }
        }

        // parallel relaxation ----------------------------------------------------------------------------------------

        // outgoing[src_tid][dest_tid] = updates that belong to thread tid
        vector<vector<vector<pair<int, double>>>> outgoing(nthreads, vector<vector<pair<int, double>>>(nthreads));

        double tp0 = omp_get_wtime();
#pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            for (int i = tid; i < (int)batch.size(); i += nthreads)
            {
                int node = batch[i];
                for (long idx = g.row_ptr[node]; idx < g.row_ptr[node + 1]; idx++)
                {
                    int neighbor = g.col_ind[idx];
                    int weight = g.data[idx];
                    double d_prime = distances[node] + weight;

                    if (d_prime < distances[neighbor])
                        outgoing[tid][neighbor % nthreads].push_back({neighbor, d_prime});
                }
            }
        }
        t_parallel += omp_get_wtime() - tp0;

        // parallel build + frontier rebuild --------------------------------------------------------------------------
        vector<vector<int>> local_next(nthreads);

        double tm0 = omp_get_wtime();
#pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            for (int src_tid = 0; src_tid < nthreads; src_tid++)
            {
                for (auto &[neighbor, d_prime] : outgoing[src_tid][tid])
                {
                    // if better path found, add neighbor to next frontier
                    if (d_prime < distances[neighbor])
                    {
                        distances[neighbor] = d_prime;

                        if (!in_frontier[neighbor])
                        {
                            in_frontier[neighbor] = true;
                            local_next[tid].push_back(neighbor);
                        }
                    }
                }
            }
        }
        t_merge += omp_get_wtime() - tm0;

        // flatten local next into global frontier
        for (int tid = 0; tid < nthreads; tid++)
            next_frontier.insert(next_frontier.end(), local_next[tid].begin(), local_next[tid].end());

        // clear in_frontier for processed nodes
        for (int node : batch)
            in_frontier[node] = false;

        // update frontier
        frontier = move(next_frontier);
    }

    cout << "  [timing] parallel sections : " << t_parallel << "s\n";
    cout << "  [timing] merge (parallel)  : " << t_merge << "s\n";
    cout << "  [timing] parallel fraction : "
         << (t_parallel / (t_parallel + t_merge)) * 100 << "%\n";

    return distances;
}

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        cerr << "Usage: " << argv[0] << " <graph_file> <source_node> <delta> <numthtreads>" << endl;
        return 1;
    }

    const string dataset_file_name = argv[1];
    int source = stoi(argv[2]);
    int delta = stoi(argv[3]); // default delta is 3
    int num_threads = stoi(argv[4]);
    omp_set_num_threads(num_threads);

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

    cout << "Running delta-stepping SSSP from source node: " << source << " (delta=" << delta << ")\n";
    auto t0 = chrono::steady_clock::now();

    vector<double> res = parallel_dijktras(g, source, delta, num_threads);

    auto t1 = chrono::steady_clock::now();
    double elapsed = chrono::duration<double>(t1 - t0).count();

    print_distances(res);

    cout << "\n----- Parallel SSSP Results -----\n";
    cout << "  Source node      : " << source << endl;
    cout << "  Nodes visited    : " << nodes_visited(res) << endl;
    cout << "  Node ID space    : " << g.num_nodes << " (max_node_id + 1)" << endl;
    cout << "  Delta            : " << delta << "\n";
    cout << "  Elapsed time     : " << elapsed << " seconds\n";

    return 0;
}
