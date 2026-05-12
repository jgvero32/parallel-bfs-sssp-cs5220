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

    // each thread owns nodes where node % nthreads == tid
    vector<unordered_map<int, unordered_set<int>>> tbuckets(nthreads);

    int src_owner = src % nthreads;
    distances[src] = 0.0;
    tbuckets[src_owner][0].insert(src); // bucket 0 = distance range [0, delta)

    double t_parallel = 0, t_merge = 0;

    while (true)
    {
        // find global min non-empty bucket across all threads
        int b = INT_MAX;
        for (int tid = 0; tid < nthreads; tid++)
            for (auto &[bucket, nodes] : tbuckets[tid])
                if (!nodes.empty() && bucket < b)
                    b = bucket;

        // no more buckets to process
        if (b == INT_MAX)
            break;

        vector<int> processed_nodes;

        // check if any thread has work remaining in bucket b
        auto work_available = [&]()
        {
            for (int tid = 0; tid < nthreads; tid++)
                if (tbuckets[tid].count(b) && !tbuckets[tid][b].empty())
                    return true;
            return false;
        };

        // process light edges until bucket b is stable ----------------------------
        while (work_available())
        {
            // each thread snapshots its own slice of bucket b
            vector<unordered_set<int>> snapshots(nthreads);
            vector<vector<int>> local_processed(nthreads);

            for (int tid = 0; tid < nthreads; tid++)
            {
                if (tbuckets[tid].count(b))
                {
                    snapshots[tid] = move(tbuckets[tid][b]);
                    tbuckets[tid].erase(b);
                }
            }

            // outgoing[src_tid][dest_tid] — thread tid writes only to outgoing[tid]
            vector<vector<vector<pair<int, double>>>> outgoing(nthreads, vector<vector<pair<int, double>>>(nthreads));

            double tp0 = omp_get_wtime();
#pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                for (int node : snapshots[tid])
                    for (long idx = g.row_ptr[node]; idx < g.row_ptr[node + 1]; idx++)
                    {
                        int neighbor = g.col_ind[idx];
                        int weight = g.data[idx];
                        if (weight > delta)
                            continue;
                        double d_prime = distances[node] + weight;
                        if (d_prime < distances[neighbor])
                            outgoing[tid][neighbor % nthreads].push_back({neighbor, d_prime});
                    }
            }
            t_parallel += omp_get_wtime() - tp0;

            // each thread merges only its own incoming updates — no conflicts
            double tm0 = omp_get_wtime();
#pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();

                // collect best update per neighbor across all src threads
                unordered_map<int, double> best_updates;
                for (int src_tid = 0; src_tid < nthreads; src_tid++)
                    for (auto &[neighbor, d_prime] : outgoing[src_tid][tid])
                        if (!best_updates.count(neighbor) || d_prime < best_updates[neighbor])
                            best_updates[neighbor] = d_prime;

                // apply best updates — thread tid owns these neighbors exclusively
                for (auto &[neighbor, d_prime] : best_updates)
                    if (d_prime < distances[neighbor])
                    {
                        if (distances[neighbor] != INF)
                            tbuckets[tid][(int)distances[neighbor] / delta].erase(neighbor);
                        distances[neighbor] = d_prime;
                        tbuckets[tid][(int)d_prime / delta].insert(neighbor);
                    }

                for (int node : snapshots[tid])
                    local_processed[tid].push_back(node);
            }
            t_merge += omp_get_wtime() - tm0;

            for (int tid = 0; tid < nthreads; tid++)
                for (int node : local_processed[tid])
                    processed_nodes.push_back(node);
        }

        // bucket b is now stable — clear it from all threads
        for (int tid = 0; tid < nthreads; tid++)
            tbuckets[tid].erase(b);

        // relax heavy edges from processed nodes (weight > delta) -----------------
        vector<vector<vector<pair<int, double>>>> outgoing(nthreads, vector<vector<pair<int, double>>>(nthreads));

        double tp0 = omp_get_wtime();
#pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            for (int node : processed_nodes)
            {
                if (node % nthreads != tid)
                    continue;
                if (distances[node] == INF || ((int)distances[node] / delta) != b)
                    continue;
                for (long idx = g.row_ptr[node]; idx < g.row_ptr[node + 1]; idx++)
                {
                    int neighbor = g.col_ind[idx];
                    int weight = g.data[idx];
                    if (weight <= delta)
                        continue;
                    double d_prime = distances[node] + weight;
                    if (d_prime < distances[neighbor])
                        outgoing[tid][neighbor % nthreads].push_back({neighbor, d_prime});
                }
            }
        }
        t_parallel += omp_get_wtime() - tp0;

        double tm0 = omp_get_wtime();
#pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();

            // collect best update per neighbor across all src threads
            unordered_map<int, double> best_updates;
            for (int src_tid = 0; src_tid < nthreads; src_tid++)
                for (auto &[neighbor, d_prime] : outgoing[src_tid][tid])
                    if (!best_updates.count(neighbor) || d_prime < best_updates[neighbor])
                        best_updates[neighbor] = d_prime;

            // apply best updates — thread tid owns these neighbors exclusively
            for (auto &[neighbor, d_prime] : best_updates)
                if (d_prime < distances[neighbor])
                {
                    if (distances[neighbor] != INF)
                        tbuckets[tid][(int)distances[neighbor] / delta].erase(neighbor);
                    distances[neighbor] = d_prime;
                    tbuckets[tid][(int)d_prime / delta].insert(neighbor);
                }
        }
        t_merge += omp_get_wtime() - tm0;
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
