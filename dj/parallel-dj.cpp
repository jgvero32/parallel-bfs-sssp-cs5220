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
vector<double> parallel_dijktras_v6(const Graph &g, int src, int delta, int nthreads)
{
    vector<double> distances(g.num_nodes, INF);
    vector<bool> in_frontier(g.num_nodes, false);

    // sparse frontier — explicit list of active nodes
    vector<int> frontier;

    distances[src] = 0.0;
    in_frontier[src] = true;
    frontier.push_back(src);

    // sparse/dense threshold — switch to full scan when frontier > n / log2(n)
    int dense_threshold = g.num_nodes / max(1.0, log2((double)g.num_nodes));

    double t_parallel = 0, t_merge = 0, t_extract = 0, t_flatten = 0;

    // outgoing[src_tid][dest_tid] — pre-allocated, reused each iteration
    vector<vector<vector<pair<int, double>>>> outgoing(nthreads, vector<vector<pair<int, double>>>(nthreads));
    vector<vector<int>> local_next(nthreads);

    while (!frontier.empty())
    {
        // find min distance in frontier — parallel reduce
        double min_dist = INF;
#pragma omp parallel for num_threads(nthreads) reduction(min : min_dist)
        for (int i = 0; i < (int)frontier.size(); i++)
            min_dist = min(min_dist, distances[frontier[i]]);

        double threshold = min_dist + delta;

        // extract batch — sparse or dense depending on frontier size
        double _te = omp_get_wtime();
        vector<int> batch;
        vector<int> next_frontier;

        if ((int)frontier.size() < dense_threshold)
        {
            // sparse mode — filter explicit frontier list in parallel
            vector<vector<int>> local_batch(nthreads), local_next_frontier(nthreads);

#pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                for (int i = tid; i < (int)frontier.size(); i += nthreads)
                {
                    int node = frontier[i];
                    if (distances[node] <= threshold)
                        local_batch[tid].push_back(node);
                    else
                        local_next_frontier[tid].push_back(node);
                }
            }

            // prefix sum scatter into batch and next_frontier
            vector<int> batch_offsets(nthreads + 1, 0), nf_offsets(nthreads + 1, 0);
            for (int tid = 0; tid < nthreads; tid++)
            {
                batch_offsets[tid + 1] = batch_offsets[tid] + local_batch[tid].size();
                nf_offsets[tid + 1] = nf_offsets[tid] + local_next_frontier[tid].size();
            }
            batch.resize(batch_offsets[nthreads]);
            next_frontier.resize(nf_offsets[nthreads]);

#pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                copy(local_batch[tid].begin(), local_batch[tid].end(),
                     batch.begin() + batch_offsets[tid]);
                copy(local_next_frontier[tid].begin(), local_next_frontier[tid].end(),
                     next_frontier.begin() + nf_offsets[tid]);
            }
        }
        else
        {
            // dense mode — scan full distance array in parallel
            vector<vector<int>> local_batch(nthreads), local_next_frontier(nthreads);

#pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                for (int i = tid; i < g.num_nodes; i += nthreads)
                {
                    if (distances[i] == INF)
                        continue;
                    if (distances[i] <= threshold)
                        local_batch[tid].push_back(i);
                    else if (in_frontier[i])
                        local_next_frontier[tid].push_back(i);
                }
            }

            // prefix sum scatter
            vector<int> batch_offsets(nthreads + 1, 0), nf_offsets(nthreads + 1, 0);
            for (int tid = 0; tid < nthreads; tid++)
            {
                batch_offsets[tid + 1] = batch_offsets[tid] + local_batch[tid].size();
                nf_offsets[tid + 1] = nf_offsets[tid] + local_next_frontier[tid].size();
            }
            batch.resize(batch_offsets[nthreads]);
            next_frontier.resize(nf_offsets[nthreads]);

#pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                copy(local_batch[tid].begin(), local_batch[tid].end(),
                     batch.begin() + batch_offsets[tid]);
                copy(local_next_frontier[tid].begin(), local_next_frontier[tid].end(),
                     next_frontier.begin() + nf_offsets[tid]);
            }
        }
        t_extract += omp_get_wtime() - _te;

        // parallel relaxation — no light/heavy split, Δ*-stepping processes all edges
        for (auto &v : outgoing)
            for (auto &vv : v)
                vv.clear();

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

        // parallel merge — each thread owns its neighbors exclusively
        for (auto &v : local_next)
            v.clear();

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

            // apply and add newly discovered nodes to next frontier
            for (auto &[neighbor, d_prime] : best_updates)
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
        t_merge += omp_get_wtime() - tm0;

        // flatten local_next into next_frontier via prefix sum scatter
        double _tf = omp_get_wtime();
        vector<int> next_offsets(nthreads + 1, 0);
        for (int tid = 0; tid < nthreads; tid++)
            next_offsets[tid + 1] = next_offsets[tid] + local_next[tid].size();

        int old_size = next_frontier.size();
        next_frontier.resize(old_size + next_offsets[nthreads]);

#pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            copy(local_next[tid].begin(), local_next[tid].end(),
                 next_frontier.begin() + old_size + next_offsets[tid]);
        }
        t_flatten += omp_get_wtime() - _tf;

// clear in_frontier for processed batch nodes — Δ* accepts re-processing
#pragma omp parallel for num_threads(nthreads)
        for (int i = 0; i < (int)batch.size(); i++)
            in_frontier[batch[i]] = false;

        frontier = move(next_frontier);
    }

    cout << "  [timing] parallel relaxation : " << t_parallel << "s\n";
    cout << "  [timing] parallel merge      : " << t_merge << "s\n";
    cout << "  [timing] extract (sparse/dense): " << t_extract << "s\n";
    cout << "  [timing] flatten             : " << t_flatten << "s\n";

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
