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

// Delta-stepping SSSP
// Buckets: bucket[i] holds nodes with tentative distance in [i*delta, (i+1)*delta)
// Light edges: weight <= delta (processed within a bucket)
// Heavy edges: weight > delta  (processed after bucket is done)
vector<double> parallel_dijktras(const Graph &g, int src, int delta)
{
    int n_threads = omp_get_max_threads();
    vector<atomic<double>> distances(g.num_nodes);
    for (auto &d : distances)
        d.store(INF);
    distances[src].store(0.0);

    unordered_map<int, vector<int>> buckets;
    buckets[0].push_back(src);

    double t_parallel = 0, t_merge = 0;

    while (!buckets.empty())
    {
        int b = INT_MAX;
        for (auto &[bk, _] : buckets)
            if (bk < b)
                b = bk;

        vector<int> processed_nodes;
        while (buckets.count(b) && !buckets[b].empty())
        {
            vector<int> snapshot = move(buckets[b]);
            buckets.erase(b);

            // collect all candidate neighbors across threads
            vector<vector<pair<int, double>>> local_updates(n_threads);

            double tp0 = omp_get_wtime();
#pragma omp parallel for schedule(dynamic, 64)
            for (int i = 0; i < (int)snapshot.size(); i++)
            {
                int node = snapshot[i];
                int tid = omp_get_thread_num();
                for (long idx = g.row_ptr[node]; idx < g.row_ptr[node + 1]; idx++)
                {
                    int neighbor = g.col_ind[idx];
                    int weight = g.data[idx];
                    if (weight > delta)
                        continue;
                    double d_prime = distances[node].load() + weight;
                    local_updates[tid].push_back({neighbor, d_prime});
                }
            }
            t_parallel += omp_get_wtime() - tp0;

            // flatten updates
            vector<pair<int, double>> all_updates;
            for (auto &v : local_updates)
                all_updates.insert(all_updates.end(), v.begin(), v.end());

            // parallel CAS — no locks
            double tm0 = omp_get_wtime();
#pragma omp parallel for schedule(dynamic, 64)
            for (int i = 0; i < (int)all_updates.size(); i++)
            {
                auto [neighbor, d_prime] = all_updates[i];
                double old = distances[neighbor].load();
                while (d_prime < old &&
                       !distances[neighbor].compare_exchange_weak(old, d_prime))
                    ;
            }

            // rebuild buckets from updated distances
            // each thread collects its own inserts, then serial merge into buckets
            vector<vector<pair<int, int>>> local_bucket_inserts(n_threads);
#pragma omp parallel for schedule(dynamic, 64)
            for (int i = 0; i < (int)all_updates.size(); i++)
            {
                int neighbor = all_updates[i].first;
                double d = distances[neighbor].load();
                if (d != INF)
                {
                    int tid = omp_get_thread_num();
                    local_bucket_inserts[tid].push_back({(int)d / delta, neighbor});
                }
            }
            for (auto &inserts : local_bucket_inserts)
                for (auto &[bk, node] : inserts)
                    buckets[bk].push_back(node);

            t_merge += omp_get_wtime() - tm0;

            for (int node : snapshot)
                processed_nodes.push_back(node);
        }

        // heavy edges
        vector<vector<pair<int, double>>> local_updates(n_threads);

        double tp0 = omp_get_wtime();
#pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < (int)processed_nodes.size(); i++)
        {
            int node = processed_nodes[i];
            int tid = omp_get_thread_num();
            double nd = distances[node].load();
            if (nd == INF || ((int)nd / delta) != b)
                continue;
            for (long idx = g.row_ptr[node]; idx < g.row_ptr[node + 1]; idx++)
            {
                int neighbor = g.col_ind[idx];
                int weight = g.data[idx];
                if (weight <= delta)
                    continue;
                double d_prime = nd + weight;
                local_updates[tid].push_back({neighbor, d_prime});
            }
        }
        t_parallel += omp_get_wtime() - tp0;

        vector<pair<int, double>> all_updates;
        for (auto &v : local_updates)
            all_updates.insert(all_updates.end(), v.begin(), v.end());

        double tm0 = omp_get_wtime();
#pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < (int)all_updates.size(); i++)
        {
            auto [neighbor, d_prime] = all_updates[i];
            double old = distances[neighbor].load();
            while (d_prime < old &&
                   !distances[neighbor].compare_exchange_weak(old, d_prime))
                ;
        }

        vector<vector<pair<int, int>>> local_bucket_inserts(n_threads);
#pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < (int)all_updates.size(); i++)
        {
            int neighbor = all_updates[i].first;
            double d = distances[neighbor].load();
            if (d != INF)
            {
                int tid = omp_get_thread_num();
                local_bucket_inserts[tid].push_back({(int)d / delta, neighbor});
            }
        }
        for (auto &inserts : local_bucket_inserts)
            for (auto &[bk, node] : inserts)
                buckets[bk].push_back(node);

        t_merge += omp_get_wtime() - tm0;
    }

    cout << "  [timing] parallel sections : " << t_parallel << "s\n";
    cout << "  [timing] merge/CAS         : " << t_merge << "s\n";
    cout << "  [timing] parallel fraction : "
         << (t_parallel / (t_parallel + t_merge)) * 100 << "%\n";

    // convert atomics back to plain doubles for return
    vector<double> result(g.num_nodes);
    for (int i = 0; i < g.num_nodes; i++)
        result[i] = distances[i].load();
    return result;
}

vector<double> parallel_dijktras_v3(const Graph &g, int src, int delta)
{
    vector<double> distances(g.num_nodes, INF);
    unordered_map<int, unordered_set<int>> buckets;
    distances[src] = 0;
    buckets[0].insert(src);
    int n_threads = omp_get_max_threads();

    double t_parallel = 0, t_merge = 0; // accumulators

    while (!buckets.empty())
    {
        int b = buckets.begin()->first;
        for (auto &[new_b, _] : buckets)
            if (new_b < b)
                b = new_b;

        vector<int> processed_nodes;
        while (!buckets[b].empty())
        {
            vector<int> snapshot(buckets[b].begin(), buckets[b].end());
            buckets[b].clear();
            vector<vector<pair<int, double>>> local_updates(n_threads);

            double tp0 = omp_get_wtime();
#pragma omp parallel for
            for (int i = 0; i < (int)snapshot.size(); i++)
            {
                int node = snapshot[i];
                int tid = omp_get_thread_num();
                for (long idx = g.row_ptr[node]; idx < g.row_ptr[node + 1]; idx++)
                {
                    int neighbor = g.col_ind[idx];
                    int weight = g.data[idx];
                    if (weight > delta)
                        continue;
                    double d_prime = distances[node] + weight;
                    if (d_prime < distances[neighbor])
                        local_updates[tid].push_back({neighbor, d_prime});
                }
            }
            t_parallel += omp_get_wtime() - tp0;

            double tm0 = omp_get_wtime();
            for (auto &updates : local_updates)
                for (auto &[neighbor, d_prime] : updates)
                    if (d_prime < distances[neighbor])
                    {
                        if (distances[neighbor] != INF)
                            buckets[(int)distances[neighbor] / delta].erase(neighbor);
                        distances[neighbor] = d_prime;
                        buckets[(int)d_prime / delta].insert(neighbor);
                    }
            t_merge += omp_get_wtime() - tm0;

            for (int node : snapshot)
                processed_nodes.push_back(node);
        }

        buckets.erase(b);

        // heavy edges — same pattern
        vector<vector<pair<int, double>>> local_updates(n_threads);

        double tp0 = omp_get_wtime();
#pragma omp parallel for
        for (int i = 0; i < (int)processed_nodes.size(); i++)
        {
            int node = processed_nodes[i];
            int tid = omp_get_thread_num();
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
                    local_updates[tid].push_back({neighbor, d_prime});
            }
        }
        t_parallel += omp_get_wtime() - tp0;

        double tm0 = omp_get_wtime();
        for (auto &updates : local_updates)
            for (auto &[neighbor, d_prime] : updates)
                if (d_prime < distances[neighbor])
                {
                    if (distances[neighbor] != INF)
                        buckets[(int)distances[neighbor] / delta].erase(neighbor);
                    distances[neighbor] = d_prime;
                    buckets[(int)d_prime / delta].insert(neighbor);
                }
        t_merge += omp_get_wtime() - tm0;
    }

    cout << "  [timing] parallel sections : " << t_parallel << "s\n";
    cout << "  [timing] serial merge      : " << t_merge << "s\n";
    cout << "  [timing] parallel fraction : "
         << (t_parallel / (t_parallel + t_merge)) * 100 << "%\n";

    return distances;
}

vector<double> parallel_dijktras_v2(const Graph &g, int src, int delta)
{
    vector<double> distances(g.num_nodes, INF);
    unordered_map<int, unordered_set<int>> buckets;

    distances[src] = 0;
    buckets[0].insert(src);

    int n_threads = omp_get_max_threads();

    while (!buckets.empty())
    {
        // determine smallest non-empty bucket
        int b = buckets.begin()->first;
        for (auto &[new_b, _] : buckets)
            if (new_b < b)
                b = new_b;

        vector<int> processed_nodes;
        while (!buckets[b].empty())
        {
            vector<int> snapshot(buckets[b].begin(), buckets[b].end());
            buckets[b].clear();

            // per-thread local update buffers (no locks needed in parallel section)
            vector<vector<pair<int, double>>> local_updates(n_threads);

#pragma omp parallel for
            for (int i = 0; i < (int)snapshot.size(); i++)
            {
                int node = snapshot[i];
                int tid = omp_get_thread_num();

                for (long idx = g.row_ptr[node]; idx < g.row_ptr[node + 1]; idx++)
                {
                    int neighbor = g.col_ind[idx];
                    int weight = g.data[idx];
                    if (weight > delta)
                        continue;

                    double d_prime = distances[node] + weight;
                    if (d_prime < distances[neighbor])
                        local_updates[tid].push_back({neighbor, d_prime});
                }
            }

            // serial merge
            for (auto &updates : local_updates)
                for (auto &[neighbor, d_prime] : updates)
                    if (d_prime < distances[neighbor])
                    {
                        if (distances[neighbor] != INF)
                            buckets[(int)distances[neighbor] / delta].erase(neighbor);
                        distances[neighbor] = d_prime;
                        buckets[(int)d_prime / delta].insert(neighbor);
                    }

            for (int node : snapshot)
                processed_nodes.push_back(node);
        }

        buckets.erase(b);

        // relax heavy edges
        vector<vector<pair<int, double>>> local_updates(n_threads);

#pragma omp parallel for
        for (int i = 0; i < (int)processed_nodes.size(); i++)
        {
            int node = processed_nodes[i];
            int tid = omp_get_thread_num();

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
                    local_updates[tid].push_back({neighbor, d_prime});
            }
        }

        // serial merge
        for (auto &updates : local_updates)
            for (auto &[neighbor, d_prime] : updates)
                if (d_prime < distances[neighbor])
                {
                    if (distances[neighbor] != INF)
                        buckets[(int)distances[neighbor] / delta].erase(neighbor);
                    distances[neighbor] = d_prime;
                    buckets[(int)d_prime / delta].insert(neighbor);
                }
    }

    return distances;
}

vector<double> parallel_dijktras_v1(const Graph &g, int src, int delta)
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
        vector<int> processed_nodes;
        while (!buckets[b].empty())
        {
            // take snapshot the current bucket
            vector<int> snapshot(buckets[b].begin(), buckets[b].end());
            buckets[b].clear();

// for each node in the bucket snapshot
#pragma omp parallel for
            for (int node : snapshot)
            {
#pragma omp critical
                processed_nodes.push_back(node);
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
#pragma omp critical
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
#pragma omp critical
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

    return 0;
}
