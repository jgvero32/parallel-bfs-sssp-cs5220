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
#include <set>

// Delta-stepping SSSP
// Buckets: bucket[i] holds nodes with tentative distance in [i*delta, (i+1)*delta)
// Light edges: weight <= delta (processed within a bucket)
// Heavy edges: weight > delta  (processed after bucket is done)
vector<double> parallel_dijktras(const Graph &g, int src, int delta, int nthreads)
{
    vector<double> distances(g.num_nodes, INF);

    // flat array tracking each node's current bucket — O(1) stale check
    vector<int> node_bucket(g.num_nodes, -1);

    // max bucket based on weights 1-100 and graph diameter
    int max_bucket = 200000 / delta + 2;

    // per-thread flat bucket lists — direct index by bucket id, no hash map
    vector<vector<vector<int>>> bucket_lists(nthreads, vector<vector<int>>(max_bucket));

    // build thread-local CSR — each thread owns nodes where node % nthreads == tid
    // allocated in parallel so memory lands on each thread's local NUMA node
    vector<vector<int>> local_col_ind(nthreads);
    vector<vector<int>> local_data(nthreads);
    vector<vector<long>> local_row_ptr(nthreads);

    double t_csr_build = 0;
    double _tcsr = omp_get_wtime();
#pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        for (int node = tid; node < g.num_nodes; node += nthreads)
        {
            local_row_ptr[tid].push_back(local_col_ind[tid].size());
            for (long idx = g.row_ptr[node]; idx < g.row_ptr[node + 1]; idx++)
            {
                local_col_ind[tid].push_back(g.col_ind[idx]);
                local_data[tid].push_back(g.data[idx]);
            }
        }
        local_row_ptr[tid].push_back(local_col_ind[tid].size()); // sentinel
    }
    t_csr_build = omp_get_wtime() - _tcsr;

    vector<vector<vector<pair<int, double>>>> outgoing(nthreads, vector<vector<pair<int, double>>>(nthreads));
    vector<vector<int>> snapshots(nthreads);
    vector<vector<int>> local_processed(nthreads);

    int src_owner = src % nthreads;
    distances[src] = 0.0;
    node_bucket[src] = 0;
    bucket_lists[src_owner][0].push_back(src); // bucket 0 = distance range [0, delta)

    double t_parallel = 0, t_merge = 0;
    double t_find_b = 0, t_work_avail = 0, t_snapshot = 0;
    double t_outgoing_clear = 0, t_flatten = 0, t_bucket_erase = 0;

    while (true)
    {
        // find global min non-empty bucket — each thread scans its own bucket_lists
        // sequential scan of flat array breaks on first non-empty — cache friendly
        double _t0 = omp_get_wtime();
        int b = INT_MAX;
#pragma omp parallel for num_threads(nthreads) reduction(min : b)
        for (int tid = 0; tid < nthreads; tid++)
            for (int bk = 0; bk < max_bucket; bk++)
                if (!bucket_lists[tid][bk].empty())
                {
                    b = min(b, bk);
                    break; // first non-empty is min for this thread
                }
        t_find_b += omp_get_wtime() - _t0;

        if (b == INT_MAX)
            break;

        vector<int> processed_nodes;
        bool any_work = true;

        // process light edges until bucket b is stable ----------------------------
        while (any_work)
        {
            // parallel snapshot — filter stale entries, clear bucket
            // no thread_min_bucket recompute — find_b handles that next outer iteration
            double _ts = omp_get_wtime();
#pragma omp parallel for num_threads(nthreads)
            for (int tid = 0; tid < nthreads; tid++)
            {
                snapshots[tid].clear();
                local_processed[tid].clear();
                for (int node : bucket_lists[tid][b])
                    if (node_bucket[node] == b)
                        snapshots[tid].push_back(node);
                bucket_lists[tid][b].clear(); // clear all — stale and fresh consumed
            }
            t_snapshot += omp_get_wtime() - _ts;

            // check if any snapshot had work — replaces serial work_available scan
            double _tw = omp_get_wtime();
            any_work = false;
            for (int tid = 0; tid < nthreads; tid++)
                if (!snapshots[tid].empty())
                {
                    any_work = true;
                    break;
                }
            t_work_avail += omp_get_wtime() - _tw;

            if (!any_work)
                break;

            // outgoing[src_tid][dest_tid] — parallel clear, thread tid clears its row
            double _tc = omp_get_wtime();
#pragma omp parallel for num_threads(nthreads)
            for (int tid = 0; tid < nthreads; tid++)
                for (auto &vv : outgoing[tid])
                    vv.clear();
            t_outgoing_clear += omp_get_wtime() - _tc;

            // parallel relaxation using thread-local CSR — NUMA-friendly reads
            double tp0 = omp_get_wtime();
#pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                for (int node : snapshots[tid])
                {
                    int local_node = node / nthreads;
                    long start = local_row_ptr[tid][local_node];
                    long end = local_row_ptr[tid][local_node + 1];
                    for (long idx = start; idx < end; idx++)
                    {
                        int neighbor = local_col_ind[tid][idx];
                        int weight = local_data[tid][idx];
                        if (weight > delta)
                            continue;
                        double d_prime = distances[node] + weight;
                        if (d_prime < distances[neighbor])
                            outgoing[tid][neighbor % nthreads].push_back({neighbor, d_prime});
                    }
                }
            }
            t_parallel += omp_get_wtime() - tp0;

            // parallel merge — tid owns all neighbors in outgoing[*][tid] exclusively
            // no unordered_map — direct apply with distance check guards correctness
            double tm0 = omp_get_wtime();
#pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                for (int src_tid = 0; src_tid < nthreads; src_tid++)
                    for (auto &[neighbor, d_prime] : outgoing[src_tid][tid])
                        if (d_prime < distances[neighbor])
                        {
                            distances[neighbor] = d_prime;
                            int new_b = (int)d_prime / delta;
                            node_bucket[neighbor] = new_b;
                            bucket_lists[tid][new_b].push_back(neighbor);
                        }

                for (int node : snapshots[tid])
                    local_processed[tid].push_back(node);
            }
            t_merge += omp_get_wtime() - tm0;

            // parallel flatten via prefix sum scatter — no serial bottleneck
            double _tf = omp_get_wtime();
            vector<int> offsets(nthreads + 1, 0);
            for (int tid = 0; tid < nthreads; tid++)
                offsets[tid + 1] = offsets[tid] + local_processed[tid].size();
            int old_size = processed_nodes.size();
            processed_nodes.resize(old_size + offsets[nthreads]);
#pragma omp parallel for num_threads(nthreads)
            for (int tid = 0; tid < nthreads; tid++)
                copy(local_processed[tid].begin(), local_processed[tid].end(),
                     processed_nodes.begin() + old_size + offsets[tid]);
            t_flatten += omp_get_wtime() - _tf;
        }

        // bucket b fully processed — clear from all threads
        double _te = omp_get_wtime();
#pragma omp parallel for num_threads(nthreads)
        for (int tid = 0; tid < nthreads; tid++)
            bucket_lists[tid][b].clear();
        t_bucket_erase += omp_get_wtime() - _te;

        // relax heavy edges from processed nodes (weight > delta) -----------------
        double _tc = omp_get_wtime();
#pragma omp parallel for num_threads(nthreads)
        for (int tid = 0; tid < nthreads; tid++)
            for (auto &vv : outgoing[tid])
                vv.clear();
        t_outgoing_clear += omp_get_wtime() - _tc;

        double tp0 = omp_get_wtime();
#pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            for (int node : processed_nodes)
            {
                if (node % nthreads != tid)
                    continue;
                if (distances[node] == INF || node_bucket[node] != b)
                    continue;
                int local_node = node / nthreads;
                long start = local_row_ptr[tid][local_node];
                long end = local_row_ptr[tid][local_node + 1];
                for (long idx = start; idx < end; idx++)
                {
                    int neighbor = local_col_ind[tid][idx];
                    int weight = local_data[tid][idx];
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
            for (int src_tid = 0; src_tid < nthreads; src_tid++)
                for (auto &[neighbor, d_prime] : outgoing[src_tid][tid])
                    if (d_prime < distances[neighbor])
                    {
                        distances[neighbor] = d_prime;
                        int new_b = (int)d_prime / delta;
                        node_bucket[neighbor] = new_b;
                        bucket_lists[tid][new_b].push_back(neighbor);
                    }
        }
        t_merge += omp_get_wtime() - tm0;
    }

    cout << "  [timing] local CSR build     : " << t_csr_build << "s\n";
    cout << "  [timing] parallel relaxation : " << t_parallel << "s\n";
    cout << "  [timing] parallel merge      : " << t_merge << "s\n";
    cout << "  [timing] find min bucket     : " << t_find_b << "s\n";
    cout << "  [timing] work_available      : " << t_work_avail << "s\n";
    cout << "  [timing] snapshot            : " << t_snapshot << "s\n";
    cout << "  [timing] outgoing clear      : " << t_outgoing_clear << "s\n";
    cout << "  [timing] flatten processed   : " << t_flatten << "s\n";
    cout << "  [timing] bucket erase        : " << t_bucket_erase << "s\n";
    cout << "  [timing] parallel fraction   : "
         << (t_parallel / (t_parallel + t_merge)) * 100 << "%\n";

    return distances;
}

vector<double> parallel_dijktras_goodscaling(const Graph &g, int src, int delta, int nthreads)
{
    vector<double> distances(g.num_nodes, INF);

    // flat array tracking each node's current bucket — O(1) stale check
    vector<int> node_bucket(g.num_nodes, -1);

    int max_bucket = 200000 / delta + 2;

    // per-thread flat bucket lists — direct index by bucket id, no hash map
    vector<vector<vector<int>>> bucket_lists(nthreads, vector<vector<int>>(max_bucket));

    // build thread-local CSR — each thread owns nodes where node % nthreads == tid
    // allocated in parallel so memory lands on each thread's local NUMA node
    vector<vector<int>> local_col_ind(nthreads);
    vector<vector<int>> local_data(nthreads);
    vector<vector<long>> local_row_ptr(nthreads); // offset into local_col_ind per owned node

#pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        for (int node = tid; node < g.num_nodes; node += nthreads)
        {
            local_row_ptr[tid].push_back(local_col_ind[tid].size());
            for (long idx = g.row_ptr[node]; idx < g.row_ptr[node + 1]; idx++)
            {
                local_col_ind[tid].push_back(g.col_ind[idx]);
                local_data[tid].push_back(g.data[idx]);
            }
        }
        local_row_ptr[tid].push_back(local_col_ind[tid].size()); // sentinel
    }

    vector<vector<vector<pair<int, double>>>> outgoing(nthreads, vector<vector<pair<int, double>>>(nthreads));
    vector<vector<int>> snapshots(nthreads);
    vector<vector<int>> local_processed(nthreads);

    // track min active bucket per thread — O(1) update instead of set insert
    vector<int> thread_min_bucket(nthreads, INT_MAX);

    int src_owner = src % nthreads;
    distances[src] = 0.0;
    node_bucket[src] = 0;
    bucket_lists[src_owner][0].push_back(src);
    thread_min_bucket[src_owner] = 0;

    double t_parallel = 0, t_merge = 0;
    double t_find_b = 0, t_work_avail = 0, t_snapshot = 0;
    double t_outgoing_clear = 0, t_flatten = 0, t_bucket_erase = 0;

    while (true)
    {
        // find global min non-empty bucket — O(nthreads) scan of min tracker
        double _t0 = omp_get_wtime();
        int b = INT_MAX;
#pragma omp parallel for num_threads(nthreads) reduction(min : b)
        for (int tid = 0; tid < nthreads; tid++)
            b = min(b, thread_min_bucket[tid]);
        t_find_b += omp_get_wtime() - _t0;

        if (b == INT_MAX)
            break;

        vector<int> processed_nodes;

        // check if any thread has non-stale work in bucket b
        auto work_available = [&]()
        {
            double _t = omp_get_wtime();
            for (int tid = 0; tid < nthreads; tid++)
                for (int node : bucket_lists[tid][b])
                    if (node_bucket[node] == b)
                    {
                        t_work_avail += omp_get_wtime() - _t;
                        return true;
                    }
            t_work_avail += omp_get_wtime() - _t;
            return false;
        };

        // process light edges until bucket b is stable ----------------------------
        while (work_available())
        {
            // parallel snapshot — O(1) node_bucket check per node
            double _ts = omp_get_wtime();
#pragma omp parallel for num_threads(nthreads)
            for (int tid = 0; tid < nthreads; tid++)
            {
                snapshots[tid].clear();
                local_processed[tid].clear();
                for (int node : bucket_lists[tid][b])
                    if (node_bucket[node] == b)
                        snapshots[tid].push_back(node);
                bucket_lists[tid][b].clear();
                // recompute thread min bucket after clearing b
                if (thread_min_bucket[tid] == b)
                {
                    thread_min_bucket[tid] = INT_MAX;
                    for (int bk = b + 1; bk < max_bucket; bk++)
                        if (!bucket_lists[tid][bk].empty())
                        {
                            thread_min_bucket[tid] = bk;
                            break;
                        }
                }
            }
            t_snapshot += omp_get_wtime() - _ts;

            double _tc = omp_get_wtime();
            for (auto &v : outgoing)
                for (auto &vv : v)
                    vv.clear();
            t_outgoing_clear += omp_get_wtime() - _tc;

            // parallel relaxation using thread-local CSR — NUMA-friendly reads
            double tp0 = omp_get_wtime();
#pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                int owned_idx = 0; // index into local_row_ptr[tid]
                for (int node : snapshots[tid])
                {
                    // node is owned by tid, so owned_idx = node / nthreads
                    int local_node = node / nthreads;
                    long start = local_row_ptr[tid][local_node];
                    long end = local_row_ptr[tid][local_node + 1];
                    for (long idx = start; idx < end; idx++)
                    {
                        int neighbor = local_col_ind[tid][idx];
                        int weight = local_data[tid][idx];
                        if (weight > delta)
                            continue;
                        double d_prime = distances[node] + weight;
                        if (d_prime < distances[neighbor])
                            outgoing[tid][neighbor % nthreads].push_back({neighbor, d_prime});
                    }
                }
            }
            t_parallel += omp_get_wtime() - tp0;

            // parallel merge — no unordered_map, direct apply with distance check
            double tm0 = omp_get_wtime();
#pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();

                // apply updates directly — tid owns all neighbors in outgoing[*][tid]
                // no race possible since neighbor % nthreads == tid exclusively
                for (int src_tid = 0; src_tid < nthreads; src_tid++)
                    for (auto &[neighbor, d_prime] : outgoing[src_tid][tid])
                        if (d_prime < distances[neighbor])
                        {
                            distances[neighbor] = d_prime;
                            int new_b = (int)d_prime / delta;
                            node_bucket[neighbor] = new_b;
                            bucket_lists[tid][new_b].push_back(neighbor);
                            if (new_b < thread_min_bucket[tid])
                                thread_min_bucket[tid] = new_b;
                        }

                for (int node : snapshots[tid])
                    local_processed[tid].push_back(node);
            }
            t_merge += omp_get_wtime() - tm0;

            double _tf = omp_get_wtime();
            for (int tid = 0; tid < nthreads; tid++)
                processed_nodes.insert(processed_nodes.end(), local_processed[tid].begin(), local_processed[tid].end());
            t_flatten += omp_get_wtime() - _tf;
        }

        // bucket b fully processed
        double _te = omp_get_wtime();
        for (int tid = 0; tid < nthreads; tid++)
            bucket_lists[tid][b].clear();
        t_bucket_erase += omp_get_wtime() - _te;

        // relax heavy edges from processed nodes (weight > delta) -----------------
        double _tc = omp_get_wtime();
        for (auto &v : outgoing)
            for (auto &vv : v)
                vv.clear();
        t_outgoing_clear += omp_get_wtime() - _tc;

        double tp0 = omp_get_wtime();
#pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            for (int node : processed_nodes)
            {
                if (node % nthreads != tid)
                    continue;
                if (distances[node] == INF || node_bucket[node] != b)
                    continue;
                int local_node = node / nthreads;
                long start = local_row_ptr[tid][local_node];
                long end = local_row_ptr[tid][local_node + 1];
                for (long idx = start; idx < end; idx++)
                {
                    int neighbor = local_col_ind[tid][idx];
                    int weight = local_data[tid][idx];
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
            for (int src_tid = 0; src_tid < nthreads; src_tid++)
                for (auto &[neighbor, d_prime] : outgoing[src_tid][tid])
                    if (d_prime < distances[neighbor])
                    {
                        distances[neighbor] = d_prime;
                        int new_b = (int)d_prime / delta;
                        node_bucket[neighbor] = new_b;
                        bucket_lists[tid][new_b].push_back(neighbor);
                        if (new_b < thread_min_bucket[tid])
                            thread_min_bucket[tid] = new_b;
                    }
        }
        t_merge += omp_get_wtime() - tm0;
    }

    cout << "  [timing] parallel relaxation : " << t_parallel << "s\n";
    cout << "  [timing] parallel merge      : " << t_merge << "s\n";
    cout << "  [timing] find min bucket     : " << t_find_b << "s\n";
    cout << "  [timing] work_available      : " << t_work_avail << "s\n";
    cout << "  [timing] snapshot            : " << t_snapshot << "s\n";
    cout << "  [timing] outgoing clear      : " << t_outgoing_clear << "s\n";
    cout << "  [timing] flatten processed   : " << t_flatten << "s\n";
    cout << "  [timing] bucket erase        : " << t_bucket_erase << "s\n";
    cout << "  [timing] parallel fraction   : "
         << (t_parallel / (t_parallel + t_merge)) * 100 << "%\n";

    return distances;
}

vector<double> parallel_dijktras_vector(const Graph &g, int src, int delta, int nthreads)
{
    vector<double> distances(g.num_nodes, INF);

    // flat array tracking each node's current bucket — O(1) stale check, no hash lookup
    vector<int> node_bucket(g.num_nodes, -1);

    // max possible bucket = max_distance / delta
    // weights 1-100, diameter ~500 hops -> max dist ~50000, max bucket ~1000
    int max_bucket = 200000 / delta + 2; // assumes max path distance < 200000

    // per-thread flat bucket lists — direct index by bucket id, no hash map
    vector<vector<vector<int>>> bucket_lists(nthreads, vector<vector<int>>(max_bucket));

    vector<vector<vector<pair<int, double>>>> outgoing(nthreads, vector<vector<pair<int, double>>>(nthreads));
    vector<vector<int>> snapshots(nthreads);
    vector<vector<int>> local_processed(nthreads);

    int src_owner = src % nthreads;
    distances[src] = 0.0;
    node_bucket[src] = 0;
    bucket_lists[src_owner][0].push_back(src); // bucket 0 = distance range [0, delta)

    // track non-empty buckets per thread to avoid scanning max_bucket every iteration
    vector<set<int>> active_buckets(nthreads);
    active_buckets[src_owner].insert(0);

    double t_parallel = 0, t_merge = 0;
    double t_find_b = 0, t_work_avail = 0, t_snapshot = 0;
    double t_outgoing_clear = 0, t_flatten = 0, t_bucket_erase = 0;

    while (true)
    {
        // find global min non-empty bucket — parallel reduce over active_buckets sets
        double _t0 = omp_get_wtime();
        int b = INT_MAX;
#pragma omp parallel for num_threads(nthreads) reduction(min : b)
        for (int tid = 0; tid < nthreads; tid++)
            if (!active_buckets[tid].empty())
                b = min(b, *active_buckets[tid].begin());
        t_find_b += omp_get_wtime() - _t0;

        if (b == INT_MAX)
            break;

        vector<int> processed_nodes;

        // check if any thread has non-stale work in bucket b
        auto work_available = [&]()
        {
            double _t = omp_get_wtime();
            for (int tid = 0; tid < nthreads; tid++)
                for (int node : bucket_lists[tid][b])
                    if (node_bucket[node] == b)
                    {
                        t_work_avail += omp_get_wtime() - _t;
                        return true;
                    }
            t_work_avail += omp_get_wtime() - _t;
            return false;
        };

        // process light edges until bucket b is stable ----------------------------
        while (work_available())
        {
            // parallel snapshot — filter stale, O(1) node_bucket check per node
            double _ts = omp_get_wtime();
#pragma omp parallel for num_threads(nthreads)
            for (int tid = 0; tid < nthreads; tid++)
            {
                snapshots[tid].clear();
                local_processed[tid].clear();
                for (int node : bucket_lists[tid][b])
                    if (node_bucket[node] == b)
                        snapshots[tid].push_back(node);
                bucket_lists[tid][b].clear(); // clear — lazy deletion consumed
                active_buckets[tid].erase(b);
            }
            t_snapshot += omp_get_wtime() - _ts;

            // outgoing[src_tid][dest_tid] — thread tid writes only to outgoing[tid]
            double _tc = omp_get_wtime();
            for (auto &v : outgoing)
                for (auto &vv : v)
                    vv.clear();
            t_outgoing_clear += omp_get_wtime() - _tc;

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

                // apply best updates — O(1) array write, no hash lookup for bucket
                for (auto &[neighbor, d_prime] : best_updates)
                    if (d_prime < distances[neighbor])
                    {
                        distances[neighbor] = d_prime;
                        int new_b = (int)d_prime / delta;
                        node_bucket[neighbor] = new_b; // O(1) flat array write
                        bucket_lists[tid][new_b].push_back(neighbor);
                        active_buckets[tid].insert(new_b);
                    }

                for (int node : snapshots[tid])
                    local_processed[tid].push_back(node);
            }
            t_merge += omp_get_wtime() - tm0;

            double _tf = omp_get_wtime();
            for (int tid = 0; tid < nthreads; tid++)
                processed_nodes.insert(processed_nodes.end(), local_processed[tid].begin(), local_processed[tid].end());
            t_flatten += omp_get_wtime() - _tf;
        }

        // bucket b fully processed — clear from all threads
        double _te = omp_get_wtime();
        for (int tid = 0; tid < nthreads; tid++)
        {
            bucket_lists[tid][b].clear();
            active_buckets[tid].erase(b);
        }
        t_bucket_erase += omp_get_wtime() - _te;

        // relax heavy edges from processed nodes (weight > delta) -----------------
        double _tc = omp_get_wtime();
        for (auto &v : outgoing)
            for (auto &vv : v)
                vv.clear();
        t_outgoing_clear += omp_get_wtime() - _tc;

        double tp0 = omp_get_wtime();
#pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            for (int node : processed_nodes)
            {
                if (node % nthreads != tid)
                    continue;
                if (distances[node] == INF || node_bucket[node] != b)
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

            unordered_map<int, double> best_updates;
            for (int src_tid = 0; src_tid < nthreads; src_tid++)
                for (auto &[neighbor, d_prime] : outgoing[src_tid][tid])
                    if (!best_updates.count(neighbor) || d_prime < best_updates[neighbor])
                        best_updates[neighbor] = d_prime;

            for (auto &[neighbor, d_prime] : best_updates)
                if (d_prime < distances[neighbor])
                {
                    distances[neighbor] = d_prime;
                    int new_b = (int)d_prime / delta;
                    node_bucket[neighbor] = new_b;
                    bucket_lists[tid][new_b].push_back(neighbor);
                    active_buckets[tid].insert(new_b);
                }
        }
        t_merge += omp_get_wtime() - tm0;
    }

    cout << "  [timing] parallel relaxation : " << t_parallel << "s\n";
    cout << "  [timing] parallel merge      : " << t_merge << "s\n";
    cout << "  [timing] find min bucket     : " << t_find_b << "s\n";
    cout << "  [timing] work_available      : " << t_work_avail << "s\n";
    cout << "  [timing] snapshot            : " << t_snapshot << "s\n";
    cout << "  [timing] outgoing clear      : " << t_outgoing_clear << "s\n";
    cout << "  [timing] flatten processed   : " << t_flatten << "s\n";
    cout << "  [timing] bucket erase        : " << t_bucket_erase << "s\n";
    cout << "  [timing] parallel fraction   : "
         << (t_parallel / (t_parallel + t_merge)) * 100 << "%\n";

    return distances;
}

vector<double> parallel_dijktras_bestusinghashmaps(const Graph &g, int src, int delta, int nthreads)
{
    vector<double> distances(g.num_nodes, INF);

    // each thread owns nodes where node % nthreads == tid
    // vector<int> buckets with lazy deletion — no erase, stale entries filtered on read
    vector<unordered_map<int, vector<int>>> tbuckets(nthreads);
    vector<vector<vector<pair<int, double>>>> outgoing(nthreads, vector<vector<pair<int, double>>>(nthreads));

    // snapshots as vector<int> — O(1) clear, parallelized across threads
    vector<vector<int>> snapshots(nthreads);
    vector<vector<int>> local_processed(nthreads);

    int src_owner = src % nthreads;
    distances[src] = 0.0;
    tbuckets[src_owner][0].push_back(src); // bucket 0 = distance range [0, delta)

    double t_parallel = 0, t_merge = 0;
    double t_find_b = 0, t_work_avail = 0, t_snapshot = 0;
    double t_outgoing_clear = 0, t_flatten = 0, t_bucket_erase = 0;

    while (true)
    {
        // find global min non-empty bucket across all threads — parallel reduce
        double _t0 = omp_get_wtime();
        int b = INT_MAX;
#pragma omp parallel for num_threads(nthreads) reduction(min : b)
        for (int tid = 0; tid < nthreads; tid++)
            for (auto &[bucket, nodes] : tbuckets[tid])
                if (!nodes.empty() && bucket < b)
                    b = bucket;
        t_find_b += omp_get_wtime() - _t0;

        // no more buckets to process
        if (b == INT_MAX)
            break;

        vector<int> processed_nodes;

        // check if any thread has non-stale work remaining in bucket b
        auto work_available = [&]()
        {
            double _t = omp_get_wtime();
            for (int tid = 0; tid < nthreads; tid++)
                if (tbuckets[tid].count(b))
                    for (int node : tbuckets[tid][b])
                        if ((int)distances[node] / delta == b)
                        {
                            t_work_avail += omp_get_wtime() - _t;
                            return true;
                        }
            t_work_avail += omp_get_wtime() - _t;
            return false;
        };

        // process light edges until bucket b is stable ----------------------------
        while (work_available())
        {
            // parallel snapshot — each thread copies its own bucket slice
            // stale entries filtered out here — only keep nodes still in bucket b
            double _ts = omp_get_wtime();
#pragma omp parallel for num_threads(nthreads)
            for (int tid = 0; tid < nthreads; tid++)
            {
                snapshots[tid].clear();
                local_processed[tid].clear();
                if (tbuckets[tid].count(b))
                {
                    for (int node : tbuckets[tid][b])
                        if ((int)distances[node] / delta == b)
                            snapshots[tid].push_back(node);
                    tbuckets[tid][b].clear(); // clear all — stale and fresh consumed
                }
            }
            t_snapshot += omp_get_wtime() - _ts;

            // outgoing[src_tid][dest_tid] — thread tid writes only to outgoing[tid]
            double _tc = omp_get_wtime();
            for (auto &v : outgoing)
                for (auto &vv : v)
                    vv.clear();
            t_outgoing_clear += omp_get_wtime() - _tc;

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

                // apply best updates — no erase, just push to new bucket
                // stale entries in old bucket will be filtered at snapshot time
                for (auto &[neighbor, d_prime] : best_updates)
                    if (d_prime < distances[neighbor])
                    {
                        distances[neighbor] = d_prime;
                        tbuckets[tid][(int)d_prime / delta].push_back(neighbor);
                    }

                for (int node : snapshots[tid])
                    local_processed[tid].push_back(node);
            }
            t_merge += omp_get_wtime() - tm0;

            double _tf = omp_get_wtime();
            for (int tid = 0; tid < nthreads; tid++)
                processed_nodes.insert(processed_nodes.end(), local_processed[tid].begin(), local_processed[tid].end());
            t_flatten += omp_get_wtime() - _tf;
        }

        // bucket b fully processed — erase from all threads
        double _te = omp_get_wtime();
        for (int tid = 0; tid < nthreads; tid++)
            tbuckets[tid].erase(b);
        t_bucket_erase += omp_get_wtime() - _te;

        // relax heavy edges from processed nodes (weight > delta) -----------------
        double _tc = omp_get_wtime();
        for (auto &v : outgoing)
            for (auto &vv : v)
                vv.clear();
        t_outgoing_clear += omp_get_wtime() - _tc;

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

            // apply best updates — no erase, lazy deletion handles stale entries
            for (auto &[neighbor, d_prime] : best_updates)
                if (d_prime < distances[neighbor])
                {
                    distances[neighbor] = d_prime;
                    tbuckets[tid][(int)d_prime / delta].push_back(neighbor);
                }
        }
        t_merge += omp_get_wtime() - tm0;
    }

    cout << "  [timing] parallel relaxation : " << t_parallel << "s\n";
    cout << "  [timing] parallel merge      : " << t_merge << "s\n";
    cout << "  [timing] find min bucket     : " << t_find_b << "s\n";
    cout << "  [timing] work_available      : " << t_work_avail << "s\n";
    cout << "  [timing] snapshot            : " << t_snapshot << "s\n";
    cout << "  [timing] outgoing clear      : " << t_outgoing_clear << "s\n";
    cout << "  [timing] flatten processed   : " << t_flatten << "s\n";
    cout << "  [timing] bucket erase        : " << t_bucket_erase << "s\n";
    cout << "  [timing] parallel fraction   : "
         << (t_parallel / (t_parallel + t_merge)) * 100 << "%\n";

    return distances;
}

vector<double> parallel_dijktras_bestsofar(const Graph &g, int src, int delta, int nthreads)
{
    vector<double> distances(g.num_nodes, INF);

    // each thread owns nodes where node % nthreads == tid
    // unordered_set for O(1) erase in merge
    vector<unordered_map<int, unordered_set<int>>> tbuckets(nthreads);
    vector<vector<vector<pair<int, double>>>> outgoing(nthreads, vector<vector<pair<int, double>>>(nthreads));

    // snapshots as vector<int> — O(1) clear, parallelized across threads
    vector<vector<int>> snapshots(nthreads);
    vector<vector<int>> local_processed(nthreads);

    int src_owner = src % nthreads;
    distances[src] = 0.0;
    tbuckets[src_owner][0].insert(src); // bucket 0 = distance range [0, delta)

    double t_parallel = 0, t_merge = 0;
    double t_find_b = 0, t_work_avail = 0, t_snapshot = 0;
    double t_outgoing_clear = 0, t_flatten = 0, t_bucket_erase = 0;

    while (true)
    {
        // find global min non-empty bucket across all threads — parallel reduce
        double _t0 = omp_get_wtime();
        int b = INT_MAX;
#pragma omp parallel for num_threads(nthreads) reduction(min : b)
        for (int tid = 0; tid < nthreads; tid++)
            for (auto &[bucket, nodes] : tbuckets[tid])
                if (!nodes.empty() && bucket < b)
                    b = bucket;
        t_find_b += omp_get_wtime() - _t0;

        // no more buckets to process
        if (b == INT_MAX)
            break;

        vector<int> processed_nodes;

        // check if any thread has work remaining in bucket b
        auto work_available = [&]()
        {
            double _t = omp_get_wtime();
            for (int tid = 0; tid < nthreads; tid++)
                if (tbuckets[tid].count(b) && !tbuckets[tid][b].empty())
                {
                    t_work_avail += omp_get_wtime() - _t;
                    return true;
                }
            t_work_avail += omp_get_wtime() - _t;
            return false;
        };

        // process light edges until bucket b is stable ----------------------------
        while (work_available())
        {
            // parallel snapshot — each thread copies its own bucket slice
            // vector clear is O(1), assign from set is parallelized across threads
            double _ts = omp_get_wtime();
#pragma omp parallel for num_threads(nthreads)
            for (int tid = 0; tid < nthreads; tid++)
            {
                snapshots[tid].clear();
                local_processed[tid].clear();
                if (tbuckets[tid].count(b) && !tbuckets[tid][b].empty())
                {
                    auto &s = tbuckets[tid][b];
                    snapshots[tid].assign(s.begin(), s.end());
                    s.clear(); // keeps hash table allocated for future inserts
                }
            }
            t_snapshot += omp_get_wtime() - _ts;

            // outgoing[src_tid][dest_tid] — thread tid writes only to outgoing[tid]
            double _tc = omp_get_wtime();
            for (auto &v : outgoing)
                for (auto &vv : v)
                    vv.clear();
            t_outgoing_clear += omp_get_wtime() - _tc;

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

            double _tf = omp_get_wtime();
            for (int tid = 0; tid < nthreads; tid++)
                processed_nodes.insert(processed_nodes.end(), local_processed[tid].begin(), local_processed[tid].end());
            t_flatten += omp_get_wtime() - _tf;
        }

        // bucket b is now stable — clear it from all threads
        double _te = omp_get_wtime();
        for (int tid = 0; tid < nthreads; tid++)
            tbuckets[tid].erase(b);
        t_bucket_erase += omp_get_wtime() - _te;

        // relax heavy edges from processed nodes (weight > delta) -----------------
        double _tc = omp_get_wtime();
        for (auto &v : outgoing)
            for (auto &vv : v)
                vv.clear();
        t_outgoing_clear += omp_get_wtime() - _tc;

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

    cout << "  [timing] parallel relaxation : " << t_parallel << "s\n";
    cout << "  [timing] parallel merge      : " << t_merge << "s\n";
    cout << "  [timing] find min bucket     : " << t_find_b << "s\n";
    cout << "  [timing] work_available      : " << t_work_avail << "s\n";
    cout << "  [timing] snapshot            : " << t_snapshot << "s\n";
    cout << "  [timing] outgoing clear      : " << t_outgoing_clear << "s\n";
    cout << "  [timing] flatten processed   : " << t_flatten << "s\n";
    cout << "  [timing] bucket erase        : " << t_bucket_erase << "s\n";
    cout << "  [timing] parallel fraction   : "
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
