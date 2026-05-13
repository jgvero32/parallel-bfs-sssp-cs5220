/*
In progress...


Parallel BFS for roadNet-CA
How to run:
   Create an interactive session on Perlmutter
   From build/, run make
   srun -N 1 --ntasks-per-node=<num processors> ./bfs_parallel 
../datasets/<dataset>.txt <source node>
   Ex. srun -N 1 --ntasks-per-node=16 ./bfs_parallel 
../datasets/roadNet-CA.txt 0
*/

#include "graph_utils.h"
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <queue>
#include <string>
#include <vector>

// Stores the rank's nodes distance from source node
static std::vector<int> dists;
static std::vector<bool> visited;
// This rank's local frontier (contains node ids in the frontier)
static std::vector<int> frontier;
static int start_row;
static int end_row; // non-inclusive
static std::vector<int> displacements;
static std::vector<int> rows_per_proc;
static std::vector<int> result;

/**
 *
 */
void initialize(const Graph &graph, int source, int rank, int num_procs) {
    // auto t0 = std::chrono::steady_clock::now();

    int graph_size = graph.num_nodes;

    if (rank >= graph.num_nodes)
        return;

    // Divide nodes evenly between processors by balancing number of outgoing edges
    displacements.resize(num_procs);
    rows_per_proc.resize(num_procs);
    displacements[0] = 0;

    int graph_edges = graph.num_edges;
    int ideal_edges = (graph_edges + num_procs - 1) / num_procs;

    int next_proc_to_assign = 1;
    for (int node_id = 0; node_id < graph_size && next_proc_to_assign < num_procs; ++node_id) {
        if (graph.row_ptr[node_id] >= next_proc_to_assign * ideal_edges) {
            displacements[next_proc_to_assign++] = node_id;
        }
    }
    while (next_proc_to_assign < num_procs) {
        displacements[next_proc_to_assign++] = graph_size;
    }

    for (int p = 0; p < num_procs - 1; ++p) {
        rows_per_proc[p] = displacements[p + 1] - displacements[p];
    }
    rows_per_proc[num_procs - 1] = graph_size - displacements[num_procs - 1];

    // Determine local start and end rows
    start_row = displacements[rank];
    if (displacements.size() - 1 == rank) {
        end_row = graph.num_nodes;
    } else {
        end_row = displacements[rank + 1];
    }

    // Only store distances for this rank's nodes [start_row, end_row]
    dists.resize(end_row - start_row);
    for (int i = 0; i < end_row - start_row; ++i) {
        dists[i] = -1; // Signifies a node hasn't been visited
    }
    visited.resize(graph.num_nodes);
    for (int i = 0; i < graph.num_nodes; ++i) {
        visited[i] = false;
    }

    // Create the first frontier
    if (start_row <= source && source < end_row) {
        frontier.push_back(source);
        dists[source - start_row] = 0;
        visited[source] = true;
    }

    // if (rank == 0) {
    //     auto t1 = std::chrono::steady_clock::now();
    //     double elapsed = std::chrono::duration<double>(t1 - t0).count();
    //     std::cout << "Initialize time: " << elapsed*100000 << std::endl;
    // }
}

// Finds the owner process of a node with binary search
int get_owner(int node) {
    int lo = 0, hi = displacements.size() - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (node < displacements[mid]) {
            hi = mid - 1;
        } else if (mid + 1 < displacements.size() &&
                   node >= displacements[mid + 1]) {
            lo = mid + 1;
        } else {
            return mid;
        }
    }
    return -1;
}

/**
* Each processor owns a partition of the nodes and is in charge of updating
* the distances of those nodes from the source node, as well as updating its
* frontier and reporting discovered neighbors to other processes (all-to-all).
*/
void parallel_bfs(const Graph &g, int rank, int num_procs) {
    if (rank >= g.num_nodes)
        return;

    int distance = 1;

    // double avg_search_time = 0;
    // double avg_communicate_time = 0;
    // double avg_update_time = 0;

    while (true) {
        // auto t0 = std::chrono::steady_clock::now();
        // Collects the discovered nodes (outgoing edges from nodes in the frontier)
        // to send to their owner processor
        std::vector<std::vector<int>> discovered_nodes(num_procs);

        // (u, v): edges from node u to node v
        for (int u : frontier) {
            for (int edge = g.row_ptr[u]; edge < g.row_ptr[u + 1]; ++edge) {
                int v = g.col_ind[edge];
                if (!visited[v]) {
                    int owner = get_owner(v);
                    discovered_nodes[owner].push_back(v);
                }
            }
        }

        std::vector<int> send_cts(num_procs), send_displacements(num_procs);
        int total_send = 0;
        for (int i = 0; i < num_procs; ++i) {
            send_cts[i] = discovered_nodes[i].size();
            send_displacements[i] = total_send;
            total_send += send_cts[i];
        }

        // Flatten discovered nodes into a 1D vector so it can be sent with MPI
        std::vector<int> send_data(total_send);
        for (int i = 0; i < num_procs; ++i) {
            std::copy(discovered_nodes[i].begin(), discovered_nodes[i].end(),
                send_data.begin() + send_displacements[i]);
        }

        // auto t2 = std::chrono::steady_clock::now();
        // double elapsed = std::chrono::duration<double>(t2 - t0).count();
        // avg_search_time += elapsed;

        // Exchange the expected counts to receive/send with all other ranks
        std::vector<int> recv_cts(num_procs);
        MPI_Alltoall(send_cts.data(), 1, MPI_INT, recv_cts.data(), 1,
            MPI_INT, MPI_COMM_WORLD);

        // Create receiving buffer and displacements (for how much data is expected from other ranks)
        std::vector<int> recv_displacements(num_procs);
        int total_recv = 0;
        for (int i = 0; i < num_procs; ++i) {
            recv_displacements[i] = total_recv;
            total_recv += recv_cts[i];
        }
        std::vector<int> recv_data(total_recv);

        // Exchange nodes in the new frontier to their respective owner processes
        // Cost: (all neighbors of frontier nodes across ranks) x num_procs
        MPI_Alltoallv(send_data.data(), send_cts.data(), send_displacements.data(), MPI_INT,
                    recv_data.data(), recv_cts.data(), recv_displacements.data(), MPI_INT,
                    MPI_COMM_WORLD);

        // auto t3 = std::chrono::steady_clock::now();
        // elapsed = std::chrono::duration<double>(t3 - t2).count();
        // avg_communicate_time += elapsed;

        // Each processor updates its partition of node distances & creates its new frontier
        // Does an OR operation over the copies of frontiers from all processes 
        std::vector<int> next_frontier;
        int has_frontier = 0;

        for (int v : recv_data) {
            if (dists[v - start_row] == -1) {
                dists[v - start_row] = distance;
                next_frontier.push_back(v);
                has_frontier = 1;
            }
        }

        // Check if any rank has a non-zero new frontier
        int global_frontier;
        MPI_Allreduce(&has_frontier, &global_frontier, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);

        // No new nodes to explore
        if (global_frontier == 0) break;

        frontier.swap(next_frontier);
        distance++;

        // auto t1 = std::chrono::steady_clock::now();
        // elapsed = std::chrono::duration<double>(t1 - t3).count();
        // avg_update_time += elapsed;
    }
    // std::cout << "Rank: " << rank << ". Average search time: " << avg_search_time/distance*100000 << std::endl;
    // std::cout << "Rank: " << rank << ". Average communicate time: " << avg_communicate_time/distance*100000 << std::endl;
    // std::cout << "Rank: " << rank << ". Average update time: " << avg_update_time/distance*100000 << std::endl;
}

/**
 * Gathers the partitions of the dists vector onto rank 0
 */
void gather_result(int num_nodes, int rank, int num_procs) {
    auto t0 = std::chrono::steady_clock::now();
    if (rank >= num_nodes)
        return;

    if (rank == 0) {
        result.resize(num_nodes);
        MPI_Gatherv(dists.data(), dists.size(), MPI_INT,
                    result.data(), rows_per_proc.data(), displacements.data(),
                    MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(dists.data(), dists.size(), MPI_INT, NULL, NULL,
                    NULL, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    // if (rank == 0) {
    //     auto t1 = std::chrono::steady_clock::now();
    //     double elapsed = std::chrono::duration<double>(t1 - t0).count();
    //     std::cout << "Gather time: " << elapsed*100000 << std::endl;
    // }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <graph_file> <source_node>"
                  << std::endl;
        return 1;
    }

    int num_procs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const std::string dataset_file_name = argv[1];
    int source = std::stoi(argv[2]);

    // Rank 0 loads the graph, then broadcasts it to other ranks
    Graph g;
    if (rank == 0) {
        std::cout << "Loading graph from: " << dataset_file_name << std::endl;
        try {
            g = load_graph(dataset_file_name, false);
        } catch (const std::exception &e) {
            std::cerr << "Error loading the graph??" << std::endl;
            return 1;
        }
    }

    MPI_Bcast(&g.num_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&g.num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);
    g.row_ptr.resize(g.num_nodes + 1);
    g.col_ind.resize(g.num_edges);
    MPI_Bcast(g.row_ptr.data(), g.num_nodes + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g.col_ind.data(), g.num_edges, MPI_INT, 0, MPI_COMM_WORLD);

    if (source < 0 || source >= g.num_nodes) {
        std::cerr << "Source node " << source << " out of range [0, "
                  << g.num_nodes << ")\n";
        return 1;
    }

    if (rank == 0) {
        std::cout << "Running parallel BFS from source node: " << source
                  << std::endl;
    }

    auto t0 = std::chrono::steady_clock::now();

    initialize(g, source, rank, num_procs);

    parallel_bfs(g, rank, num_procs);

    gather_result(g.num_nodes, rank, num_procs);

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    if (rank == 0) {
        std::cout << "\n----- Parallel BFS Results -----\n";
        std::cout << "  Source node      : " << source << std::endl;
        std::cout << "  Nodes visited    : " << nodes_visited(result)
                  << std::endl;
        std::cout << "  Node ID space    : " << g.num_nodes
                  << " (max_node_id + 1)" << std::endl;
        print_diameter(result);
        std::cout << "  Elapsed time     : " << elapsed << " seconds\n";

        print_distances(result, 50);
    }

    MPI_Finalize();

    return 0;
}
