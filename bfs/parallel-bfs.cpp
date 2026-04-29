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

// Stores the distances from source node
static std::vector<int> dists;
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
    if (rank >= graph.num_nodes)
        return;

    dists.reserve(graph.num_nodes);
    for (int i = 0; i < graph.num_nodes; ++i) {
        dists.push_back(-1); // Signifies a node hasn't been visited
    }

    // TODO: load-balancing nodes based on number of outgoing edges
    // Divide nodes evenly between processors
    int base_rows = graph.num_nodes / num_procs;
    int remainder_rows =
        graph.num_nodes % num_procs; // Number of ranks with extra row
    displacements.reserve(num_procs);
    rows_per_proc.reserve(num_procs);

    int disp = 0;
    for (int i = 0; i < num_procs; ++i) {
        displacements.push_back(disp);
        if (i < remainder_rows) {
            disp += base_rows + 1;
            rows_per_proc.push_back(base_rows + 1);
        } else {
            disp += base_rows;
            rows_per_proc.push_back(base_rows);
        }
    }

    start_row = displacements[rank];
    if (displacements.size() - 1 == rank) {
        end_row = graph.num_nodes;
    } else {
        end_row = displacements[rank + 1];
    }

    // Create the first frontier
    if (start_row <= source && source < end_row) {
        frontier.push_back(source);
        dists[source] = 0;
    }
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

    while (true) {
        // Collects the discovered nodes (outgoing edges from nodes in the frontier)
        // to send to their owner processor
        std::vector<std::vector<int>> discovered_nodes(num_procs);

        // (u, v): edges from node u to node v
        for (int u : frontier) {
            for (int edge = g.row_ptr[u]; edge < g.row_ptr[u + 1]; ++edge) {
                int v = g.col_ind[edge];
                int owner = get_owner(v);
                discovered_nodes[owner].push_back(v);
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

        // Each processor updates its partition of node distances & creates its new frontier
        // Does an OR operation over the copies of frontiers from all processes 
        std::vector<int> next_frontier;
        int has_frontier = 0;

        for (int v : recv_data) {
            if (dists[v] == -1) {
                dists[v] = distance;
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
    }
}

/**
 * Gathers the partitions of the dists vector onto rank 0
 */
void gather_result(int num_nodes, int rank, int num_procs) {
    if (rank >= num_nodes)
        return;
    // TODO: can I just use a pointer to start index instead of making a new
    // vector?
    std::vector<int> rank_dists(dists.begin() + start_row,
                                dists.begin() + end_row);

    if (rank == 0) {
        result.resize(num_nodes);
        MPI_Gatherv(rank_dists.data(), rank_dists.size(), MPI_INT,
                    result.data(), rows_per_proc.data(), displacements.data(),
                    MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(rank_dists.data(), rank_dists.size(), MPI_INT, NULL, NULL,
                    NULL, MPI_INT, 0, MPI_COMM_WORLD);
    }
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
        std::cout << "  Elapsed time     : " << elapsed << " seconds\n";

        print_distances(result, 50);
    }

    MPI_Finalize();

    return 0;
}
