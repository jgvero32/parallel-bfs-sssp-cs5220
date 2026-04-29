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
// This rank's local frontier
static std::vector<bool> frontier;
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
    frontier.reserve(graph.num_nodes);
    for (int i = 0; i < graph.num_nodes; ++i) {
        dists.push_back(-1); // Signifies a node hasn't been visited
        frontier.push_back(false);
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
        frontier[source] = true;
        dists[source] = 0;
    }
}

int get_owner(int node) {
    // binary search over displacements
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

void parallel_bfs(const Graph &g, int rank, int num_procs) {
    if (rank >= g.num_nodes)
        return;

    std::vector<int> frontier_nodes;

    // initialize frontier_nodes from your boolean frontier
    for (int i = start_row; i < end_row; ++i) {
        if (frontier[i]) frontier_nodes.push_back(i);
    }

    int distance = 1;

    while (true) {
        // 1. Build send buffers (sparse)
        std::vector<std::vector<int>> sendbuf(num_procs);

        for (int u : frontier_nodes) {
            for (int e = g.row_ptr[u]; e < g.row_ptr[u + 1]; ++e) {
                int v = g.col_ind[e];
                int owner = get_owner(v);
                sendbuf[owner].push_back(v);
            }
        }

        // 2. Build sendcounts + displs
        std::vector<int> sendcounts(num_procs), sdispls(num_procs);
        int total_send = 0;

        for (int i = 0; i < num_procs; ++i) {
            sendcounts[i] = sendbuf[i].size();
            sdispls[i] = total_send;
            total_send += sendcounts[i];
        }

        // 3. Flatten send buffer
        std::vector<int> senddata(total_send);
        for (int i = 0; i < num_procs; ++i) {
            std::copy(sendbuf[i].begin(), sendbuf[i].end(),
                    senddata.begin() + sdispls[i]);
        }

        // 4. Exchange counts
        std::vector<int> recvcounts(num_procs);
        MPI_Alltoall(sendcounts.data(), 1, MPI_INT,
                    recvcounts.data(), 1, MPI_INT,
                    MPI_COMM_WORLD);

        // 5. Build recv displacements
        std::vector<int> rdispls(num_procs);
        int total_recv = 0;
        for (int i = 0; i < num_procs; ++i) {
            rdispls[i] = total_recv;
            total_recv += recvcounts[i];
        }

        std::vector<int> recvdata(total_recv);

        // 6. Exchange actual node IDs
        MPI_Alltoallv(senddata.data(), sendcounts.data(), sdispls.data(), MPI_INT,
                    recvdata.data(), recvcounts.data(), rdispls.data(), MPI_INT,
                    MPI_COMM_WORLD);

        // 7. Build next frontier (owner only!)
        std::vector<int> next_frontier;
        int has_frontier = 0;

        for (int v : recvdata) {
            if (dists[v] == -1) {
                dists[v] = distance;
                next_frontier.push_back(v);
                has_frontier = 1;
            }
        }

        // 8. Global termination check
        int global_active;
        MPI_Allreduce(&has_frontier, &global_active, 1, MPI_INT, MPI_SUM,
                    MPI_COMM_WORLD);

        if (global_active == 0) break;

        frontier_nodes.swap(next_frontier);
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
