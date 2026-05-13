#include "graph_utils.h"
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

/*
    To run:
    cd bfs
    salloc -N 1 -C cpu -q interactive -t 01:00:00 -A m4341 -n 1
    mpicxx -O2 -std=c++23 2d-parallel-bfs.cpp -o 2d_bfs_parallel
    srun -n 4 ./2d_bfs_parallel ../datasets/soc-LiveJournal1.txt 0
*/

// sets local_row_ptr and local_col_ind to contain a CSR version of the submatrix for the current processor
void build_local_submatrix(
    const std::vector<int>& full_row_ptr,
    const std::vector<int>& full_col_ind,
    int row_start, int total_my_rows,
    int col_start, int total_my_columns,
    std::vector<int>& local_row_ptr,
    std::vector<int>& local_col_ind)
{
    local_row_ptr.resize(total_my_rows + 1, 0);
    local_col_ind.clear();

    for (int i = 0; i < total_my_rows; i++) {
        int global_row = row_start + i;
        local_row_ptr[i] = local_col_ind.size();
        for (int idx = full_row_ptr[global_row]; idx < full_row_ptr[global_row + 1]; idx++) {
            int c = full_col_ind[idx];
            if (c >= col_start && c < col_start + total_my_columns) {
                local_col_ind.push_back(c - col_start);
            }
        }
    }
    local_row_ptr[total_my_rows] = local_col_ind.size();
}

std::vector<int> bfs_2d_mpi(
    const std::vector<int>& local_row_ptr,
    const std::vector<int>& local_col_ind,
    int n, int src,
    int total_my_rows, int total_my_columns,
    int row_start,  int col_start,
    MPI_Comm comm, MPI_Comm row_comm, MPI_Comm col_comm,
    int processor_row, int processor_column, int pr, int pc,
    const std::vector<int>& to_vertices_range) {

    int comm_size = pr * pc;

    std::vector<int> local_dist(total_my_rows, -1); // stores the distance of each "to" nodeID this processor owns
    std::vector<int> local_frontier(total_my_columns, 0); // frontier contains 1 or 0 for all "from" nodeIDs are in the current frontier 

    // seeds the source vertex into the local_dist and local_frontier vectors in their correct (adjusted) spot
    if (src >= row_start && src < row_start + total_my_rows)
        local_dist[src - row_start] = 0; // ex: if row_start = 2 and src = 5 local_dist[3] is the spot for the 5th row
    if (src >= col_start && src < col_start + total_my_columns)
        local_frontier[src - col_start] = 1;

    int bfs_level = 1;
    std::vector<int> partial_spmv_result(total_my_rows); // this holds the SpMV result of the current process (before row communication)
    std::vector<int> reduced_result(total_my_rows); // this holds the combined vector addition result (after row communication) 
    std::vector<int> local_new_verts;
    std::vector<int> all_counts(comm_size);
    std::vector<int> displacements(comm_size);
    std::vector<int> all_new_verts;

    while (true) {
        // This is the partial SpMV part of the process!
        partial_spmv_result.assign(total_my_rows, 0); // zero out partial_spmv_result every level iteration
        for (int i = 0; i < total_my_rows; i++) {
            if (local_dist[i] != -1) continue; // don't go through already visited rows
            // this is basically ANDing to matrix multiply
            for (int index = local_row_ptr[i]; index < local_row_ptr[i + 1]; index++) { // for each "from" edge in row i -> aka when the element in the submatrix is 1
                if (local_frontier[local_col_ind[index]]) { // if the element local_frontier[local_col_ind[index]] is 1, then we can set the partial_spmv_result for this row to have 1 
                    partial_spmv_result[i] = 1;
                    break; // break bc we've already set it to 1 and don't need to calculate the other ANDs
                }
            }
        }

        // this is just the the partial SpMV vectors from each processor added together
        MPI_Allreduce(partial_spmv_result.data(), reduced_result.data(), total_my_rows, MPI_INT, MPI_LOR, row_comm); // every processor in a processor row will get a reduced version of the partial_spmv_result

        // we want a list of the new vertices in the frontier
        local_new_verts.clear();
        for (int i = 0; i < total_my_rows; i++) {
            if (reduced_result[i] && local_dist[i] == -1) { // vertices that have already been visited (local_dist[i] == -1) should not appear in the new frontier
                local_dist[i] = bfs_level; // update the distance of the "to" vertex
                local_new_verts.push_back(row_start + i);
            }
        }


        int local_count = local_new_verts.size();
        MPI_Allgather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, comm); // every process shares the number of new vertices it found with all other processes
        // after this all_counts looking like: [2, 0, 1, 3] (for 4 processors for ex)

        int total_new = 0;
        for (int i = 0; i < comm_size; i++) {
            displacements[i] = total_new;
            total_new += all_counts[i];
        }

        if (total_new == 0) break;  // no new vertices discovered, so BFS is done

        all_new_verts.resize(total_new);
        MPI_Allgatherv(local_new_verts.data(), local_count, MPI_INT, all_new_verts.data(), all_counts.data(), displacements.data(), MPI_INT, comm); // displacements will tell AllGatherv where to put each process' new vertices

        // updates this processor's frontier with newly discovered vertices
        local_frontier.assign(total_my_columns, 0); // clear out the old frontier
        for (int v : all_new_verts) {
            if (v >= col_start && v < col_start + total_my_columns)
                local_frontier[v - col_start] = 1;
        }

        bfs_level++;
    }

    // gather the full distances vector to rank 0
    std::vector<int> full_dist;
    if (processor_column == 0) {
        std::vector<int> recvcounts(pr), gv_displs(pr);
        for (int r = 0; r < pr; r++) {
            recvcounts[r] = to_vertices_range[r + 1] - to_vertices_range[r];
            gv_displs[r]  = to_vertices_range[r];
        }

        if (processor_row == 0) {
            full_dist.resize(n, -1);
            MPI_Gatherv(local_dist.data(), total_my_rows, MPI_INT, full_dist.data(), recvcounts.data(), gv_displs.data(), MPI_INT, 0, col_comm);

        }
        else {
            MPI_Gatherv(local_dist.data(), total_my_rows, MPI_INT, nullptr, recvcounts.data(), gv_displs.data(), MPI_INT, 0, col_comm);

        }
    }
    return full_dist;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <graph_file> <source_node>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    Graph g;
    const std::string dataset_file_name = argv[1];
    if (rank == 0) {
        // only rank 0 loads da graph so we don't have contention over the file
        std::cout << "Loading graph from: " << dataset_file_name << std::endl;
        try {
            g = load_graph(dataset_file_name, true);
            std::cout << "n=" << g.num_nodes << " total_num_edges=" << g.row_ptr[g.num_nodes] << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "Error loading the graph??" << std::endl;
            return 1;
        }
    }

    MPI_Bcast(&g.num_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD); // send num_nodes
    if (rank != 0) {
        g.row_ptr.resize(g.num_nodes + 1);
    }
    MPI_Bcast(g.row_ptr.data(), g.num_nodes + 1, MPI_INT, 0, MPI_COMM_WORLD); // send row_ptr vector
    int total_num_edges = g.row_ptr[g.num_nodes];
    if (rank != 0) {
        g.col_ind.resize(total_num_edges);
    }
    MPI_Bcast(g.col_ind.data(), total_num_edges, MPI_INT, 0, MPI_COMM_WORLD); // send col_ind vector

    int n = g.num_nodes;
    int source = std::stoi(argv[2]);

    int pr = (int)std::sqrt((double)size); // number of process rows is sqrt(num_processes)
    while (size % pr != 0) { // keep decreasing pr until you get a number that is divisible by the size -> ex. 32 processes sqrt(32)= 5.656 -> 5 -> 5-1=4 -> pr = 4
        pr--;
    }
    int pc = size / pr; // number of processor columns 
    int processor_row = rank / pc; // gets processor row in 2d grid
    int processor_column = rank % pc; // gets processor column in 2d grid

    if (rank == 0)
        std::cout << "Process grid: " << pr << " x " << pc << std::endl;

    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, processor_row, rank, &row_comm); // this groups processors in the same processor row together -> when they try to communicate with MPI_Comm of row_comm, it will only be with other processors communicating through row_comm
    MPI_Comm_split(MPI_COMM_WORLD, processor_column, rank, &col_comm); // same as comment above but for col_comm


    auto compute_offsets = [](int total, int parts, std::vector<int>& offsets) {
        offsets.resize(parts + 1);
        offsets[0] = 0;
        int base = total / parts; // base number of vertices per processor 
        int remainder = total % parts; // num left we need to give away
        for (int i = 0; i < parts; i++) {
            int stripe_size = base;
            if (i < remainder) {
                stripe_size++;
            }
            offsets[i + 1] = offsets[i] + stripe_size;
        }
    };

    std::vector<int> to_vertices_range, from_vertices_range;
    compute_offsets(g.num_nodes, pr, to_vertices_range); // tells us the range of "to" vertices each processor owns [0, 10, 20...]
    compute_offsets(g.num_nodes, pc, from_vertices_range); // tells us the range of "from" vertices each processor owns: [0, 10, 20...]

    int row_start  = to_vertices_range[processor_row]; // gives us this processor's first "to" nodeID
    int total_my_rows = to_vertices_range[processor_row + 1] - row_start; // gives us the number of "to" vertices in a processor's row -> aka how many rows a processor owns
    int col_start  = from_vertices_range[processor_column]; // gives us this processor's first "from" nodeID
    int total_my_columns = from_vertices_range[processor_column + 1] - col_start; // gives us the number of "from" vertices in a processor's column -> aka how many columns a processor owns

    double t0 = MPI_Wtime();

    std::vector<int> local_row_ptr, local_col_ind;
    build_local_submatrix(g.row_ptr, g.col_ind, row_start, total_my_rows, col_start, total_my_columns, local_row_ptr, local_col_ind); // put CSR submatrix for this processor into local_row_ptr and local_col_ind

    // get rid of old g.row_ptr and g.col_ind memory lol
    g.row_ptr.clear();
    g.row_ptr.shrink_to_fit();
    g.col_ind.clear();
    g.col_ind.shrink_to_fit();

    // MPI_Barrier(MPI_COMM_WORLD); // wait for all processes to have completed this for fair bfs timing 

    auto dist = bfs_2d_mpi(
    local_row_ptr, local_col_ind,
    g.num_nodes, source,
    total_my_rows, total_my_columns,
    row_start, col_start,
    MPI_COMM_WORLD, row_comm, col_comm,
    processor_row, processor_column, pr, pc,
    to_vertices_range);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        std::cout << "\n----- 2D BFS -----\n";
        std::cout << "Source node   : " << source << std::endl;
        std::cout << "Nodes visited : " << nodes_visited(dist) << std::endl;
        std::cout << "Time          : " << (t1 - t0) << " sec" << std::endl;
        print_distances(dist, 50);
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
    return 0;
}