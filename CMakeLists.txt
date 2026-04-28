cmake_minimum_required(VERSION 3.14)
project(sssp_bfs)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Optimization flag to match your manual compile commands
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# Serial Dijkstra
add_executable(dj_serial dj/serial-dj.cpp)
target_include_directories(dj_serial PRIVATE dj/)

# Parallel delta-stepping
add_executable(dj_parallel dj/parallel-dj.cpp)
target_include_directories(dj_parallel PRIVATE dj/)

# Serial BFS
add_executable(bfs_serial bfs/serial-bfs.cpp)
target_include_directories(bfs_serial PRIVATE bfs/)

# Parallel BFS
add_executable(bfs_parallel bfs/parallel-bfs.cpp)
target_include_directories(bfs_parallel PRIVATE bfs/)