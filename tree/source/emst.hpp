#pragma once
#include "puffinn/collection.hpp"
#include "puffinn/hash_source/hash_source.hpp"
#include "puffinn/similarity_measure/cosine.hpp"
#include "puffinn/format/real_vector.hpp"
#include "puffinn/math.hpp"
#include "puffinn/linkcutree.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <unistd.h>
#include <stack>

using EdgeTuple = std::tuple<float, std::pair<unsigned int, unsigned int>>;

namespace puffinn{

    struct Prefix{
        unsigned int i;
        uint_fast32_t j;
        float delta;
    };

    struct DSU {
        std::vector<int> parent, rank;
    
        DSU(int n) : parent(n), rank(n, 0) {
            for (int i = 0; i < n; i++) {
                parent[i] = i;
            }
        }
    
        int find(int x) {
            if (parent[x] != x)
                parent[x] = find(parent[x]); // Path compression
            return parent[x];
        }
    
        bool union_sets(int x, int y) {
            int rootX = find(x);
            int rootY = find(y);
    
            if (rootX == rootY)
                return false; // Cycle detected
    
            // Union by rank
            if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
            return true;
        }
    };
        

    class EMST {
        // Object variables
        uint32_t dimensionality;
        uint64_t memory_limit;
        size_t MAX_REPETITIONS;
        uint32_t MAX_HASHBITS;
        Index<CosineSimilarity, FHTCrossPolytopeHash, SimHash> table;
        std::vector<std::vector<float>> data {};
        std::vector<CollisionEnumerator> segments;
        uint32_t num_data {0};
        const float delta {0.0000001};
        const float epsilon {0.00001};
        DSU dsu_true;
        // Sets for the confimed and the unconfirmed edges
        std::vector<EdgeTuple> Tc;
        std::vector<EdgeTuple> Tu;
        std::vector<std::vector<unsigned int>> neighbors;

        std::vector<EdgeTuple> top;


        public:
            /**
             * @brief Constructs an Euclidean Minimum Spanning Tree (EMST) from input data points
             * 
             * @param dimensionality The number of dimensions in the input vectors
             * @param memory_limit Maximum memory usage allowed for the index structure
             * @param data_in Input data points
             * @param delta Probability of failure parameter (default: 0.01)
             * @param epsilon Approximation factor parameter (default: 0.01)
             *
             * @details This constructor initializes an EMST object by:
             * 1. Setting up the LSH index table with cosine similarity metric
             * 2. Inserting all input vectors into the index
             * 3. Rebuilding the index structure
             * 4. Construct a Union Find data structure
             * 5. 
             * 
             * The constructor takes ownership of the input data through a move operation.
             */
            EMST(uint32_t dimensionality, uint64_t memory_limit, std::vector<std::vector<float>> &data_in, const float delta = 0.01, const float epsilon = 0.01)
                : dimensionality(dimensionality),
                  memory_limit(memory_limit),
                  table(Index<CosineSimilarity,FHTCrossPolytopeHash, SimHash>(dimensionality, memory_limit)),
                  data(data_in),
                  num_data((data_in).size()),
                  delta(delta),
                  epsilon(epsilon),
                  dsu_true(DSU(num_data))
            {
                for (auto vec : data) {
                    table.insert(vec);
                }
                table.rebuild();
                MAX_HASHBITS = table.get_hashbits();
                MAX_REPETITIONS = table.get_repetitions();
                segments = table.order_segments();
                neighbors.resize(num_data);
                dirty_start();
                std::cout << "EMST constructed " << MAX_REPETITIONS <<  " L, K " << MAX_HASHBITS << " num data " << num_data << std::endl;
            };

            /// @brief Destructor
            ~EMST() = default;
            
            /// @brief Computes the exact MST by doing all pairwise comparisons
            /// @return weight of the exact MST
            float exact_tree(){
                // Clear from any previous runs
                clear();
                //Compute all the distances
                std::vector<EdgeTuple> all_edges;
                std::vector<std::vector<EdgeTuple>> local_edges (num_data);
                #pragma omp parallel for
                for (uint32_t i = 0; i < num_data; i++) {
                    for (uint32_t j = i+1; j < num_data; j++) {
                        float dist = table.get_similarity(i, j);
                        local_edges[i].emplace_back(1-dist, std::make_pair(i, j));
                    }
                }
                std::cout << "Computed all pairs" << std::endl;
                // Merge the local edges
                for (const auto& local : local_edges) {
                    all_edges.insert(all_edges.end(), local.begin(), local.end());
                }
                //Sort the edges
                std::sort(all_edges.begin(), all_edges.end());
                //Create the DSU
                DSU dsu(num_data);
                float tree_weight = 0;
                std::cout << "Creating the MST" << std::endl;
                for (const auto& edge : all_edges) {
                    add_edge(edge, dsu, top);
                }
                for (const auto& edge : top) {
                    tree_weight += std::get<0>(edge);
                }
                return tree_weight;
            }
            /// @brief Find the Minimum Spanning Tree using only confirmed edges
            std::vector<std::pair<unsigned int, unsigned int>> find_tree() {
    
                std::vector<std::pair<unsigned int, unsigned int>> tree;
                bool found = false;
                for (int i=MAX_HASHBITS; i>= 0; i--) {
                    if (found) {
                        break;
                    }
                   // std::cout << "Iteration: " << segments[0].i <<" "<< i << std::endl;
                   // #pragma omp parallel for
                    for (size_t j=0; j<MAX_REPETITIONS; j++) {
                        if (found) {
                            continue;
                        }
                        std::vector<EdgeTuple> local_Tu, local_Tc;
                        enumerate_edges(segments[j], local_Tu, local_Tc);

                       // #pragma omp critical
                        {
                            for(auto edge : local_Tc) {
                                // add_edge_nocheck(edge);
                                // Tu.insert(edge);
                                if (std::find(top.begin(), top.end(), edge) != top.end())
                                    continue;
                                if(add_edge(edge, dsu_true, top)) {
                                    Tc.push_back(edge);
                                }
                            }
                            
                            // In this case Tu is just a waiting room, they are not part of the confirmed tree
                            for(auto edge : local_Tu) {
                                Tu.push_back(edge);
                            }

                            // Move all the confirmed edges in Tu to Tc
                            std::vector<EdgeTuple> edges_to_move;
                            // Sort Tu by weight
                            std::sort(Tu.begin(), Tu.end());
                            std::cout << "Tu sorted"  << std::endl;
                            int igh = 0;
                            for(const auto& edge : Tu) {
                               // std::cout << "probability " << table.get_probability(i, j, std::get<0>(edge)) << std::endl;
                               float prob = table.get_probability(i, j, (1-std::get<0>(edge)));
                               std::cout << "Weight: " << std::get<0>(edge) << " Probability: " << prob << igh<< std::endl;
                               igh++;
                                if(prob <= 1-delta) {
                                    if(add_edge(edge, dsu_true, top)) {
                                        Tc.push_back(edge);
                                        edges_to_move.push_back(edge);
                                        std::cout << "moved" << std::endl;
                                    }
                                }
                                else {// The edges are sorted by weight, so if one doesn't satisfy
                                     // the probability then the rest can't
                                     std::cout << "Skip";
                                    break;
                                }
                            }
                            for(auto edge : edges_to_move) {
                                Tu.erase(std::remove(Tu.begin(), Tu.end(), edge), Tu.end());
                            }

                        }
                    
                        //std::cout << "top size: " << top.size() << std::endl;
                        //If we have num_data -1 edges compute the spanning tree weight
                        if(top.size()==num_data-1) {
                            //if (is_connected()) {
                            // std::cout << "Connected" << std::endl;}
                            float tree_weight = 0;
                            for (const auto& edge : top) {
                                tree_weight += std::get<0>(edge);
                            }
                            std::cout << "Tree weight: " << tree_weight << std::endl;
                            found = true;
                            return tree;
                            }
                        }
                    
                    //std::cout << "merging segments" << std::endl;
                    table.merge_segments(segments);
                }
                    //std::cout << "merged segments" << std::endl;
                
                return tree;

            }


            /// @brief Find the ɛ-EMST
            std::vector<std::pair<unsigned int, unsigned int>> find_epsilon_tree() {
    
                std::vector<std::pair<unsigned int, unsigned int>> tree;
                bool found = false;
                for (int i=MAX_HASHBITS; i>= 0; i--) {
                    if (found) {
                        break;
                    }
                   // std::cout << "Iteration: " << segments[0].i <<" "<< i << std::endl;
                   // #pragma omp parallel for
                    for (size_t j=0; j<MAX_REPETITIONS; j++) {
                        if (found) {
                            continue;
                        }
                        std::vector<EdgeTuple> local_Tu, local_Tc;
                        // std::cout << "Enumerating edges for prefix: " << i << " " << j << std::endl;
                        enumerate_edges(segments[j], local_Tu, local_Tc);
                        std::cout << "Tu_l size: " << local_Tu.size() << " Tu size: " << Tu.size() << " Tc size: " << local_Tc.size() << "Top size: " << top.size() << std::endl;

                       // #pragma omp critical
                        {
                            for(auto edge : local_Tc) {
                                auto first_edge = std::get<1>(edge).first;
                                auto second_edge = std::get<1>(edge).second;
                                if (std::find(neighbors [first_edge].begin(), neighbors[first_edge].end(), second_edge) != neighbors[first_edge].end())
                                    continue;
                                if(add_edge(edge, dsu_true, top)) {
                                    Tc.push_back(edge);
                                }
                            }

                            //Just add them to the waiting room
                            for(auto edge : local_Tu) {
                                    Tu.push_back(edge);
                                }

                            // Move all the confirmed edges in Tu to Tc
                            std::vector<EdgeTuple> edges_to_move;
                            // Sort Tu by weight
                            std::sort(Tu.begin(), Tu.end());

                            for(auto& edge : Tu) {
                               // std::cout << "probability " << table.get_probability(i, j, std::get<0>(edge)) << std::endl;
                                if(table.get_probability(i, j, (1-std::get<0>(edge))) <= 1-delta) {                                  
                                    Tc.push_back(edge);
                                    edges_to_move.push_back(edge);
                                }
                                else {// The edges are sorted by weight, so if one doesn't satisfy
                                     // the probability then the rest can't
                                    break;
                                }
                            }
                            for(auto edge : edges_to_move) {
                                Tu.erase(std::remove(Tu.begin(), Tu.end(), edge), Tu.end());
                            }

                        }
                        if (top.size() == num_data - 1) {
                            std::tie(tree, found) = fill_tree(dsu_true);
                            if (found) {
                                is_connected(tree);
                                return tree;
                            }
                        }
                    }
                    //Fill the tree
                    std::tie(tree, found) = fill_tree(dsu_true);
                    if (found) {
                        is_connected(tree);
                        return tree;
                    }
                    table.merge_segments(segments);
                }
                return tree;
            };

        //*** Private methods */
        private:

            
            void enumerate_edges(CollisionEnumerator st, std::vector<EdgeTuple>& Tu_local, std::vector<EdgeTuple>& Tc_local) {
                // Discover edges that share the same prefix at iteration st.i, st.j
                std::vector<EdgeTuple> couples = table.all_close_pairs(st);
                // Evaluate all pair distances
                for (auto couple : couples) {
                    if(std::get<1>(couple).first < std::get<1>(couple).second) {
                        std::swap(std::get<1>(couple).first, std::get<1>(couple).second);
                    }
                    // If the distance is less than the threshold, add it to the confirmed edges
                    if (table.get_probability(st.i, st.j, 1-std::get<0>(couple))  <= 1 - delta) {
                        Tc_local.emplace_back(couple);
                    }
                    // Otherwise, add it to the unconfirmed edges
                    else{
                        Tu_local.emplace_back(couple);
                    }
                }
                return;
            };

            /// @brief Return the bound weight (1+ɛ)(sum over Tc + |Tu|*max(Tu) )
            /// @param Tu set of unconfirmed edges 
            /// @param Tc set of confirmed edges
            /// @return the weight
            float bound_weight(std::vector<EdgeTuple> Tu, std::vector<EdgeTuple> Tc) {
                float weight = 0;
                if (Tc.size() == 0) {
                    std::cout << "No confirmed edges" << std::endl;
                    return 0;
                }
                for (const auto& edge : Tc) {
                    weight += std::get<0>(edge);
                }
                float max_confirmed = std::get<0>(*std::max_element(Tc.begin(), Tc.end()));
                std::cout << "Max confirmed: " << max_confirmed << std::endl;
                std::cout << "Current weight: " << weight << std::endl;
                weight += max_confirmed * (data.size()-1-Tc.size());

                return weight;
            }

            bool is_connected(std::vector<std::pair<unsigned int, unsigned int>> tree) {
                // Check if the tree is connected
                std::vector<std::pair<unsigned int, unsigned int>> edges = tree;
                std::vector<bool> visited(num_data, false);
                std::vector<std::vector<unsigned int>> adj_list(num_data);
                for (const auto& edge : edges) {
                    adj_list[edge.first].push_back(edge.second);
                    adj_list[edge.second].push_back(edge.first);
                }
                std::vector<unsigned int> stack;
                stack.push_back(0);
                visited[0] = true;
                while (!stack.empty()) {
                    unsigned int node = stack.back();
                    stack.pop_back();
                    for (const auto& neighbor : adj_list[node]) {
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;
                            stack.push_back(neighbor);
                        }
                    }
                }
                // for (const auto& edge : tree) {
                //     std::cout << edge.first << " " << edge.second << std::endl;
                // }

                for (const auto& v : visited) {
                    if (!v) {
                        std::cout << "Not connected" << std::endl;
                        return false;
                    }
                }
                std::cout << "Connected" << std::endl;

                return true;
            };
            
            bool add_edge(const EdgeTuple& new_edge_input, DSU &dsu, std::vector<EdgeTuple>& edge_list) {
                //std::sort(top.begin(), top.end());
                // Extract new edge and its weight.
                std::pair<unsigned int, unsigned int> new_edge = std::get<1>(new_edge_input);
                float new_weight = std::get<0>(new_edge_input);
            
                // Try to add new edge normally.
                if (dsu.union_sets(new_edge.first, new_edge.second)) {
                    edge_list.push_back(new_edge_input);
                    neighbors[new_edge.first].push_back(new_edge.second);
                    neighbors[new_edge.second].push_back(new_edge.first);
                    return true;
                }
                
                // Otherwise, adding new_edge forms a cycle.
                // Use DFS to get the unique path between new_edge.first and new_edge.second
                std::vector<EdgeTuple> cycleEdges;
                dfs_cycle_discover(new_edge.first, new_edge.second, cycleEdges);
                if (cycleEdges.empty()) {
                    // Should not happen in a spanning tree.
                    return false;
                }
                
                // Find the heaviest edge along the cycle.
                std::sort(cycleEdges.begin(), cycleEdges.end());
                EdgeTuple maxCycleTuple = *cycleEdges.rbegin();
                float maxCycleWeight = std::get<0>(maxCycleTuple);
                
                // If the new edge is not strictly better than the heaviest edge on the cycle, do nothing.
                if (new_weight >= maxCycleWeight) {
                    return false;
                }
                
                // Remove the heaviest edge from the spanning tree.
                std::pair<unsigned int, unsigned int> maxCycleEdge = std::get<1>(maxCycleTuple);
                auto it = std::find(edge_list.begin(), edge_list.end(), maxCycleTuple);
                if (it != edge_list.end()) {
                    edge_list.erase(it);
                } else {
                    return false;
                }
                
                // Remove the heavy edge from the neighbors.
                auto remove_neighbor = [&](unsigned int u, unsigned int v) {
                    neighbors[u].erase(std::remove(neighbors[u].begin(), neighbors[u].end(), v), neighbors[u].end());
                };
                remove_neighbor(maxCycleEdge.first, maxCycleEdge.second);
                remove_neighbor(maxCycleEdge.second, maxCycleEdge.first);
                
                // Insert the new edge.
                edge_list.push_back(new_edge_input);
                neighbors[new_edge.first].push_back(new_edge.second);
                neighbors[new_edge.second].push_back(new_edge.first);
                
                // Rebuild the DSU from the updated tree.
                dsu = DSU(num_data);
                for (const auto &edge : edge_list) {
                    std::pair<unsigned int, unsigned int> e = std::get<1>(edge);
                    dsu.union_sets(e.first, e.second);
                }
                
                return true;
            }
            
            std::pair<std::vector<std::pair<unsigned int, unsigned int>>, bool> fill_tree(DSU dsu_copy) {
                std::vector<std::pair<unsigned int, unsigned int>> tree;
                std::vector<EdgeTuple> top_copy (top);               

                bool found = false;
                if (Tc.size() == 0) {
                    return {tree, found};
                }

                for (const auto& edge : Tu) {
                    // Finish the tree with the best unconfirmed edges
                    if (top_copy.size() == num_data - 1) {
                        break;
                    }
                    add_edge(edge, dsu_copy, top_copy);
                }

                if (top_copy.size() != num_data - 1) {
                    return {tree, found};
                }

                float tree_weight = 0;
                for (const auto& edge : top_copy) {
                    tree_weight += std::get<0>(edge);
                }
                float bounded_weigth = bound_weight(Tu, Tc);
                std::cout << "Tree weight: " << tree_weight << " Bounded weight: " << bounded_weigth << std::endl;
                // If less than (1+ɛ)(sum over Tc + |Tu|*max(Tu) ) we return, else we continue
                if (tree_weight <= (1+epsilon)*bounded_weigth) {
                    for (const auto& edge : top_copy) {
                        tree.push_back(std::get<1>(edge));
                    }

                    found = true;
                    top = top_copy;
                    return {tree, found};
                }
                return {tree, found};
            }

            /// @brief Generate a random spanning tree to have an initial solution
            void dirty_start() {
                std::vector<unsigned int> vertices(num_data);
                std::iota(vertices.begin(), vertices.end(), 0);
                std::random_shuffle(vertices.begin(), vertices.end());
                for (size_t i = 1; i < vertices.size(); i++) {
                    Tu.emplace_back(1-table.get_similarity(vertices[i-1], vertices[i]), std::make_pair(vertices[i-1], vertices[i]));
                    //top.emplace_back(1-table.get_similarity(vertices[i-1], vertices[i]), std::make_pair(vertices[i-1], vertices[i]));
                    //dsu.union_sets(vertices[i-1], vertices[i]);
                }
            }

            /// @brief Clear the data structures from previous runs
            void clear() {
                Tc.clear();
                Tu.clear();
                top.clear();
            }

            /// @brief Find the path that connects source to find
            /// @param source, the source node
            /// @param find, the node to find
            /// @param cycleEdges, the edges of the path
            void dfs_cycle_discover(unsigned int source, unsigned int find, std::vector<EdgeTuple> &cycleEdges) {
                return;
                std::vector<unsigned int> s;
                std::vector<char> visited(num_data, 0);
                std::vector<int> parent(num_data, -1);  // Track parent nodes for path reconstruction
                bool found = false;
            
                s.push_back(source);
                visited[source] = 1;
            
                while (!s.empty()) {
                    unsigned int current = s.back();
                    s.pop_back();
            
                    // If we found the target, break out
                    if (current == find) {
                        found = true;
                        break;
                    }
            
                    for (auto& neighbor : neighbors[current]) {
                        if (!visited[neighbor]) {
                            visited[neighbor] = 1;
                            parent[neighbor] = current; // Track the path
                            s.push_back(neighbor);
                        }
                    }
                }
            
                // If the path is found, reconstruct it
                if (found) {
                    std::vector<unsigned int> path;
                    for (int node = find; node != -1; node = parent[node]) {
                        path.push_back(node);
                    }
                    std::reverse(path.begin(), path.end()); // Reverse to get source -> target order
            
                    // Store the path in cycleEdges
                    for (size_t i = 0; i < path.size() - 1; i++) {
                        cycleEdges.emplace_back(
                            table.get_similarity(path[i], path[i + 1]),
                            std::make_pair(path[i], path[i + 1])
                        );
                       // std::cout << path[i] << " -> " << path[i + 1] << " ";
                    }
                } 
            }
            
    };  //closes class
}       //closes namespace
