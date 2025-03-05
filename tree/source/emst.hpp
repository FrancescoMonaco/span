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

using EdgeTuple = std::tuple<float, std::pair<uint32_t, uint32_t>>;

namespace puffinn{

    struct Prefix{
        uint32_t i;
        uint32_t j;
        float delta;
    };

    /// @brief Disjoint Set Union data structure, implemented with path compression and union by rank.
    struct DSU {
        std::vector<uint32_t> parent, rank;
    
        /// @brief Create a Union Find data structure
        /// @param n number of elements
        DSU(uint32_t n) : parent(n), rank(n, 0) {
            for (uint32_t i = 0; i < n; i++) {
                parent[i] = i;
            }
        }
    
        /// @brief Return the parent of the set containing x
        /// @param x, the element to find
        /// @return the parent of the set containing x
        uint32_t find(uint32_t x) {
            if (parent[x] != x)
                parent[x] = find(parent[x]); // Path compression
            return parent[x];
        }
    
        /// @brief Merge the sets containing x and y in time O(ɑ(n))
        /// @param x first element 
        /// @param y second element
        /// @return true if the sets containing x and y were disjoint and were successfully merged, false otherwise
        bool union_sets(uint32_t x, uint32_t y) {
            uint32_t rootX = find(x);
            uint32_t rootY = find(y);
    
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
        const float delta {0.01};
        const float epsilon {0.01};
        DSU dsu_true;
        // Sets for the confimed and the unconfirmed edges
        std::vector<EdgeTuple> Tc;
        std::vector<EdgeTuple> Tu;
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
            EMST(uint32_t dimensionality, uint64_t memory_limit, std::vector<std::vector<float>> &data_in, const float delta = 0.2, const float epsilon = 0.2)
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
                dirty_start();
                std::cout << "Index constructed " << MAX_REPETITIONS <<  " L, K " << MAX_HASHBITS << " num data " << num_data << std::endl;
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
                std::vector<std::vector<EdgeTuple>> local_edges (num_data);
                std::vector<std::vector<EdgeTuple>> local_Tus (num_data);

                bool found = false;
                float max_confirmed = 0;
                std::vector<EdgeTuple> edges;
                for (int i=MAX_HASHBITS; i>= 0; i--) {
                    if (found) {
                        break;
                    }
                    #pragma omp parallel for
                    for (size_t j=0; j<MAX_REPETITIONS; j++) {
                        if (found) {
                            continue;
                        }
                        DSU local_dsu(num_data);
                        std::vector<EdgeTuple>local_Tc, local_top;
                        enumerate_edges(segments[j], local_Tus[j], local_Tc);    


                        std::sort(local_Tc.begin(), local_Tc.end());
                        for(auto edge : local_Tc) {
                            if (std::get<float>(edge) > max_confirmed) {
                                max_confirmed = std::get<float>(edge);
                            }
                            add_edge(edge, local_dsu, local_top);
                        }
                        local_edges[j].insert(local_edges[j].end(), local_top.begin(), local_top.end());
                        


                        #pragma omp critical
                        {
                        // Every x iterations we have a batch, construct the MST from these edges
                        if (((int)MAX_REPETITIONS/2)%(j+1) == 0) {   
                            //Move the top edges in with the new edges
                            edges.insert(edges.end(), top.begin(), top.end());
                            top.clear();
                            dsu_true = DSU(num_data);
                            for (const auto& local : local_edges) {
                                edges.insert(edges.end(), local.begin(), local.end());
                            }

                            std::sort(edges.begin(), edges.end());
                            for (const auto& edge : edges) {
                                add_edge(edge, dsu_true, top);
                                if (top.size() == num_data - 1) {
                                    break;
                                }
                            }
                            //std::cout << "Top size: " << top.size() << std::endl;
                            if (top.size() == num_data - 1) {
                                float tree_weight = 0;
                                for (const auto& edge : top) {
                                    tree_weight += std::get<0>(edge);
                                }                           
                                std::cout << "Tree weight: " << tree_weight << std::endl;
                                found = true;
                                // Fill the tree
                                for (const auto& edge : top) {
                                    tree.push_back(std::get<1>(edge));
                                }
                            }
                            // Lose the unused edges
                            edges.clear();
                        }
                        }
                    }
                    // Move to the next prefix
                    table.merge_segments(segments);
                }
                is_connected(tree);
                return tree;

            }

            /// @brief Find the ɛ-EMST
            std::vector<std::pair<unsigned int, unsigned int>> find_epsilon_tree() {
                // Variables to store the tree and the edges
                std::vector<std::pair<unsigned int, unsigned int>> tree;
                bool found = false;
                std::vector<EdgeTuple> edges;

                for (int i=MAX_HASHBITS; i>= 0; i--) {
                    if (found) {
                        break;
                    }
                   #pragma omp parallel for
                    for (size_t j=0; j<MAX_REPETITIONS; j++) {
                        if (found) {
                            continue;
                        }
                        // Local variables
                        std::vector<EdgeTuple> local_Tu, local_Tc, local_top;
                        DSU local_dsu(num_data);
                        enumerate_edges(segments[j], local_Tu, local_Tc);
                        // Create a local spanning tree with the confirmed edges
                        for(const auto& edge : local_Tc) {
                                add_edge(edge, local_dsu, local_top);
                        }
                        // Fill the spanning tree with the unconfirmed edges
                        for(const auto& edge : local_Tu) {
                            if(local_top.size() == num_data - 1) {
                                break;
                            }
                            add_edge(edge, local_dsu, local_top);
                        }

                        // The edges of the local spanning tree are added to the global edges
                        #pragma omp critical
                        {
                        edges.insert(edges.end(), 
                                        std::make_move_iterator(local_top.begin()),
                                        std::make_move_iterator(local_top.end()));
                        }


                        // Every ẋ iterations we have a batch, construct the MST from the global edges
                        if (((int)MAX_REPETITIONS/2)%(j+1)==0) {
                            #pragma omp critical
                            {
                            // Move the top edges in with the new edges, in this way we keep the last spanning tree
                            // But we allow for a better solution
                            edges.insert(edges.end(),
                                        std::make_move_iterator(top.begin()),
                                        std::make_move_iterator(top.end()));
                            top.clear();
                            dsu_true = DSU(num_data);
                            // This is just Kruksal's algorithm, top stores our global spanning tree
                            std::sort(edges.begin(), edges.end());
                            for (const auto& edge : edges) {
                                add_edge(edge, dsu_true, top);
                                if (top.size() == num_data - 1) {
                                    break;
                                }
                            }
                            edges.clear();
                            // If we have a tree, we check if it is a valid ɛ-EMST
                            if (top.size() == num_data - 1) {
                                float tree_weight = 0;
                                for (const auto& edge : top) {
                                    tree_weight += std::get<0>(edge);
                                }  

                                if(tree_weight <= (1+epsilon)*bound_weight(top, i, j)) {                       
                                    // Fill the tree
                                    for (const auto& edge : top) {
                                        tree.push_back(std::get<1>(edge));
                                    }
                                    found = true;
                                }
                                else {
                                    found = false;
                                }
                         
                                std::cout << "Tree weight: " << tree_weight << " Bound weight: " << (1+epsilon)*bound_weight(top, i, j) << std::endl;
                            }
                            // Lose the unused edges
                            //edges.clear();

                            }
                        }
                    }
                    // Move to the next prefix
                    table.merge_segments(segments);
                }
                // Check for connectivity
                is_connected(tree);
                return tree;
            };

        //*** Private methods */
        private:

            /// @brief Obtain the couples of nodes that share the same prefix from the hash table
            ///        and split them into edges whose recall is above the threshold and the others
            /// @param st CollisionEnumerator object that stores the repetition and the prefix
            /// @param Tu_local vector that stores the unconfirmed edges
            /// @param Tc_local vector that stores the confirmed edges
            void enumerate_edges(CollisionEnumerator st, std::vector<EdgeTuple>& Tu_local, std::vector<EdgeTuple>& Tc_local) {
                // Discover edges that share the same prefix at iteration st.i, st.j
                std::vector<EdgeTuple> couples = table.all_close_pairs(st);
                // Evaluate all pair distances
                for (auto couple : couples) {
                    // If the distance is less than the threshold, add it to the confirmed edges
                    if (table.get_probability(st.i, st.j, 1-std::get<0>(couple))  <  delta) {
                        Tc_local.emplace_back(couple);
                    }
                    // Otherwise, add it to the unconfirmed edges
                    else{
                        if (Tu_local.size() > num_data*10) {
                            // Resize the vector
                            continue;
                        }
                        Tu_local.emplace_back(couple);
                    }
                }
                std::sort(Tc_local.begin(), Tc_local.end());
                std::sort(Tu_local.begin(), Tu_local.end());
                return;
            };

            /// @brief Return the bound weight (1+ɛ)(sum over Tc + |Tu|*max(Tu) )
            /// @param top_copy a vector that contains the edges in the spanning tree
            /// @return the weight
            float bound_weight(std::vector<EdgeTuple>& top_copy, int i, int j) {
                float weight = 0;
                float max_confirmed = 0;
                int unconfirmed = 0;
                for (const auto& edge : top_copy) {
                    auto edge_weight = std::get<0>(edge);  
                    auto probability = table.get_probability(i, j, 1-edge_weight);
                    if (probability < delta) {
                        weight += edge_weight;
                        if (edge_weight > max_confirmed) {
                            max_confirmed = edge_weight;
                        }
                    }
                    else{
                        unconfirmed++;
                    }
                }
                weight += max_confirmed * unconfirmed;
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

                if (!std::accumulate(visited.begin(), visited.end(),true, std::logical_and<bool>())){
                    std::cout << "Not connected" << std::endl;
                    return false;
                }
                std::cout << "Connected" << std::endl;

                return true;
            };
            
            bool add_edge(const EdgeTuple& new_edge_input, DSU &dsu, std::vector<EdgeTuple>& edge_list) {
                // Extract new edge and its weight.
                std::pair<unsigned int, unsigned int> new_edge = std::get<1>(new_edge_input);
            
                // Try to add new edge normally.
                if (dsu.union_sets(new_edge.first, new_edge.second)) {
                    edge_list.push_back(new_edge_input);
                    return true;
                }
                return false;
            }

            /// @brief Generate a random spanning tree to have an initial solution
            void dirty_start() {
                std::vector<unsigned int> vertices(num_data);
                std::iota(vertices.begin(), vertices.end(), 0);
                std::random_shuffle(vertices.begin(), vertices.end());
                for (size_t i = 1; i < vertices.size(); i++) {
                    Tu.emplace_back(1-table.get_similarity(vertices[i-1], vertices[i]), std::make_pair(vertices[i-1], vertices[i]));
                }
            }

            /// @brief Clear the data structures from previous runs
            void clear() {
                top.clear();
            }
            
    };  //closes class
}       //closes namespace
