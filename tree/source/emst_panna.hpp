#pragma once
#include "panna/trieindex.hpp"
#include "panna/linalg.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <unistd.h>
#include <stack>
#include <set>
using EdgeTuple = std::tuple<float, std::pair<uint32_t, uint32_t>>;

namespace panna{

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
        
    template <typename Dataset, typename Hasher, typename Distance>
    class EMST {
        // Object variables
        uint32_t dimensionality;
        uint64_t memory_limit;
        size_t MAX_REPETITIONS;
        uint32_t MAX_HASHBITS;
        Index<Dataset, Hasher, Distance> table;
        std::vector<std::vector<float>> data {};
        uint32_t num_data {0};
        float delta {0.01};
        const float epsilon {0.01};
        DSU dsu_true;
        // Sets for the confimed and the unconfirmed edges
        std::vector<EdgeTuple> top;
        std::vector<std::vector<EdgeTuple>> local_edges;
        std::vector<float> probabilities;


        public:
            /**
             * @brief Class to construct an approximate Euclidean Mininmum Spanning Tree from data points
             * 
             * @param dimensions Dimension of the hash index
             * @param repetitions Number of repetitions for the LSH index
             * @param builder Builder for the hash function
             * @param data_in Input data points
             * @param data_dimensionality Dimensionality of the input data
             * @param delta Probability of failure parameter (default: 0.01)
             * @param epsilon Approximation factor parameter (default: 0.01)
             *
             * @details This constructor initializes an EMST object by:
             * 1. Setting up the LSH index table with cosine similarity metric
             * 2. Inserting all input vectors into the index
             * 3. Rebuilding the index structure
             * 4. Construct a Union Find data structure
             * The constructor takes ownership of the input data through a move operation.
             */
            EMST(size_t dimensions, size_t repetitions, typename Hasher::Builder builder, std::vector<std::vector<float>> &data_in, size_t data_dimensionality, const float delta = 0.01, const float epsilon = 0.01)
                : dimensionality(data_dimensionality),
                  memory_limit(memory_limit),
                  table(Index<Dataset, Hasher, Distance>(dimensions, builder, repetitions)),
                  data(data_in),
                  num_data((data_in).size()),
                  delta(delta),
                  epsilon(epsilon),
                  dsu_true(DSU(num_data))   {
                
                // Insert the data
                for ( auto& point: data_in ) {
                    normalize( point );
                    table.insert( point.begin(), point.end() );
                }
                table.rebuild();

                // Get info on the index
                MAX_HASHBITS = table.num_concatenations();
                MAX_REPETITIONS = table.num_repetitions();

                local_edges.resize(MAX_REPETITIONS);
                probabilities.resize(MAX_REPETITIONS, 1);
                //dirty_start(local_Tus[0]);
                std::cout << "Index constructed, L: " << MAX_REPETITIONS <<  " K: " << MAX_HASHBITS << " num data: " << num_data << std::endl;
            };

            /// @brief Destructor
            ~EMST() = default;
            
            /// @brief Computes the exact MST with Kruskal's algorithm in a naive way
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
                        float dist = table.get_distance( i, j );
                        local_edges[i].emplace_back(dist, std::make_pair(i, j));
                    }
                }
                std::cout << "Computed all pairs" << std::endl;
                // Merge the local edges
                for (const auto& local : local_edges) {
                    all_edges.insert(all_edges.end(), local.begin(), local.end());
                }
                std::cout << "Number of edges: " << all_edges.size() << std::endl;
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
                std::vector<EdgeTuple> edges;
                for (int i=MAX_HASHBITS; i> 0; i--) {
                    if (found) {
                        break;
                    }
                    //#pragma omp parallel for
                    for (size_t j=0; j<MAX_REPETITIONS; j++) {
                        if (found) {
                            continue;
                        }
                        DSU local_dsu(num_data);
                        std::vector<EdgeTuple> local_Tc, local_top, local_Tu;
                        enumerate_edges(i, j, local_Tu, local_Tc);   
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
                        local_edges[j].insert(local_edges[j].end(),
                                                std::make_move_iterator(local_top.begin()),
                                                std::make_move_iterator(local_top.end()));
                        local_top.clear();
                        
                        //#pragma omp critical
                        {
                        // Every x iterations we have a batch, construct the MST from these edges
                        if ((j+1)%((int)MAX_REPETITIONS/4) == 0 && !found) { 
                            //Move the top edges in with the new edges
                            edges.insert(edges.end(), 
                                         std::make_move_iterator(top.begin()),
                                         std::make_move_iterator(top.end()));
                            top.clear();

                            dsu_true = DSU(num_data);
                            for (auto& local : local_edges) {
                                edges.insert(edges.end(), std::make_move_iterator(local.begin()), std::make_move_iterator(local.end()));
                                local.clear();
                            }
                            if (!(edges.size() < num_data -1)) {

                                std::sort(edges.begin(), edges.end());
                                for (const auto& edge : edges) {
                                    add_edge(edge, dsu_true, top);
                                    if (top.size() == num_data - 1) {
                                        break;
                                    }
                                }
                                std::cout << "Tree size: " << top.size() << std::endl;
                                // Print also the current weight
                                float tree_weight = 0;
                                for (const auto& edge : top) {
                                    tree_weight += std::get<0>(edge);
                                }
                                std::cout << "Weight: " << tree_weight << std::endl;
                                if (top.size() == num_data - 1) {
                                    //Check that all edges are confirmed by the probability
                                    bool valid = true;
                                    for (const auto& edge : top) {
                                        auto probability = table.fail_probability( std::get<float>(edge), i, j+1 );
                                        if (probability > delta) {
                                            valid = false;
                                            std::cout << "Not confirmed " << probability << " at distance " << std::get<float>(edge) << std::endl;
                                            break;
                                        }
                                    }
                                    if(valid) {
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
                                }
                                // Lose the unused edges
                                edges.clear();
                            }
                        }
                        }
                    }
                    std::cout << "Finished prefix " << i << std::endl;
                }
                is_connected(tree);
                return tree;

            }

            /// @brief Find the ɛ-EMST using both confirmed and unconfirmed edges
            std::vector<std::pair<unsigned int, unsigned int>> find_epsilon_tree() {
                // Variables to store the tree and the edges
                std::vector<std::pair<unsigned int, unsigned int>> tree;
                bool found = false;
                std::vector<EdgeTuple> edges;

                for (int i=MAX_HASHBITS; i>= 0; i--) {
                    if (found) {
                        break;
                    }
                   //#pragma omp parallel for
                    for (size_t j=0; j<MAX_REPETITIONS; j++) {
                        if (found) {
                            continue;
                        }
                        // Local variables
                        std::vector<EdgeTuple> local_top, local_Tu, local_Tc;
                        DSU local_dsu(num_data);
                        enumerate_edges(i, j, local_Tu, local_Tc);
                        
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
                        local_edges[j].insert(local_edges[j].end(), 
                                        std::make_move_iterator(local_top.begin()),
                                        std::make_move_iterator(local_top.end()));
                        local_top.clear();

                       // #pragma omp critical
                        {
                        // Every ẋ iterations we have a batch, construct the MST from the global edges
                        if ((j+1)%((int)MAX_REPETITIONS/4) == 0 && !found) {
                            // Move the top edges in with the new edges, in this way we keep the last spanning tree
                            // But we allow for a better solution
                            edges.insert(edges.end(), std::make_move_iterator(top.begin()), std::make_move_iterator(top.end()));                           
                            top.clear();

                            for (auto& local : local_edges) {
                                edges.insert(edges.end(), std::make_move_iterator(local.begin()), std::make_move_iterator(local.end()));
                                local.clear();
                            }
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
                            probabilities = std::vector<float>(MAX_REPETITIONS, std::get<float>(top[(int)top.size()/2]));
                            // If we have a tree, we check if it is a valid ɛ-EMST
                            if (top.size() == num_data - 1) {
                                float tree_weight = 0;
                                for (const auto& edge : top) {
                                    tree_weight += std::get<0>(edge);
                                }  
                                float bound_w = (1+epsilon)*bound_weight(top, i, j);
                                if(tree_weight <= bound_w) {                       
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
                        }
                        }
                    }
                    // Move to the next prefix
                    std::cout << "Prefix " << i << " completed" << std::endl;
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
            void enumerate_edges(size_t i, size_t j, std::vector<EdgeTuple>& Tu_local, std::vector<EdgeTuple>& Tc_local) {
                // Discover edges that share the same prefix at iteration i, j
                std::vector<EdgeTuple> couples;
                table.search_pairs(j, i, couples);
                std::cout << "Size couples: " << couples.size() << std::endl;

                // Find the edges that are confirmed and the ones that are not, the edges are ordered by weight so we can binary search the splitting point
                // We compute the probability using collision_probability(distance) of each edge, and find all the edges that are above the threshold delta
                auto it = std::partition_point( couples.begin(), couples.end(), [&] (const auto& e) { 
                    auto failure_prob = table.fail_probability( std::get<float> (e), j, i );
                    return failure_prob <= delta;
                } );
                Tu_local.insert(Tu_local.end(), it, couples.end());
                couples.erase(it, couples.end());
                Tc_local.insert(Tc_local.end(), couples.begin(), couples.end());
                std::cout << "Size Tu: " << Tu_local.size() << " Size Tc: " << Tc_local.size() << std::endl;


                return;
            };

            /// @brief Return the bound weight (1+ɛ)(sum over Tc + |Tu|*max(Tu) )
            /// @param top_copy a vector that contains the edges in the spanning tree
            /// @return the weight
            float bound_weight(std::vector<EdgeTuple>& top_copy, int i, int j) {
                float weight = 0;
                float max_confirmed = 0;
                int unconfirmed = 0;
                // Add the weight of the confirmed edges, keep track of the max confirmed and count the unconfirmed ones
                for (const auto& edge : top_copy) {
                    auto edge_weight = std::get<0>(edge);  
                    auto probability = table.fail_probability( edge_weight, i, j+1 );
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

            /// @brief Checks wheter a tree is connected
            /// @param tree the tree that we want to check
            /// @return true if all edge are connected, false otherwise.
            bool is_connected(std::vector<std::pair<unsigned int, unsigned int>>& tree) {
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
            
            /// @brief Add the edge to the tree if it does not create a cycle using the DSU data structure
            /// @param new_edge_input the edge that we have to add
            /// @param dsu the data structure that keeps track of the connected components
            /// @param edge_list the current edges in the tree
            /// @return true if an edge has been added to the edge_list and the DSU data structure, false otherwise
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
            void dirty_start(std::vector<EdgeTuple>& clean) {
                std::vector<unsigned int> vertices(num_data);
                std::iota(vertices.begin(), vertices.end(), 0);
                std::random_shuffle(vertices.begin(), vertices.end());
                for (size_t i = 1; i < vertices.size(); i++) {
                    clean.emplace_back(Distance::compute(vertices[i-1], vertices[i]) , std::make_pair(vertices[i-1], vertices[i]));
                }
                std::sort(clean.begin(), clean.end());
            }

            /// @brief Clear the data structures from previous runs
            void clear() {
                top.clear();
                for (auto& local : local_edges) {
                    local.clear();
                }
            }
            
    };  //closes class
}       //closes namespace
