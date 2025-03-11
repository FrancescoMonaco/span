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
        Index<CosineSimilarity, FHTCrossPolytopeHash, FHTCrossPolytopeHash> table;
        std::vector<std::vector<float>> data {};
        std::vector<CollisionEnumerator> segments;
        uint32_t num_data {0};
        float delta {0.01};
        const float epsilon {0.01};
        DSU dsu_true;
        // Sets for the confimed and the unconfirmed edges
        std::vector<EdgeTuple> top;
        std::vector<std::vector<EdgeTuple>> local_edges;


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
             * The constructor takes ownership of the input data through a move operation.
             */
            EMST(uint32_t dimensionality, uint64_t memory_limit, std::vector<std::vector<float>> &data_in, const float delta = 0.01, const float epsilon = 0.01)
                : dimensionality(dimensionality),
                  memory_limit(memory_limit),
                  table(Index<CosineSimilarity,FHTCrossPolytopeHash, FHTCrossPolytopeHash>(dimensionality, memory_limit)),
                  data(data_in),
                  num_data((data_in).size()),
                  delta(delta),
                  epsilon(epsilon),
                  dsu_true(DSU(num_data))   {
                for (auto vec : data) {
                    table.insert(vec);
                }
                table.rebuild();
                MAX_HASHBITS = table.get_hashbits();
                MAX_REPETITIONS = table.get_repetitions();
                segments = table.order_segments();
                local_edges.resize(MAX_REPETITIONS);
                //dirty_start(local_Tus[0]);
                std::cout << "Index constructed " << MAX_REPETITIONS <<  " L, K " << MAX_HASHBITS << " num data " << num_data << std::endl;
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
                        std::vector<EdgeTuple> local_Tc, local_top, local_Tu;
                        enumerate_edges(segments[j], local_Tu, local_Tc);   
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
                        
                        #pragma omp critical
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
                                        auto probability = table.get_probability(i, j, 1-std::get<0>(edge));
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
                float max_dist = 0;

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
                        std::vector<EdgeTuple> local_top, local_Tu, local_Tc;
                        DSU local_dsu(num_data);
                        enumerate_edges(segments[j], local_Tu, local_Tc, max_dist);
                        
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
                        // if (local_Tu.size() > 0)    
                        //     max_dist = std::get<0>(local_Tu.back());

                        // The edges of the local spanning tree are added to the global edges

                        local_edges[j].insert(local_edges[j].end(), 
                                        std::make_move_iterator(local_top.begin()),
                                        std::make_move_iterator(local_top.end()));
                        local_top.clear();
                        // local_Tcs[j].clear();
                        // local_Tus[j].clear();


                       // #pragma omp critical
                        {
                        // Every ẋ iterations we have a batch, construct the MST from the global edges
                        if ((j+1)%((int)MAX_REPETITIONS) == 0 && j!=0 && !found) {
                            // Move the top edges in with the new edges, in this way we keep the last spanning tree
                            // But we allow for a better solution
                            edges.insert(edges.end(),
                                        std::make_move_iterator(top.begin()),
                                        std::make_move_iterator(top.end()));
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
                            //max_dist = std::get<0>(edges.back());
                            edges.clear();
                            // If we have a tree, we check if it is a valid ɛ-EMST
                            if (top.size() == num_data - 1) {
                                float tree_weight = 0;
                                for (const auto& edge : top) {
                                    tree_weight += std::get<0>(edge);
                                }  
                                float bound_w = (1+epsilon)*bound_weight(top, i, j);
                                // If no edge is confirmed then we increase the probability
                                if (bound_w == 0 && j+1 == MAX_REPETITIONS) {
                                    delta+=0.05;
                                    std::cout << "Delta increased to " << delta << std::endl;
                                }
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
            void enumerate_edges(CollisionEnumerator& st, std::vector<EdgeTuple>& Tu_local, std::vector<EdgeTuple>& Tc_local, float max_dist=0) {
                // Discover edges that share the same prefix at iteration st.i, st.j
                std::vector<EdgeTuple> couples = table.all_close_pairs(st, max_dist);
                std::sort(couples.begin(), couples.end());
                std::cout << "Size couples: " << couples.size() << std::endl;
                size_t index = 0;
                if (couples.size() == 0) return;
                // Evaluate all pair distances
                for (auto couple : couples) {
                    // If the distance is less than the threshold, add it to the confirmed edges
                    if (table.get_probability(st.i, st.j, 1-std::get<0>(couple))  <  delta) {
                        index++;
                    }
                    else {
                        break;
                    }
                }
                // Split the couples into confirmed and unconfirmed edges
                Tc_local.insert(Tc_local.end(), std::make_move_iterator(couples.begin()), std::make_move_iterator(couples.begin()+index));
                // Tu local must have at most 10*num_data edges
                //Find if we can insert all the rest of the edges or we have to cut them
                // if (couples.size() - index > 5*num_data) {
                //     Tu_local.insert(Tu_local.end(), std::make_move_iterator(couples.begin()+index), std::make_move_iterator(couples.begin()+index+5*num_data));
                // }
                // else {
                    Tu_local.insert(Tu_local.end(), std::make_move_iterator(couples.begin()+index), std::make_move_iterator(couples.end()));
                //}
                //std::cout << "Size Tc: " << Tc_local.size() << " Size Tu: " << Tu_local.size() << std::endl;
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
                    clean.emplace_back(1-table.get_similarity(vertices[i-1], vertices[i]), std::make_pair(vertices[i-1], vertices[i]));
                }
                std::sort(clean.begin(), clean.end());
            }

            /// @brief Clear the data structures from previous runs
            void clear() {
                top.clear();
            }
            
    };  //closes class
}       //closes namespace
