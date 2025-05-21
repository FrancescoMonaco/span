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

    // @brief Computes the binomial coefficient, 
    //        adapted from https://stackoverflow.com/questions/55421835/c-binomial-coefficient-is-too-slow
    // @param n number of elements
    // @param k number of elements to choose
    // @return the binomial coefficient C(n, k)
    double BinomialCoefficient(const double n, const double k) {
        double aSolutions, oldSolutions;
        oldSolutions = n - k + 1;

        for (double i = 1; i < k; ++i) {
            aSolutions = oldSolutions * (n - k + 1 + i) / (i + 1);
            oldSolutions = aSolutions;
        }
        return aSolutions;
    }

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
            // Union by size
            // if (rank[rootX] < rank[rootY]) {
            //     uint32_t temp = rootX;
            //     rootX = rootY;
            //     rootY = temp;
            // }

            // parent[rootY] = rootX;
            // rank[rootX] += rank[rootY];
            return true;
        }
    };
        
    template <typename Dataset, typename Hasher, typename Distance>
    class EMST {
        // Object variables
        uint32_t dimensionality;
        size_t MAX_REPETITIONS;
        uint32_t MAX_HASHBITS;
        Index<Dataset, Hasher, Distance> table;
        std::vector<std::vector<float>> data {};
        uint32_t num_data {0};
        double delta {0.01};
        const float epsilon {0.01};
        DSU dsu_true;
        // Sets for the confimed and the unconfirmed edges
        std::vector<EdgeTuple> top;
        std::vector<std::vector<EdgeTuple>> local_confirmed;
        std::vector<std::vector<EdgeTuple>> local_edges;
        std::vector<EdgeTuple> confirmed;
        std::vector<EdgeTuple> unconfirmed;


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
             * 1. Set up the LSH index table with the distance metric
             * 2. Insert all input vectors into the index
             * 3. Rebuilding the index structure
             * 4. Construct a Union Find data structure
             * The constructor takes ownership of the input data through a move operation.
             */
            EMST(const size_t dimensions, const size_t repetitions, const typename Hasher::Builder builder, std::vector<std::vector<float>>& data_in, const double delta_in = 0.1, const float epsilon = 0.2)
                : dimensionality(dimensions),
                  table( Index<Dataset, Hasher, Distance>(dimensions, builder, repetitions) ),
                  data( data_in ),
                  num_data( (data).size() ),
                  epsilon(epsilon),
                  dsu_true( DSU(num_data) )   {
                
                // Insert the data
                for ( auto& point: data_in ) {
                    normalize( point );
                    table.insert( point.begin(), point.end() );
                }
                table.rebuild();


                // Get info on the index
                MAX_HASHBITS = table.num_concatenations();
                MAX_REPETITIONS = table.num_repetitions();
                delta = delta_in/BinomialCoefficient(num_data, 2);

                local_edges.resize(MAX_REPETITIONS);
                local_confirmed.resize(MAX_REPETITIONS);
                //dirty_start(local_Tus[0]);
                std::cout << "Index constructed, L: " << MAX_REPETITIONS <<  " K: " << MAX_HASHBITS << " num data: " << num_data << std::endl;
                std::cout << "Delta: " << delta << std::endl;
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
                local_edges.resize(num_data);
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
                clear();
                std::vector<std::pair<unsigned int, unsigned int>> tree;

                bool found = false;
                std::vector<EdgeTuple> edges;
                for (size_t i = MAX_HASHBITS; i >= 0; i--) {
                    if (found)
                        break;
#pragma omp parallel for
                    for (size_t j = 0; j < MAX_REPETITIONS; j++) {
                        if (found) 
                            continue;
                        DSU local_dsu(num_data);
                        std::vector<EdgeTuple> local_Tc, local_top, local_Tu;
                        enumerate_edges(i, j, local_Tu, local_Tc);   
                        // local_confirmed[j].insert( local_confirmed[j].end(),
                        //                             std::make_move_iterator( local_Tc.begin() ),
                        //                             std::make_move_iterator( local_Tc.end() ) );
                        // local_edges[j].insert( local_edges[j].end(),
                        //                             std::make_move_iterator( local_Tu.begin() ),
                        //                             std::make_move_iterator( local_Tu.end() ) );
                        
                        for ( auto& edge: local_Tc ) {
                            if (local_top.size() == num_data - 1) {
                                break;
                            }
                            add_edge(edge, local_dsu, local_top);
                        }

                        for ( auto& edge: local_Tu ) {
                            if (local_top.size() == num_data - 1) {
                                break;
                            }
                            add_edge(edge, local_dsu, local_top);
                        }

                        local_confirmed[j].insert( local_confirmed[j].end(),
                                                    std::make_move_iterator( local_top.begin() ),
                                                    std::make_move_iterator( local_top.end() ) );
#pragma omp critical
                        {
                        // Every x iterations we have a batch, construct the MST from these edges
                        if ( ((j+1) == MAX_REPETITIONS || (j+1) == (size_t)MAX_REPETITIONS/2) ) {
                            edges.insert( edges.end(),
                                std::make_move_iterator(top.begin()),
                                std::make_move_iterator(top.end()) );
                            top.clear();
                            dsu_true = DSU(num_data);
                            for (auto& local : local_confirmed) {
                                    edges.insert( edges.end(), local.begin(), local.end() );
                                // std::make_move_iterator(local.begin()), std::make_move_iterator(local.end()) );
                                local.clear();
                            }
                            // for ( auto& local : local_edges ) {
                            //     auto it = std::partition_point( local.begin(), local.end(), [&] (const auto& e) { 
                            //         return table.fail_probability( std::get<float> (e), i, j ) < delta;                               
                            //         } );
                            //     edges.insert ( edges.end(), local.begin(), it );
                            //     // std::make_move_iterator(local.begin()), std::make_move_iterator(it) );
                            //     local.erase(local.begin(), it);   
                            // } 
                            // std::sort( unconfirmed.begin(), unconfirmed.end() );
                            // unconfirmed.erase( std::unique( unconfirmed.begin(), unconfirmed.end() ), unconfirmed.end() );

                            // // Find the slitting point
                            // auto it = std::partition_point( unconfirmed.begin(), unconfirmed.end(), [&] (const auto& e) { 
                            //     return table.fail_probability( std::get<float> (e), i, j ) < delta;                               
                            //     } );
                            // confirmed.insert ( confirmed.end(), unconfirmed.begin(), it );
                            std::sort ( edges.begin(), edges.end() );
                            edges.erase( std::unique( edges.begin(), edges.end() ), edges.end() );
                            if (edges.size() > num_data -1) {
                                for (const auto& edge : edges) {
                                    add_edge( edge, dsu_true, top );
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
                                if (top.size() == num_data - 1) {
                                    float tree_weight = 0;
                                    for (const auto& edge : top) {
                                            tree_weight += std::get<0>(edge);
                                        }                           
                                    std::cout << "Tree weight: " << tree_weight << std::endl;
                                    std::cout << "Probability: " << table.fail_probability(std::get<float>(top.back()), i, j) << std::endl;
                                    if (table.fail_probability( std::get<float>(top.back()), i, j) < delta ) {
                                        found = true;
                                    }
                                        // Fill the tree
                                    for (const auto& edge : top) {
                                            tree.push_back( std::get<1>(edge) );
                                    }
                                }
                                // Lose the unused edges, MST is composable wrt to edge partitioning
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
                clear();
                std::vector<std::pair<unsigned int, unsigned int>> tree;

                bool found = false;
                std::vector<EdgeTuple> edges;
                for (size_t i = MAX_HASHBITS; i > 0; i--) {
                    if (found)
                        break;
                    //#pragma omp parallel for
                    for (size_t j = 0; j < MAX_REPETITIONS; j++) {
                        if (found) {
                            continue;
                        }
                        DSU local_dsu(num_data);
                        std::vector<EdgeTuple> local_Tc, local_top, local_Tu;
                        enumerate_edges(i, j, local_Tu, local_Tc);   
                        local_confirmed[j].insert( local_confirmed[j].end(),
                                                    std::make_move_iterator( local_Tc.begin() ),
                                                    std::make_move_iterator( local_Tc.end() ) );
                        local_edges[j].insert( local_edges[j].end(),
                                                    std::make_move_iterator( local_Tu.begin() ),
                                                    std::make_move_iterator( local_Tu.end() ) );
                        std::sort( local_confirmed[j].begin(), local_confirmed[j].end() );
                        std::sort( local_edges[j].begin(), local_edges[j].end() );
                        //#pragma omp critical
                        {
                        //! Every x iterations we have a batch, construct the MST from these edges
                        if (!found && ((j+1) == MAX_REPETITIONS || (j+1) == (size_t)MAX_REPETITIONS/2)) {
                            // edges.insert( edges.end(),
                            //     std::make_move_iterator(top.begin()),
                            //     std::make_move_iterator(top.end()) );
                            std::vector<EdgeTuple> unconfirmed_ordered_edges;
                            top.clear();
                            dsu_true = DSU(num_data);
                            for (auto& local : local_confirmed) {
                                edges.insert( edges.end(), local.begin(), local.end() );
                                // std::make_move_iterator(local.begin()), std::make_move_iterator(local.end()) );
                                //local.clear();
                            }
                            for ( auto& local : local_edges ) {
                                auto it = std::partition_point( local.begin(), local.end(), [&] (const auto& e) { 
                                    return table.fail_probability( std::get<float> (e), i, j ) < delta;                               
                                    } );
                                edges.insert( edges.end(), local.begin(), it );
                                unconfirmed_ordered_edges.insert( unconfirmed_ordered_edges.end(), it, local.end() );
                                // std::make_move_iterator(local.begin()), std::make_move_iterator(it) );
                                //local.erase(local.begin(), it);   
                            } 
                            std::sort( edges.begin(), edges.end() );
                            for (const auto& edge : edges) {
                                add_edge( edge, dsu_true, top );
                                if (top.size() == num_data - 1) {
                                    break;
                                }
                            }
                            // If the tree is not full, complete it with the unconfirmed edges
                            if (top.size() < num_data - 1) {
                                std::sort(unconfirmed_ordered_edges.begin(), unconfirmed_ordered_edges.end());
                                std::cout << "Unconfirmed edges: " << unconfirmed_ordered_edges.size() << " Size " << top.size() << std::endl;
                                for (const auto& edge : unconfirmed_ordered_edges) {
                                    if (top.size() == num_data - 1) {
                                        break;
                                    }
                                    add_edge(edge, dsu_true, top);
                                }
                            }
                            std::cout << "Tree size: " << top.size() << std::endl;
                            // Print also the current weight
                            float tree_weight = 0;
                            for (const auto& edge : top) {
                                tree_weight += std::get<0>(edge);
                            }

                            std::cout << "Tree weight: " << tree_weight << " Bound: " << (1+epsilon)*bound_weight(top, i, j) << std::endl;
                            if ( tree_weight < (1+epsilon)*bound_weight(top, i, j) ) {                        
                                    found = true;
                                        // Fill the tree
                                    for (const auto& edge : top) {
                                            tree.push_back(std::get<1>(edge));
                                    }
                            }
                            else {
                                found = false;
                            }
                            // Lose the unused edges
                            edges.clear();                         
                        }
                        }
                    }
                    std::cout << "Finished prefix " << i << std::endl;
                }
                is_connected(tree);
                return tree;
            }

        //*** Private methods */
        private:

            /// @brief Obtain the couples of nodes that share the same prefix from the hash table
            ///        and split them into edges whose recall is above the threshold and the others
            /// @param i Current concatenation in the hash index
            /// @param j current repetition in the hash index
            /// @param Tu_local vector that stores the unconfirmed edges
            /// @param Tc_local vector that stores the confirmed edges
            void enumerate_edges(size_t i, size_t j, std::vector<EdgeTuple>& Tu_local, std::vector<EdgeTuple>& Tc_local) {
                // Discover edges that share the same prefix at iteration i, j
                std::vector<EdgeTuple> couples;
                table.search_pairs(j, i, couples);
                // Find the edges that are confirmed and the ones that are not, the edges are ordered in ascending order so we can binary search the splitting point
                // We compute the probability using collision_probability(distance) of each edge, and find all the edges that are above the threshold delta
                auto it = std::partition_point( couples.begin(), couples.end(), [&] (const auto& e) { 
                    return table.fail_probability( std::get<float> (e), i, j ) < delta;                
                    } );
                Tu_local.insert(Tu_local.end(), it, couples.end());
                Tc_local.insert(Tc_local.end(), couples.begin(), it);
                // std::cout << "Size Tu: " << Tu_local.size() << " Size Tc: " << Tc_local.size() << std::endl;
                return;
            };

            /// @brief Return the bound weight (1+ɛ)(sum over Tc + |Tu|*max(Tu) )
            /// @param top_copy a vector that contains the edges in the spanning tree
            /// @return the weight
            float bound_weight(const std::vector<EdgeTuple>& top_copy, size_t i, size_t j) {
                float weight = 0;
                float max_confirmed = 0;
                int unconfirmed = 0;
                // Add the weight of the confirmed edges, keep track of the max confirmed and count the unconfirmed ones
                for (const auto& edge : top_copy) {
                    auto edge_weight = std::get<float>(edge);  
                    auto probability = table.fail_probability( edge_weight, i, j );
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
                std::cout << "Max confirmed: " << max_confirmed << " Unconfirmed: " << unconfirmed << std::endl;
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
            bool add_edge(const EdgeTuple& new_edge_input, DSU& dsu, std::vector<EdgeTuple>& edge_list) {
                // Extract new edge and its weight.
                std::pair<uint32_t, uint32_t> new_edge = std::get<1>(new_edge_input);
            
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
                std::iota( vertices.begin(), vertices.end(), 0 );
                std::random_shuffle( vertices.begin(), vertices.end() );
                for (size_t i = 1; i < vertices.size(); i++) {
                    clean.emplace_back( Distance::compute(vertices[i-1], vertices[i]), std::make_pair(vertices[i-1], vertices[i]) );
                }
                std::sort( clean.begin(), clean.end() );
            }

            /// @brief Clear the data structures from previous runs
            void clear() {
                top.clear();
                for (auto& local: local_edges) {
                    local.clear();
                }
                for (auto& local: local_confirmed) {
                    local.clear();
                }
                local_confirmed.resize(MAX_REPETITIONS);
                local_edges.resize(MAX_REPETITIONS);
            }
    };  //closes class
}       //closes namespace
