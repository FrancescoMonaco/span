#pragma once
#include "puffinn/collection.hpp"
#include "puffinn/hash_source/hash_source.hpp"
#include "puffinn/similarity_measure/cosine.hpp"
#include "puffinn/format/real_vector.hpp"
#include "puffinn/math.hpp"
#include <iostream>
#include <set>
#include <stack>
#include <vector>
#include <unistd.h>
using EdgeTuple = std::tuple<float, std::pair<unsigned int, unsigned int>>;

namespace puffinn{

    struct Prefix{
        uint_fast32_t i;
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
        

    class EMST{
        // Object variables
        uint32_t dimensionality;
        uint64_t memory_limit;
        size_t MAX_REPETITIONS;
        uint32_t MAX_HASHBITS;
        Index<CosineSimilarity> table;
        std::vector<std::vector<float>> data {};
        std::vector<CollisionEnumerator> segments;
        uint32_t num_data {0};
        // Sets for the confimed and the unconfirmed edges
        std::set<EdgeTuple> Tc;
        std::set<EdgeTuple> Tu;
        DSU dsu;


        std::set<EdgeTuple> top;
        const float delta {0.01};
        const float epsilon {0.2};


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
                  table(Index<CosineSimilarity>(dimensionality, memory_limit)),
                  data(data_in),
                  num_data((data_in).size()),
                  delta(delta),
                  epsilon(epsilon),
                  dsu(DSU(num_data))
            {
                for (auto vec : data) {
                    table.insert(vec);
                }
                table.rebuild();
                MAX_HASHBITS = table.get_hashbits();
                MAX_REPETITIONS = table.get_repetitions();
                segments = table.order_segments();
                std::cout << "EMST constructed " << MAX_REPETITIONS <<  " L, K " << MAX_HASHBITS << " num data " << num_data << std::endl;
            };

            /// @brief Destructor
            ~EMST() = default;

            /// @brief Find the Minimum Spanning Tree using only confirmed edges
            std::vector<std::pair<unsigned int, unsigned int>> find_tree() {
                std::vector <std::pair<unsigned int, unsigned int>> tree;
                return tree;
            }


            /// @brief Find the ɛ-EMST
            std::vector<std::pair<unsigned int, unsigned int>> find_epsilon_tree() {
    
                std::vector<std::pair<unsigned int, unsigned int>> tree;
                bool found = false;
                for (int i=MAX_HASHBITS; i>= 0; i--){
                    if (found){
                        break;
                    }
                   // std::cout << "Iteration: " << segments[0].i <<" "<< i << std::endl;
                   // #pragma omp parallel for
                    for (size_t j=0; j<MAX_REPETITIONS; j++){
                        if (found){
                            continue;
                        }
                        std::vector<EdgeTuple> local_Tu, local_Tc;
                        enumerate_edges(segments[j], local_Tu, local_Tc);

                       // #pragma omp critical
                        {
                            for(auto edge : local_Tu){
                                // add_edge_nocheck(edge);
                                // Tu.insert(edge);
                                //If the edge is not already in the tree
                                if( top.find(edge) != top.end() )
                                    continue;
                                //If it doesn't form a cycle
                                if(add_edge(edge, dsu)){
                                    Tu.insert(edge);
                                }
                            }

                            for(auto edge : local_Tc){
                                // add_edge_nocheck(edge);
                                // Tu.insert(edge);
                                if (top.find(edge) != top.end())
                                    continue;
                                if(add_edge(edge, dsu)){
                                    Tc.insert(edge);
                                }
                            }
                            //

                            // Move all the confirmed edges in Tu to Tc
                            std::vector<EdgeTuple> edges_to_move;
                            for(auto edge : Tu){
                               // std::cout << "probability " << table.get_probability(i, j, std::get<0>(edge)) << std::endl;
                                if(table.get_probability(i, j, (std::get<0>(edge))) <= 1-delta){
                                    Tc.insert(edge);
                                    edges_to_move.push_back(edge);
                                }
                                else{
                                    // The edges are sorted by weight, so if one doesn't satisfy
                                    // the probability then the rest can't
                                    break;
                                }
                            }
                            for(auto edge : edges_to_move){
                                Tu.erase(edge);
                            }

                        }
                    
                        //std::cout << "top size: " << top.size() << std::endl;
                        //If we have num_data -1 edges compute the spanning tree weight
                        if(top.size()==num_data-1){
                            if (is_connected()){
                            std::cout << "Connected" << std::endl;}
                            float tree_weight = 0;
                            for (const auto& edge : top){
                                tree_weight += std::get<0>(edge);
                            }
                            float bounded_weigth = bound_weight(Tu, Tc);

                            std::cout << "Tree weight: " << tree_weight << " Bounded weight: " << bounded_weigth << std::endl;
                            // If less than (1+ɛ)(sum over Tc + |Tu|*max(Tu) ) we return, else we continue
                            if (tree_weight <= (1+epsilon)*bounded_weigth){
                                for (const auto& edge : top){
                                    tree.push_back(std::get<1>(edge));
                                }
                                //#pragma omp cancel for
                                found = true;
                                //return tree;
                            }
                        }
                    }
                    //std::cout << "merging segments" << std::endl;
                    table.merge_segments(segments);
                    //std::cout << "merged segments" << std::endl;
                }
                return tree;
            };

        //*** Private methods */
        private:

            
            void enumerate_edges(CollisionEnumerator st, std::vector<EdgeTuple>& Tu_local, std::vector<EdgeTuple>& Tc_local){
                // Discover edges that share the same prefix at iteration st.i, st.j
                //std::cout << "Enumerating edges for prefix: " << st.i << " " << st.j << std::endl;
                std::vector<EdgeTuple> couples = table.all_close_pairs(st);
                //std::cout << "Couples size: " << couples.size() << std::endl;
                // Evaluate all pair distances
                for (auto couple : couples){
                  //  std::cout << "Couple: " << (std::get<1>(couple)).first << " " << std::get<1>(couple).second << " " << std::get<float>(couple) << std::endl;
                    // If the distance is less than the threshold, add it to the confirmed edges
                    if (table.get_probability(st.i, st.j+1, std::get<float>(couple)) <= 1 - delta){
                        Tc_local.emplace_back(couple);

                        //std::cout << "Adding to Tc" << std::endl;
                    }
                    // Otherwise, add it to the unconfirmed edges
                    else{
                        Tu_local.emplace_back(couple);
                        //std::cout << "Adding to Tu" << std::endl;
                    }
                }

                return;
            };

            void prim_union_find(){
                return;
            }

            /// @brief Return the bound weight (1+ɛ)(sum over Tc + |Tu|*max(Tu) )
            /// @param Tu set of unconfirmed edges 
            /// @param Tc set of confirmed edges
            /// @return the weight
            float bound_weight(std::set<EdgeTuple> Tu, std::set<EdgeTuple> Tc){
                float weight = 0;
                if (Tc.size() == 0){
                    std::cout << "No confirmed edges" << std::endl;
                    return 0;
                }
                for (const auto edge : Tc){
                    weight += l2_distance_float_simple(&data[std::get<1>(edge).first][0], &data[std::get<1>(edge).second][0], dimensionality);
                    //std::get<0>(edge);
                }
                float max_confirmed = l2_distance_float_simple(&data[std::get<1>(*Tc.rbegin()).first][0], &data[std::get<1>(*Tc.rbegin()).second][0], dimensionality);
                //std::get<0>(*Tc.rbegin());
                weight += max_confirmed * Tu.size();

                return weight;
            }

            bool is_connected(){
                if(top.size() == (table.get_size()-1)){
                    std::unordered_map<unsigned int, std::vector<std::pair<unsigned int, float>>> graph;
                    for (const auto& edge_tuple : top) {
                        float w;
                        std::pair<unsigned int, unsigned int> e;
                        std::tie(w, e) = edge_tuple;
                        // Normalize stored edge.
                        std::pair<unsigned int, unsigned int> norm_e = {
                            std::min(e.first, e.second),
                            std::max(e.first, e.second)
                        };
                        graph[norm_e.first].push_back({norm_e.second, w});
                        graph[norm_e.second].push_back({norm_e.first, w});
                    }

                    for (auto const& t: graph){
                        std::cout << t.first << " :";
                        for (auto elem : t.second){
                            std::cout << elem.first << " ";
                        }
                        std::cout << " | ";
                    }

                
                return top.size() == (table.get_size()-1);
                }
                return false;
            };
            
            bool add_edge(const EdgeTuple& new_edge_input, DSU &dsu) {
                // Extract new edge
                std::pair<unsigned int, unsigned int> new_edge = std::get<1>(new_edge_input);
                float new_weight = std::get<0>(new_edge_input);
            
                // If the tree has reached max edges (n-1), check for the heaviest edge
                if (top.size() == num_data - 1) {
                    auto maxIt = top.rbegin();
                    float maxWeight = std::get<0>(*maxIt);
                    std::pair<unsigned int, unsigned int> edgeToRemove = std::get<1>(*maxIt);
            
                    if (new_weight >= maxWeight) {
                        return false; // New edge is not better
                    }
            
                    // Temporarily remove the heaviest edge
                    top.erase(*maxIt);
                    dsu = DSU(num_data); // Reset DSU
            
                    // Rebuild DSU without max edge
                    for (const auto& edge_tuple : top) {
                        std::pair<unsigned int, unsigned int> e = std::get<1>(edge_tuple);
                        dsu.union_sets(e.first, e.second);
                    }
            
                    // Try inserting new edge
                    if (dsu.union_sets(new_edge.first, new_edge.second)) {
                        top.insert(std::make_tuple(new_weight, new_edge));
                        return true;
                    } else {
                        // Reinsert max edge if cycle was found
                        top.insert(std::make_tuple(maxWeight, edgeToRemove));
                        return false;
                    }
                }
            
                // Check if adding the new edge forms a cycle
                if (dsu.union_sets(new_edge.first, new_edge.second)) {
                    top.insert(std::make_tuple(new_weight, new_edge));
                    return true;
                }
            
                return false;
            }

            bool has_cycle(
                const std::unordered_map<unsigned int, std::vector<std::pair<unsigned int, float>>>& graph,
                const std::pair<unsigned int, unsigned int>& new_edge,
                std::vector<int>& parent,
                std::vector<bool>& visited
            ) {
                std::stack<unsigned int> s;
                s.push(new_edge.first);
                visited[new_edge.first] = true;

                while (!s.empty()) {
                    unsigned int cur = s.top();
                    s.pop();
                    if (graph.find(cur) == graph.end()) {
                        continue;
                    }
                    for (const auto& neighbor : graph.at(cur)) {
                        if (!visited[neighbor.first]) {
                            visited[neighbor.first] = true;
                            parent[neighbor.first] = cur;

                            if (neighbor.first == new_edge.second) {
                                return true; // Cycle detected
                            }
                            s.push(neighbor.first);
                        }
                    }
                }
                return false; // No cycle found
            }
    };  //closes class
}       //closes namespace
