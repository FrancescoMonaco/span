#pragma once
namespace panna {
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

        /// @brief Check if x and y are in the same set
        /// @param x first element
        /// @param y second element
        /// @return true if x and y are in the same set, false otherwise
        bool is_connected(uint32_t x, uint32_t y) {
            return find(x) == find(y);
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

}