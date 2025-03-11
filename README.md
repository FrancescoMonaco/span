# SPAN
SPAN is library to find the approximate Euclidean Minumum Spanning Tree. The algorithm is based on the parameterless Locality Sensitive Hashing index [PUFFINN](https://github.com/puffinn/puffinn).
# Usage
Currently implemented in C++. The library keeps the header-only approach of PUFFINN, including <code>emst.hpp</code> is sufficient. To obtain a MST create an <code>EMST</code> object
and use <code>find_epsilon_tree</code> or <code>find_tree</code>.

```cpp
#include "emst.hpp"

int main() {
    std::vector<std::vector<float>> dataset = ...;
    int dimensions = ...;
    int memory = ...;
    puffinn::EMST emst(dimensions, memory, dataset);
    std::vector<std::pair<unsigned int, unsigned int>> tree = emst.find_epsilon_tree();
    
}
```
