PyPivoter
=========

Purpose
-------

The purpose of this package is to count and/or list the cliques, or 
fully-connected subgraphs, within a graph using the fast Pivoter 
algorithm developed by Shweta Jain and C. Seshadhri.

Installation
------------

### From a repository checkout

```bash
make install
```
or
```bash
CYTHONIZE=1 pip install --user .
```

### From PyPi

```bash
pip install --user pypivoter
```


Use
---

The two main functions available to the user are countCliques and 
enumerateCliques, which can be imported as follows:

```
>> from pypivoter.degeneracy_cliques import countCliques, enumerateCliques
```

Both take two arguments. The first is an m x 2 NumPy array of 
integer indices of vertex pairs that comprise the edges of a 
graph, with no repeated or reversed pairs or self-adjacency. 
The second is an integer, the maximum clique size to output. 
If the second argument is 0, all sizes will be output.

Example output from countCliques is:

```
>> import numpy as np
>> from pypivoter.degeneracy_cliques import countCliques
>> tetrahedron = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
>> countCliques(tetrahedron, 0)
[1 4 6 4 1]
```

Example output from enumerateCliques is:
```
>> import numpy as np
>> from pypivoter.degeneracy_cliques import enumerateCliques
>> tetrahedron = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
>> enumerateCliques(tetrahedron, 0)
[array([], shape=(0, 0), dtype=int32), array([[0],
       [1],
       [2],
       [3]], dtype=int32), array([[1, 0],
       [2, 0],
       [2, 1],
       [3, 1],
       [3, 0],
       [3, 2]], dtype=int32), array([[2, 0, 1],
       [3, 1, 0],
       [3, 1, 2],
       [3, 0, 2]], dtype=int32), array([[3, 1, 0, 2]], dtype=int32)]
```
