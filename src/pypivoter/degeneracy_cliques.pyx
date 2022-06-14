import os
import numpy as np

from libc.stdlib cimport calloc, malloc, free
from libc.stdio cimport printf, fflush, stdout, stderr
from libc.string cimport memcpy

from .degeneracy_helper cimport *

# This code is a modified version of the code of quick-cliques-1.0 library for
# counting maximal cliques by Darren Strash (first name DOT last name AT gmail
# DOT com).

# Original author: Darren Strash (first name DOT last name AT gmail DOT com)

# Copyright (c) 2011 Darren Strash. This code is released under the GNU Public
# License (GPL) 3.0.

# Modification copyright (c) 2020 Shweta Jain
# Modification copyright (c) 2022 Rian Kormos

cdef int ipow(int base, int exp):
    """Raise one integer to the power of another integer.

    Parameters
    ----------
    base : int
        The integer that will be raised to a power.
    exp : int
        The integer that is the power to which the base will be raised.

    Returns
    -------
    int
        Base raised to the power of exp.
    """
    if exp == 0:
        return 1
    assert exp > 0
    cdef int result = 1
    while 1:
        if exp & 1:
            result *= base
        exp >>= 1 # bit-shift the exponent
        if not exp:
            break
        base *= base
    return result


cdef LinkedList** graphAdjArrayToDoubleEdges(int n, int m, int** arr):
    """Read in a graph from a 2 x m array of vertices that form edges and
       return an adjacency list as an array of linked lists of integers.

    Parameters
    ----------
    n : int
        Pointer to memory address where the number of vertices in the graph
        is stored.
    m : int
        Pointer to memory address where the number of edges in the graph
        is stored.
    arr : int**
        Pointer to 2 x n array of vertices that form edges of a graph, with
        no repeated or reversed pairs and no self-adjacency.

    Returns
    -------
    LinkedList**
        An array of linked lists of integers (the adjacency list representation
        of the graph)
    """
    cdef int u, v # end vertices, to read edges

    cdef LinkedList** adjList = <LinkedList**> calloc(n, sizeof(LinkedList*))

    cdef int i = 0
    while i < n:
        adjList[i] = createLinkedList()
        i += 1

    i = 0
    while i < m:
        u = arr[0][i]
        v = arr[1][i]
        assert u < n and u > -1
        assert v < n and v > -1
        assert u != v

        addLast(adjList[u], v)
        addLast(adjList[v], u)

        i += 1

    return adjList


cdef void fillInPandXForRecursiveCall(int vertex, int orderNumber,
                                      int* vertexSets, int* vertexLookup,
                                      NeighborListArray** orderingArray,
                                      int** neighborsInP, int* numNeighbors,
                                      int lastVertIndex, int* pBeginP,
                                      int* pBeginR):
    """Move vertex to R, set P to vertex's later neighbors, and set X to
       vertex's earlier neighbors.

    Parameters
    ----------
    vertex : int
        The integer index of the vertex to move to R.
    orderNumber : int
        The position of the vertex in the degeneracy ordering.
    vertexSets : int*
        An array containing sets of vertices divided into sets X, P, and R.
    vertexLookup : int*
        A lookup table indexed by vertex number, storing the index of that
        vertex in vertexSets.
    orderingArray : NeighborListArray**
        A degeneracy order of the input graph.
    neighborsInP : int**
        Maps vertices to arrays of neighbors such that neighbors in P fill the
        first cells.
    numNeighbors : int*
        The number of neighbors a vertex had in P the first time this function
        is called (used as a bound to prevent allocation of more than linear
        space in memory).
    lastVertIndex : int
        The index of the last vertex in vertexSets, equal to the number of
        vertices in the graph minus one.
    pBeginP : int*
        After the function, contains the index where set P begins in vertexSets
        after adding vertex to R.
    pBeginR : int*
        After the function, contains the index where set R begins in vertexSets
        after adding vertex to R.
    """
    cdef int vertexLocation = vertexLookup[vertex]

    # move vertex to R and update data structures accordingly
    vertexSets[vertexLocation] = vertexSets[lastVertIndex]
    vertexLookup[vertexSets[lastVertIndex]] = vertexLocation
    vertexSets[lastVertIndex] = vertex
    vertexLookup[vertex] = lastVertIndex

    pBeginR[0] = lastVertIndex # cython analog of *pBeginR = lastVertIndex
    pBeginP[0] = lastVertIndex # cython analog of *pBeginP = lastVertIndex

    # swap later neighbors of vertex into P section of vertexSets
    cdef int j = 0
    cdef int neighbor, neighborLocation
    while j < orderingArray[orderNumber].laterDegree:
        neighbor = orderingArray[orderNumber].later[j]
        neighborLocation = vertexLookup[neighbor]

        pBeginP[0] -= 1 # cython analog of *pBeginP -= 1

        vertexSets[neighborLocation] = vertexSets[pBeginP[0]]
        vertexLookup[vertexSets[pBeginP[0]]] = neighborLocation
        vertexSets[pBeginP[0]] = neighbor
        vertexLookup[neighbor] = pBeginP[0]

        j += 1

    # reset numNeighbors and neighborsInP for this vertex
    cdef int vertexInP
    cdef int calloc_size
    j = pBeginP[0]
    while j < pBeginR[0]:
        vertexInP = vertexSets[j]
        numNeighbors[vertexInP] = 0
        j += 1

    # count neighbors in P, and fill array of neighbors in P
    j = pBeginP[0]
    cdef int k, laterNeighbor, laterNeighborLocation
    while j < pBeginR[0]:
        vertexInP = vertexSets[j]
        k = 0
        while k < orderingArray[vertexInP].laterDegree:
            laterNeighbor = orderingArray[vertexInP].later[k]
            laterNeighborLocation = vertexLookup[laterNeighbor]

            if laterNeighborLocation >= pBeginP[0] and \
                    laterNeighborLocation < pBeginR[0]:
                neighborsInP[vertexInP][numNeighbors[vertexInP]] = \
                    laterNeighbor
                numNeighbors[vertexInP] += 1
                neighborsInP[laterNeighbor][numNeighbors[laterNeighbor]] = \
                    vertexInP
                numNeighbors[laterNeighbor] += 1

            k += 1
        j += 1


cdef void moveToR(int vertex, int* vertexSets, int* vertexLookup,
                  int** neighborsInP, int* numNeighbors, int* pBeginP,
                  int* pBeginR, int* pNewBeginP, int* pNewBeginR, int n):
    """Move a vertex to the set R, and update the sets P and X, and the arrays
       of neighbors in P.

    Parameters
    ----------
    vertex : int
       The integer index of the vertex to move to R.
    vertexSets : int*
       An array containing sets of vertices divided into sets X, P, and R.
    vertexLookup : int*
       A lookup table indexed by vertex number, storing the index of that
       vertex in vertexSets.
    neighborsInP : int**
       Maps vertices to arrays of neighbors such that neighbors in P fill the
       first cells.
    numNeighbors : int*
       The number of neighbors a vertex had in P the first time this function
       is called (used as a bound to prevent allocation of more than linear
       space in memory).
    pBeginP : int*
       The index where set P begins in vertexSets.
    pBeginR : int*
       The index where set R begins in vertexSets.
    pNewBeginP : int*
       After the function, contains the new index where set P begins in
       vertexSets after adding vertex to R.
    pNewBeginR : int*
       After the function, contains the new index where set R begins in
       vertexSets after adding vertex to R.
    n : int
       Total number of vertices.
    """
    cdef int vertexLocation = vertexLookup[vertex]

    pBeginR[0] -= 1 # cython analog of *pBeginR -= 1
    vertexSets[vertexLocation] = vertexSets[pBeginR[0]]
    vertexLookup[vertexSets[pBeginR[0]]] = vertexLocation
    vertexSets[pBeginR[0]] = vertex
    vertexLookup[vertex] = pBeginR[0]

    pNewBeginP[0] = pBeginP[0] # cython analog of *pNewBeginP = *pBeginP
    pNewBeginR[0] = pBeginP[0] # cython analog of *pNewBeginR = *pBeginP

    cdef int sizeOfP = pBeginR[0] - pBeginP[0]

    cdef int j = pBeginP[0]
    cdef int neighbor, neighborLocation, numPotentialNeighbors, k
    while j < pBeginR[0]:
        neighbor = vertexSets[j]
        neighborLocation = j

        numPotentialNeighbors = min(sizeOfP, numNeighbors[neighbor])
        k = 0
        while k < numPotentialNeighbors:
            if neighborsInP[neighbor][k] == vertex:
                vertexSets[neighborLocation] = vertexSets[pNewBeginR[0]]
                vertexLookup[vertexSets[pNewBeginR[0]]] = neighborLocation
                vertexSets[pNewBeginR[0]] = neighbor
                vertexLookup[neighbor] = pNewBeginR[0]
                pNewBeginR[0] += 1 # cython analog of *pnewBeginR += 1
            k += 1
        j += 1

    j = pNewBeginP[0]
    cdef int thisVertex, numNeighborsInP
    while j < pNewBeginR[0]:
        thisVertex = vertexSets[j]
        numPotentialNeighbors = min(sizeOfP, numNeighbors[thisVertex])
        numNeighborsInP = 0
        k = 0
        while k < numPotentialNeighbors: 
            neighbor = neighborsInP[thisVertex][k]
            neighborLocation = vertexLookup[neighbor]
            if neighborLocation >= pNewBeginP[0] and \
                    neighborLocation < pNewBeginR[0]:
                neighborsInP[thisVertex][k] = \
                    neighborsInP[thisVertex][numNeighborsInP]
                neighborsInP[thisVertex][numNeighborsInP] = neighbor
                numNeighborsInP += 1
            k += 1
        j += 1

cdef void moveFromRToX(int vertex, int* vertexSets, int* vertexLookup,
                       int* pBeginP, int* pBeginR):
    """Move a vertex from the set R to the set X and update all necessary
       pointers and arrays of neighbors in P.

    Parameters
    ----------
    vertex : int
       The integer index of the vertex to move to R.
    vertexSets : int*
       An array containing sets of vertices divided into sets X, P, and R.
    vertexLookup : int*
       A lookup table indexed by vertex number, storing the index of that
       vertex in vertexSets.
    pBeginP : int*
       The index where set P begins in vertexSets.
    pBeginR : int*
       The index where set R begins in vertexSets.
    """
    cdef int vertexLocation = vertexLookup[vertex]

    # swap vertex into X and increment beginP and beginR
    vertexSets[vertexLocation] = vertexSets[pBeginP[0]]
    vertexLookup[vertexSets[pBeginP[0]]] = vertexLocation
    vertexSets[pBeginP[0]] = vertex
    vertexLookup[vertex] = pBeginP[0]

    pBeginP[0] = pBeginP[0] + 1 # cython analog of *pBeginP = *pBeginP + 1
    pBeginR[0] = pBeginR[0] + 1 # cython analog of *pBeginR = *pBeginR + 1


cdef int findBestPivotNonNeighbors(int** pivotNonNeighbors,
                                   int* numNonNeighbors,
                                   int* vertexSets, int* vertexLookup,
                                   int** neighborsInP, int* numNeighbors,
                                   int beginP, int beginR):
    """Computes the vertex v in P union X that has the most neighbors in P,
       and places P \ {neighborhood of v} in an array.  These are the vertices
       to consider adding to the partial clique during the current recursive
       call of the algorithm.

    Parameters
    ----------
    pivotNonNeighbors : int**
        An initially unallocated pointer, which will contain the set
        P \ {neighborhood of v} when this function completes.
    numNonNeighbors : int*
        A pointer to a single integer, which has been preallocated and which
        will contain the number of elements in pivotNonNeighbors.
    vertexSets : int*
       An array containing sets of vertices divided into sets X, P, and R.
    vertexLookup : int*
       A lookup table indexed by vertex number, storing the index of that
       vertex in vertexSets.
    neighborsInP : int**
       Maps vertices to arrays of neighbors such that neighbors in P fill the
       first cells.
    numNeighbors : int*
       The number of neighbors a vertex had in P the first time this function
       is called (used as a bound to prevent allocation of more than linear
       space in memory).
    BeginP : int
       The index where set P begins in vertexSets.
    BeginR : int
       The index where set R begins in vertexSets.

    Returns
    -------
    int
        The integer index of the pivot vertex (i.e. the one with the most nbrs.)
    """
    cdef int pivot = -1
    cdef int maxIntersectionSize = -1

    # iterate over each vertex in P union X to find the vertex with the most
    # neighbors in P.
    cdef int j = beginP
    cdef int vertex, numPotentialNeighbors, numNeighborsInP, \
             k, neighbor, neighborLocation
    while j < beginR:
        vertex = vertexSets[j]
        numPotentialNeighbors = min(beginR - beginP, numNeighbors[vertex])

        numNeighborsInP = 0

        k = 0
        while k < numPotentialNeighbors:
            neighbor = neighborsInP[vertex][k]
            neighborLocation = vertexLookup[neighbor]

            if neighborLocation >= beginP and neighborLocation < beginR:
                numNeighborsInP += 1
            else:
                break

            k += 1

        if numNeighborsInP > maxIntersectionSize:
            pivot = vertex
            maxIntersectionSize = numNeighborsInP

        j += 1

    # Compute non neighbors of pivot by marking its neighbors and moving non-
    # marked vertices into pivotNonNeighbors.
    # This must be done because it is an efficient way to compute non-neighbors
    # of a vertex in an adjacency list.
    # Enough space is initialized for all of P; this is slightly space
    # inefficient, but it results in faster computation of non-neighbors.
    pivotNonNeighbors[0] = <int*> calloc(beginR - beginP, sizeof(int))
    # cython analog of *pivotNonNeighbors = ...
    memcpy(pivotNonNeighbors[0], &vertexSets[beginP],
           (beginR - beginP) * sizeof(int))

    # numNonNeighbors will be decremented as neighbors are found
    numNonNeighbors[0] = beginR - beginP
    # cython analog of *numNonNeighbors = beginR - beginP

    cdef int numPivotNeighbors = min(beginR - beginP, numNeighbors[pivot])

    # mark the neighbors of pivot that are in P
    j = 0
    while j < numPivotNeighbors:
        neighbor = neighborsInP[pivot][j]
        neighborLocation = vertexLookup[neighbor]

        if neighborLocation >= beginP and neighborLocation < beginR:
            pivotNonNeighbors[0][neighborLocation - beginP] = -1
            # cython analog of *pivotNonNeighbors[neighborLocation - beginP] =-1
        else:
            break
        j += 1

    # Move non-neighbors of pivot in P to the beginning of pivotNonNeighbors
    # and set numNonNeighbors appropriately.
    # If a vertex is marked as a neighbor, it is moved to the end of
    # pivotNonNeighbors and numNonNeighbors is decremented.
    j = 0
    while j < numNonNeighbors[0]:
        vertex = pivotNonNeighbors[0][j]

        if vertex == -1:
            numNonNeighbors[0] -= 1 # cython analog of *numNonNeighbors* -= 1
            pivotNonNeighbors[0][j] = pivotNonNeighbors[0][numNonNeighbors[0]]
            continue
        j += 1

    return pivot


cdef void listAllCliquesDegeneracyRecursive(int** cliques, int* cliqueCounts,
                                            int* vertexSets, int* vertexLookup,
                                            int** neighborsInP,
                                            int* numNeighbors, int* isHold,
                                            int* isPivot, int beginP,
                                            int beginR, int n, int max_k,
                                            int rsize, int drop, int* cliqueNum,
                                            int enumerate):
    """Recursively list all maximal cliques containing all vertices in R,
       some vertices in P, and no vertices in X.

    Parameters
    ----------
    cliques : int**
        Integer array [nCliques x max_k] to be populated with the vertices of
        all k-cliques, only if enumerate is 1.
    cliqueCounts : int*
        Zero-valued array of length max_k + 1 in which the clique counts are to
        be stored.
    orderingArray : NeighborListArray**
        An array of linked lists of integers (the adjacency list representation
        of the graph).  TODO: REMOVE
    vertexSets : int*
        An array containing sets of vertices divided into sets X, P, and R.
    vertexLookup : int*
        A lookup table indexed by vertex number, storing the index of that
        vertex in vertexSets.
    neighborsInP : int**
        Maps vertices to arrays of neighbors such that neighbors in P fill the
        first cells.
    numNeighbors : int*
        The number of neighbors a vertex had in P the first time this function
        is called (used as a bound to prevent allocation of more than linear
        space in memory).
    isHold : int*
        1 if a vertex is a hold vertex, 0 otherwise, for all vertices.
    isPivot : int*
        1 if a vertex is a pivot vertex, 0 otherwise, for all vertices.
    beginP : int
        The index where set P begins in vertexSets.
    beginR : int
        The index where set R begins in vertexSets.
    n : int
        The number of vertices in the graph.
    max_k : int
        The maximum size of k-cliques to be counted, or 0 if all k-cliques
        should be counted.
    rsize : int
        The size of the set R.
    drop : int
        The number of vertices to potentially drop from R (i.e. # pivots).
    cliqueNum : int*
        A counter to keep track of cliques in the case that enumeration is
        being carried out.
    enumerate : int
        1 if cliques are to be enumerated, 0 if cliques are to be counted.
    """
    cdef int i, j, k, l, m
    cdef int vertexNum, nCliquesToAdd, nAdded, nCr_num, nCr_denom
    if beginP >= beginR or rsize - drop > max_k:
        if enumerate == 1: # enumerate cliques
            nCliquesToAdd = ipow(2, drop)
            nAdded = 0
            for i in range(nCliquesToAdd):
                # for each clique, add all holds and some subset of the pivots
                k = 0 # this is not the same k as the clique size
                      # this counter keeps track of the # of pivots seen
                l = 0 # this counter keeps track of the # of pivots accepted
                m = 0 # this counter keeps track of the # of holds seen/accepted
                for j in range(beginR, n):
                    if isPivot[vertexSets[j]]:
                        if (i >> k) & 1:
                            cliques[cliqueNum[0] + nAdded][rsize - drop + l] = \
                                vertexSets[j]
                            l += 1
                        k += 1
                    elif isHold[vertexSets[j]]:
                        cliques[cliqueNum[0] + nAdded][m] = vertexSets[j]
                        m += 1
                if l + m <= max_k:
                    # if clique does not exceed max_k, increment nAdded so
                    # that it is not overwritten
                    nAdded += 1
                else:
                    for j in range(l + m):
                        cliques[cliqueNum[0] + nAdded][j] = -1
            cliqueNum[0] += nAdded
        else: # count cliques
            i = drop
            k = rsize - drop
            while i >= 0 and k <= max_k:
                k = rsize - i
                nCr_num = 1
                nCr_denom = 1
                for j in range(i):
                    nCr_num *= drop - j
                    nCr_denom *= j + 1
                cliqueCounts[k] += nCr_num // nCr_denom
                # cliqueCounts[k] += <int> nCr[drop][i]
                i -= 1
        
        return

    cdef int* candidatesToIterateThrough
    cdef int numCandidatesToIterateThrough = 0

    # get the candidates to add to R to make a maxmial clique 
    cdef int pivot = findBestPivotNonNeighbors(&candidatesToIterateThrough,
                                               &numCandidatesToIterateThrough,
                                               vertexSets, vertexLookup,
                                               neighborsInP, numNeighbors,
                                               beginP, beginR)

    # add candidate vertices to the partial clique one at a time and search
    # for maximal cliques
    cdef int iterator, vertex, vertexLocation, newBeginP, newBeginR
    if numCandidatesToIterateThrough != 0:
        iterator = 0
        while iterator < numCandidatesToIterateThrough:
            # vertex to be added to the partial clique
            vertex = candidatesToIterateThrough[iterator]
            # to add vertex into the partial clique, represented by R,
            # swap vertex into R and update all data structures
            moveToR(vertex, vertexSets, vertexLookup, neighborsInP,
                    numNeighbors, &beginP, &beginR, &newBeginP, &newBeginR, n)

            # recursively compute maximal cliques with new sets R, P, and X
            if vertex == pivot:
                isPivot[vertex] = 1 # keep track of pivot vertices in R

                listAllCliquesDegeneracyRecursive(cliques, cliqueCounts,
                                                  vertexSets, vertexLookup,
                                                  neighborsInP, numNeighbors,
                                                  isHold, isPivot, newBeginP,
                                                  newBeginR, n, max_k, rsize+1,
                                                  drop+1, cliqueNum, enumerate)
                
                isPivot[vertex] = 0 # reset state of isPivot for higher calls

            else:
                isHold[vertex] = 1 # keep track of hold vertices in R

                listAllCliquesDegeneracyRecursive(cliques, cliqueCounts,
                                                  vertexSets, vertexLookup,
                                                  neighborsInP, numNeighbors,
                                                  isHold, isPivot, newBeginP,
                                                  newBeginR, n, max_k, rsize+1,
                                                  drop, cliqueNum, enumerate)

                isHold[vertex] = 0 # reset state of isHold for higher calls

            moveFromRToX(vertex, vertexSets, vertexLookup, &beginP, &beginR)

            iterator += 1
        
        # swap vertices back from X to P, for higher recursive calls
        while iterator < numCandidatesToIterateThrough:
            vertex = candidatesToIterateThrough[iterator]
            vertexLocation = vertexLookup[vertex]

            beginP -= 1
            vertexSets[vertexLocation] = vertexSets[beginP]
            vertexSets[beginP] = vertex
            vertexLookup[vertex] = beginP
            vertexLookup[vertexSets[vertexLocation]] = vertexLocation

            iterator += 1

    # don't need to check for emptiness before freeing, since something will
    # always be there (we allocated enough memory for all of P, a nonempty set)
    free(candidatesToIterateThrough)

    return


cdef void listAllCliquesDegeneracy(int** cliques, int* cliqueCounts,
                                   NeighborListArray** orderingArray,
                                   int n, int deg, int max_k, int enumerate):
    """Populate the cliqueCounts array with clique counts for cliques up to size
       max_k given the degeneracy-ordered graph in orderingArray.

    Parameters
    ----------
    cliques : int**
        Integer array [nCliques x max_k] to be populated with the vertices of
        all k-cliques, only if enumerate is 1.
    cliqueCounts : int*
        Zero-valued array of length max_k + 1 in which the clique counts are to
        be stored.
    orderingArray : NeighborListArray**
        An array of linked lists of integers (the adjacency list representation
        of the graph).
    n : int
        The number of vertices in the graph.
    deg : int
        The degeneracy of the graph.
    max_k : int
        The maximum size of k-cliques to be counted.
    enumerate : int
        1 if cliques are to be enumerated, 0 if cliques are to be counted.
    """
    # Vertex sets are stored in an array as follows: |--X--|--P--|--R--|
    # Vertices are moved to R during the recursion thru the succinct clique tree
    cdef int* vertexSets = <int*> calloc(n, sizeof(int))
    # vertex i is stored in vertexSets[vertexLookup[i]]
    cdef int* vertexLookup = <int*> calloc(n, sizeof(int))
    cdef int** neighborsInP = <int**> calloc(n * max_k, sizeof(int))
    cdef int* numNeighbors = <int*> calloc(n, sizeof(int))
    cdef int* isHold = <int*> calloc(n, sizeof(int))
    cdef int* isPivot = <int*> calloc(n, sizeof(int))

    cdef int i = 0

    while i < n:
        vertexLookup[i] = i
        vertexSets[i] = i
        neighborsInP[i] = <int*> calloc(deg, sizeof(int))
        numNeighbors[i] = 1
        i += 1

    cdef int vertex, beginP, beginR, drop, rsize
    cdef int* cliqueNum = <int*> malloc(sizeof(int))
    cliqueNum[0] = 0
    cdef int j
    # for each vertex
    for i in range(n):
        vertex = orderingArray[i].vertex
        # set P to be later neighbors and X to be earlier neighbors of vertex
        fillInPandXForRecursiveCall(i, vertex, vertexSets, vertexLookup,
                                    orderingArray, neighborsInP, numNeighbors,
                                    n - 1, &beginP, &beginR)

        # recursively compute maximal cliques containing a vertex, some of its
        # later neighbors, and none of its earlier neighbors
        drop = 0
        rsize = 1
        isHold[vertex] = 1 # label the first vertex in the branch as a hold

        listAllCliquesDegeneracyRecursive(cliques, cliqueCounts,
                                          vertexSets, vertexLookup,
                                          neighborsInP, numNeighbors, isHold,
                                          isPivot, beginP, beginR, n, max_k,
                                          rsize, drop, cliqueNum, enumerate)

        isHold[vertex] = 0 # reset the state of isHold for future iterations

    cliqueCounts[0] = 1 # account for empty set as a clique

    return


cdef int* countCliques_c(LinkedList** adjListLinked, int n, int* max_k, 
                         int* deg):
    """Given an adjacency list of a graph as an array of linked lists of
       integers, output the number of k-cliques in the graph as an int array.

    Parameters
    ----------
    adjListLinked : LinkedList**
        An array of linked lists of integers (the adjacency list representation
        of the graph).
    n : int
        The number of vertices in the graph.
    max_k : int*
        Pointer to int providing the maximum size of k-cliques to be counted,
        or 0 if all k-cliques should be counted.  In the latter case, max_k[0]
        will be updated to reflect the maximum clique size that is found.
    deg : int*
        Pointer to int that will store the degeneracy of the graph.

    Returns
    -------
    int*
        Integer array of k-clique counts, indexed by k <= degeneracy.
    """
    cdef NeighborListArray** orderingArray = \
        computeDegeneracyOrderArray(adjListLinked, n)

    cdef int i
    for i in range(n):
        if deg[0] < orderingArray[i].laterDegree:
            deg[0] = orderingArray[i].laterDegree

    if max_k[0] == 0:
        max_k[0] = deg[0] + 1

    cdef int* cliqueCounts = <int*> calloc(max_k[0] + 1, sizeof(int))
    listAllCliquesDegeneracy(NULL, cliqueCounts, orderingArray, n, 
                             deg[0], max_k[0], 0)
    # pass NULL pointer since the cliques argument will never be used

    free(orderingArray)

    return cliqueCounts


cdef int** enumerateCliques_c(LinkedList** adjListLinked, 
                              int n, int* max_k, int* nCliques):
    """Given an adjacency list of a graph as an array of linked lists of
       integers, output the vertices of k-cliques in the graph as an int array.

    Parameters
    ----------
    adjListLinked : LinkedList**
        An array of linked lists of integers (the adjacency list representation
        of the graph).
    n : int
        The number of vertices in the graph.
    max_k : int*
        Pointer to int providing the maximum size of k-cliques to be counted,
        or 0 if all k-cliques should be counted.  In the latter case, max_k[0]
        will be updated to reflect the maximum clique size that is found.
    nCliques : int*
        Pointer to int that will store the number of cliques that is determined
        by the function.

    Returns
    -------
    int**
        Integer array [nCliques x max_k] of k-cliques with -1 as a placeholder
        for cliques of length less than max_k.
    """
    cdef int deg = 0
    cdef int* cliqueCounts = countCliques_c(adjListLinked, n, max_k, &deg)
    
    cdef NeighborListArray** orderingArray = \
        computeDegeneracyOrderArray(adjListLinked, n)
   
    cdef int new_max_k
    for i in range(max_k[0] + 1):
        nCliques[0] += cliqueCounts[i]
        if cliqueCounts[i] > 0:
            new_max_k = i 

    cdef int** cliques = <int**> calloc(nCliques[0] * max_k[0], sizeof(int))

    cdef int j
    for i in range(nCliques[0]):
        cliques[i] = <int*> calloc(max_k[0], sizeof(int))
        for j in range(max_k[0]):
            cliques[i][j] = -1

    listAllCliquesDegeneracy(cliques, cliqueCounts, orderingArray, n, 
                             deg, max_k[0], 1)

    max_k[0] = new_max_k

    free(orderingArray)
    free(cliqueCounts)

    return cliques


cpdef countCliques(adj, int max_k):
    """Given an adjacency list of a graph as an n x 2 NumPy array of adjacent
       vertex pairs, output the number of k-cliques as a NumPy array.

    Parameters
    ----------
    adj : np.array [m x 2]
        NumPy array of adjacent vertex pairs in a graph, with no repeats,
        reversals, or self-adjacency.
    max_k : int
        The maximum size of k-cliques to be counted, or 0 if all k-cliques
        should be counted.

    Returns
    -------
    cliqueCounts : np.array [max_k + 1]
        Numpy array of clique counts for k = 0 up to max_k.
    """
    cdef int n = np.max(adj) + 1 # number of vertices in the graph
    cdef int m = len(adj) # number of edges in the graph
    # create C-style int array from numpy array
    cdef int adj_len = len(adj)
    cdef int* adj_c[2]
    adj_c[0] = <int*> malloc(adj_len * sizeof(int))
    adj_c[1] = <int*> malloc(adj_len * sizeof(int))
    cdef int i
    for i in range(adj_len):
        adj_c[0][i] = <int> adj[i][0]
        adj_c[1][i] = <int> adj[i][1]

    cdef LinkedList** adjacencyList = graphAdjArrayToDoubleEdges(n, m, adj_c)
    cdef int deg = 0
    cliqueCounts = countCliques_c(adjacencyList, n, &max_k, &deg)

    # create numpy array from C-style int array for returning the counts
    cliqueCounts_np = np.zeros(max_k + 1).astype(np.dtype("i"))
    for i in range(max_k + 1):
        cliqueCounts_np[i] = cliqueCounts[i]
    return np.trim_zeros(cliqueCounts_np)


cpdef enumerateCliques(adj, int max_k):
    """Given an adjacency list of a graph as an n x 2 NumPy array of adjacent
       vertex pairs, output the number of k-cliques as a NumPy array.

    Parameters
    ----------
    adj : np.array [m x 2]
        NumPy array of adjacent vertex pairs in a graph, with no repeats,
        reversals, or self-adjacency.
    max_k : int
        The maximum size of k-cliques to be counted, or 0 if all k-cliques
        should be counted.

    Returns
    -------
    k_cliques : list
        A list of NumPy arrays containing the indices of the k-cliques for each
        value of k.
    """
    cdef int n = np.max(adj) + 1 # number of vertices in the graph
    cdef int m = len(adj) # number of edges in the graph
    # create C-style int array from numpy array
    cdef int adj_len = len(adj)
    cdef int* adj_c[2]
    adj_c[0] = <int*> malloc((adj_len + 1) * sizeof(int))
    adj_c[1] = <int*> malloc((adj_len + 1) * sizeof(int))
    cdef int i
    for i in range(adj_len):
        adj_c[0][i] = <int> adj[i][0]
        adj_c[1][i] = <int> adj[i][1]

    cdef LinkedList** adjacencyList = graphAdjArrayToDoubleEdges(n, m, adj_c)
    cdef int nCliques = 0
    cliques = enumerateCliques_c(adjacencyList, n, &max_k, &nCliques)

    # create numpy array from C-style int array for returning the cliques
    cliques_np = np.zeros((nCliques, max_k)).astype(np.dtype("i"))
    cdef int j
    for i in range(nCliques):
        for j in range(max_k):
            cliques_np[i][j] = cliques[i][j]
    k_cliques = [cliques_np[np.logical_and(cliques_np[:, k] == -1,
                                           cliques_np[:, k-1] != -1)][:, :k]
                 for k in range(max_k)] + \
                [cliques_np[cliques_np[:, max_k-1] != -1]]
    return k_cliques
