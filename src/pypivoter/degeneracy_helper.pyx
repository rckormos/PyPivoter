from libc.stdlib cimport calloc, malloc, free

# This code is a modified version of the code of quick-cliques-1.0 library for
# counting maximal cliques by Darren Strash (first name DOT last name AT gmail
# DOT com).

# Original author: Darren Strash (first name DOT last name AT gmail DOT com)

# Copyright (c) 2011 Darren Strash. This code is released under the GNU Public
# License (GPL) 3.0.

# Modification copyright (c) 2020 Shweta Jain
# Modification copyright (c) 2022 Rian Kormos

### The following structs provide the basic data structures of links and lists.

# Stores data and pointers to next and previous links.
cdef struct s_Link:
    int data # arbitrary data stored in the link
    s_Link* next # the previous link in the chain
    s_Link* prev # the next link in the chain

# Stores a linked list, with sentinel links for head and tail.
cdef struct s_LinkedList:
    s_Link* head # head of the linked list, a dummy sentinel
    s_Link* tail # tail of the linked list, a dummy sentinel


### The next several functions operate on Link structures.

cdef int isHead(Link* list):
    """Determine if a Link is the head sentinel.

    Parameters
    ----------
    list : Link
        The link structure to be determined whether it is the head.

    Returns
    -------
    bool
        True if the link is the head sentinel, false otherwise.
    """
    assert list != NULL
    return list.prev == NULL


cdef int isTail(Link* list):
    """Determine if a Link is the tail sentinel.

    Parameters
    ----------
    list : Link
        The link structure to be determined whether it is the tail.

    Returns
    -------
    bool
        True if the link is the tail sentinel, false otherwise.
    """
    assert list != NULL
    return list.next == NULL


cdef Link* addAfter(Link* list, int data):
    """A location-aware function to add a link after a given link.

    Parameters
    ----------
    list : Link*
        The link after which to add the data.
    data : int
        A piece of data to put in the added link.

    Returns
    -------
    newLink : Link*
        A pointer to the Link that was added after link.
    """
    assert list != NULL
    assert list.next != NULL

    cdef Link* newLink = <Link*> malloc(sizeof(Link))

    newLink.data = data

    newLink.next = list.next
    newLink.prev = list

    list.next.prev = newLink
    list.next = newLink

    return newLink


cdef Link* addBefore(Link* list, int data):
    """A location-aware function to add a link before a given link.

    Parameters
    ----------
    list : Link*
        The link before which to add the data.
    data : int
        A piece of data to put in the added link.

    Returns
    -------
    newLink : Link*
        A pointer to the Link that was added before link.
    """
    assert list != NULL
    assert list.prev != NULL

    cdef Link* newLink = <Link*> malloc(sizeof(Link))

    newLink.data = data

    newLink.next = list
    newLink.prev = list.prev

    list.prev.next = newLink
    list.prev = newLink

    return newLink


cdef Link* removeLink(Link* list):
    """A location-aware method to remove a link and return it.

    Parameters
    ----------
    list : Link*
        The link to be removed and returned.

    Returns
    -------
    Link*
        The removed link.
    """
    assert list != NULL
    assert list.next != NULL
    assert list.prev != NULL

    list.next.prev = list.prev
    list.prev.next = list.next

    list.next = NULL
    list.prev = NULL

    return list


cdef int deleteLink(Link* list):
    """Delete the given link and return its data.

    Parameters
    ----------
    list : Link*
        The link to be deleted.

    Returns
    -------
    int
        The data that was in the link (must be freed if previously allocated.)
    """
    assert list != NULL
    assert list.next != NULL
    assert list.prev != NULL

    cdef int data = list.data
    cdef Link* linkToFree = removeLink(list)

    free(linkToFree)

    return data

### The next several functions operate on LinkedList structures.

cdef LinkedList* createLinkedList():
    """Create a new empty linked list.

    Returns
    -------
    LinkedList*
        The created linked list.
    """
    cdef LinkedList* linkedList = <LinkedList*> malloc(sizeof(LinkedList))

    linkedList.head = <Link *> malloc(sizeof(Link))
    linkedList.tail = <Link *> malloc(sizeof(Link))

    linkedList.head.prev = NULL
    linkedList.head.next = linkedList.tail
    linkedList.head.data = <int> 0xDEAD0000

    linkedList.tail.prev = linkedList.head
    linkedList.tail.next = NULL
    linkedList.tail.data = <int> 0xDEADFFFF

    return linkedList


cdef void destroyLinkedList(LinkedList* linkedList):
    """Destroy a linked list.

    Parameters
    ----------
    linkedList : LinkedList*
        The linked list to destroy.
    """
    cdef Link* curr = linkedList.head
    cdef Link* currNext

    while curr != NULL:
        currNext = curr.next
        free(curr)
        curr = currNext

    free(linkedList)


cdef void copyLinkedList(LinkedList* destination, LinkedList* source):
    """Copy a linked list.

    Parameters
    ----------
    destination : LinkedList*
        Location at which to copy the linked list.
    source : LinkedList*
        Linked list to be copied.
    """
    assert destination != NULL and source != NULL

    cdef Link* curr = source.head.next

    while not isTail(curr):
        addLast(destination, curr.data)
        curr = curr.next

cdef Link* addFirst(LinkedList* linkedList, int data):
    """A location-aware function to add data to the beginning of a linked list.

    Parameters
    ----------
    linkedList : LinkedList*
        The linked list to the beginning of which the data is to be added.
    data : int
        The data to be added at the beginning of the linked list.

    Returns
    -------
    Link*
        The link where the data was placed in the linked list.
    """
    assert linkedList != NULL
    return addAfter(linkedList.head, data)


cdef Link* addLast(LinkedList* linkedList, int data):
    """A location-aware function to add data to the end of a linked list.

    Parameters
    ----------
    linkedList : LinkedList*
        The linked list to the end of which the data is to be added.
    data : int
        The data to be added at the end of the linked list.

    Returns
    -------
    Link*
        The link where the data was placed in the linked list.
    """
    assert linkedList != NULL
    return addBefore(linkedList.tail, data)


cdef int isEmpty(LinkedList* linkedList):
    """Determine if a linked list is empty.

    Parameters
    ----------
    linkedList : LinkedList*
        The linked list to determine if empty.

    Returns
    -------
    int
        Non-zero if the linked list is empty, zero otherwise.
    """
    assert linkedList != NULL
    return isTail(linkedList.head.next)


cdef int getFirst(LinkedList* linkedList):
    """Return the first piece of data in a linked list.

    Parameters
    ----------
    linkedList : LinkedList*
        The linked list from which to return the first piece of data.

    Returns
    -------
    int
        The data in the first link of the linked list.
    """
    assert linkedList != NULL
    assert not isEmpty(linkedList)
    return linkedList.head.next.data


cdef Link* removeFirst(LinkedList* linkedList):
    """Remove and return the first link from a linked list.

    Parameters
    ----------
    linkedList : LinkedList*
        The linked list from which to remove the first link.

    Returns
    -------
    Link*
        The first link of the linked list.
    """
    assert linkedList != NULL

    if not isEmpty(linkedList):
        return removeLink(linkedList.head.next)
    return NULL


cdef Link* removeLast(LinkedList* linkedList):
    """Remove and return the last link from a linked list.

    Parameters
    ----------
    linkedList : LinkedList*
        The linked list from which to remove the last link.

    Returns
    -------
    Link*
        The last link of the linked list.
    """
    assert linkedList != NULL

    if not isEmpty(linkedList):
        return removeLink(linkedList.tail.prev)
    return NULL


cdef void deleteLast(LinkedList* linkedList):
    """Delete the last link in a linked list.

    Parameters
    ----------
    linkedList : LinkedList*
        The linked list from which to delete the last link.
    """
    assert linkedList != NULL
    if not isEmpty(linkedList):
        deleteLink(linkedList.tail.prev)
    return


cdef int length(LinkedList* linkedList):
    """Compute the number of data elements in a linked list.

    Parameters
    ----------
    linkedList : LinkedList*
        The linked list for which to compute the number of data elements.

    Returns
    -------
    int
        The number of data elements in the linked list.
    """
    cdef int length = 0
    cdef Link* curr = linkedList.head.next

    while not isTail(curr):
        length += 1
        curr = curr.next

    return length

### The following structs provide the basic data structures of neighbor lists.

# For a given ordering, this struct stores later and earlier neighbors for a
# given vertex in linked lists
cdef struct s_NeighborList:
    int vertex # the vertex that owns this neighbor list
    LinkedList* earlier # a list of neighbors that are ordered before the vertex
    LinkedList* later # a list of neighbors that are ordered after the vertex
    int orderNumber # the position of the vertex in the ordering

# For a given ordering, this struct stores later and earlier neighbors for a
# given vertex in arrays
cdef struct s_NeighborListArray:
    int vertex # the vertex that owns this neighbor list
    int* earlier # an array of neighbors that are ordered before the vertex
    int earlierDegree # the number of neighbors in earlier
    int* later # an array of neighbors that are ordered after the vertex
    int laterDegree # the number of neighbors in later
    int orderNumber # the position of the vertex in the ordering


cdef int computeDegeneracy(LinkedList** list, int size):
    """Compute the degeneracy (maximum number of edges per node) of a graph.

    Parameters
    ----------
    list : LinkedList**
        An input graph, represented as an array of linked lists of integers.
    size : int
        The number of vertices in the graph.

    Returns
    -------
    int
        The degeneracy of the graph.
    """
    cdef int i = 0
    cdef degeneracy =0

    # array of lists of vertices, indexed by degree
    cdef LinkedList** verticesByDegree = \
        <LinkedList**> calloc(size, sizeof(LinkedList*))
    # array of lists of vertices that are first among those of their degree
    cdef Link** vertexLocator = <Link**> calloc(size, sizeof(Link*))

    cdef int* degree = <int*> calloc(size, sizeof(int))

    for i in range(size):
        verticesByDegree[i] = createLinkedList()

    # fill each cell of the degree lookup table, then use that degree to
    # populate the lists of vertices indexed by degree
    for i in range(size):
        degree[i] = length(list[i])
        vertexLocator[i] = addFirst(verticesByDegree[degree[i]], i)

    cdef int currentDegree = 0
    cdef int numVerticesRemoved = 0
    cdef int vertex
    cdef int neighbor
    cdef LinkedList* neighborList
    cdef Link* neighborLink

    while numVerticesRemoved < size:
        if not isEmpty(verticesByDegree[currentDegree]):
            degeneracy = max(degeneracy, currentDegree)
            vertex = getFirst(verticesByDegree[currentDegree])
            deleteLink(vertexLocator[vertex])
            degree[vertex] = -1

            neighborList = list[vertex]
            neighborLink = neighborList.head.next

            while not isTail(neighborLink):
                neighbor = neighborLink.data
                if degree[neighbor] != -1:
                    deleteLink(vertexLocator[neighbor])
                    degree[neighbor] -= 1

                    if degree[neighbor] != -1:
                        vertexLocator[neighbor] = \
                            addFirst(verticesByDegree[degree[neighbor]],
                                     neighbor)
                neighborLink = neighborLink.next
            numVerticesRemoved += 1
            currentDegree = 0
        else:
            currentDegree += 1

    for i in range(size):
        destroyLinkedList(verticesByDegree[i])

    free(vertexLocator)
    free(verticesByDegree)
    free(degree)

    return degeneracy


cdef NeighborList** computeDegeneracyOrderList(LinkedList** list, int size):
    """Find the smallest-last ordering of a graph as an array of NeighborLists.

    Parameters
    ----------
    list : LinkedList**
        An input graph, represented as an array of linked lists of integers.
    size : int
        The number of vertices in the graph.

    Returns
    -------
    NeighborList**
        An array of NeighborLists representing a smallest-last degeneracy
        ordering of the graph vertices.
    """
    cdef NeighborList** ordering = \
        <NeighborList**> calloc(size, sizeof(NeighborList*))
    cdef int i = 0
    cdef int degeneracy = 0
    # array of lists of vertices, indexed by degree
    cdef LinkedList** verticesByDegree = \
        <LinkedList**> calloc(size, sizeof(LinkedList*))
    # array of lists of vertices that are first among those of their degree
    cdef Link** vertexLocator = \
        <Link**> calloc(size, sizeof(Link*))

    cdef int* degree = <int*> calloc(size, sizeof(int))

    for i in range(size):
        verticesByDegree[i] = createLinkedList()
        ordering[i] = <NeighborList*> malloc(sizeof(NeighborList))
        ordering[i].earlier = createLinkedList()
        ordering[i].later = createLinkedList()

    # fill each cell of the degree lookup table, then use that degree to
    # populate the lists of vertices indexed by degree
    for i in range(size):
        degree[i] = length(list[i])
        vertexLocator[i] = addFirst(verticesByDegree[degree[i]], i)

    cdef int currentDegree = 0
    cdef int numVerticesRemoved = 0
    cdef int vertex
    cdef int neighbor
    cdef LinkedList* neighborList
    cdef Link* neighborLink

    while numVerticesRemoved < size:
        if not isEmpty(verticesByDegree[currentDegree]):
            degeneracy = max(degeneracy, currentDegree)
            vertex = getFirst(verticesByDegree[currentDegree])

            deleteLink(vertexLocator[vertex]) # delete lowest-degeneracy vertex

            ordering[vertex].vertex = vertex
            ordering[vertex].orderNumber = numVerticesRemoved

            degree[vertex] = -1

            neighborList = list[vertex]
            neighborLink = neighborList.head.next

            while not isTail(neighborLink):
                neighbor = neighborLink.data

                if degree[neighbor] != -1:
                    deleteLink(vertexLocator[neighbor])
                    addLast(ordering[vertex].later, neighbor)

                    degree[neighbor] -= 1

                    if degree[neighbor] != -1:
                        vertexLocator[neighbor] = \
                            addFirst(verticesByDegree[degree[neighbor]],
                                     neighbor)
                else:
                    addLast(ordering[vertex].earlier, neighbor)

                neighborLink = neighborLink.next
            numVerticesRemoved += 1
            currentDegree = 0
        else:
            currentDegree += 1

    for i in range(size):
        destroyLinkedList(verticesByDegree[i])

    free(vertexLocator)
    free(verticesByDegree)
    free(degree)

    return ordering


cdef NeighborListArray** computeDegeneracyOrderArray(LinkedList** list, int size):
    """Find the smallest-last ordering of a graph as an array of NeighborLists.

    Parameters
    ----------
    list : LinkedList**
        An input graph, represented as an array of linked lists of integers.
    size : int
        The number of vertices in the graph.

    Returns
    -------
    NeighborListArray**
        An array of NeighborListArrays representing a smallest-last degeneracy
        ordering of the graph vertices.
    """
    cdef NeighborList** ordering = \
        <NeighborList**> calloc(size, sizeof(NeighborList*))
    cdef int i = 0
    cdef int degeneracy = 0
    # array of lists of vertices, indexed by degree
    cdef LinkedList** verticesByDegree = \
        <LinkedList**> calloc(size, sizeof(LinkedList*))
    # array of lists of vertices that are first among those of their degree
    cdef Link** vertexLocator = \
        <Link**> calloc(size, sizeof(Link*))

    cdef int* degree = <int*> calloc(size, sizeof(int))

    for i in range(size):
        verticesByDegree[i] = createLinkedList()
        ordering[i] = <NeighborList*> malloc(sizeof(NeighborList))
        ordering[i].earlier = createLinkedList()
        ordering[i].later = createLinkedList()

    # fill each cell of the degree lookup table, then use that degree to
    # populate the lists of vertices indexed by degree
    for i in range(size):
        degree[i] = length(list[i])
        vertexLocator[i] = addFirst(verticesByDegree[degree[i]], i)

    cdef int currentDegree = 0
    cdef int numVerticesRemoved = 0
    cdef int vertex
    cdef int neighbor
    cdef LinkedList* neighborList
    cdef Link* neighborLink

    while numVerticesRemoved < size:
        if not isEmpty(verticesByDegree[currentDegree]):
            degeneracy = max(degeneracy, currentDegree)
            vertex = getFirst(verticesByDegree[currentDegree])

            deleteLink(vertexLocator[vertex]) # delete lowest-degeneracy vertex

            ordering[vertex].vertex = vertex
            ordering[vertex].orderNumber = numVerticesRemoved

            degree[vertex] = -1

            neighborList = list[vertex]
            neighborLink = neighborList.head.next

            while not isTail(neighborLink):
                neighbor = neighborLink.data

                if degree[neighbor] != -1:
                    deleteLink(vertexLocator[neighbor])
                    addLast(ordering[vertex].later, neighbor)

                    degree[neighbor] -= 1

                    if degree[neighbor] != -1:
                        vertexLocator[neighbor] = \
                            addFirst(verticesByDegree[degree[neighbor]],
                                     neighbor)
                else:
                    addLast(ordering[vertex].earlier, neighbor)

                neighborLink = neighborLink.next
            numVerticesRemoved += 1
            currentDegree = 0
        else:
            currentDegree += 1

    cdef NeighborListArray** orderingArray = \
        <NeighborListArray**> calloc(size, sizeof(NeighborListArray))
    cdef int j
    cdef Link* curr

    for i in range(size):
        orderingArray[i] = \
            <NeighborListArray*> malloc(sizeof(NeighborListArray))
        orderingArray[i].vertex = ordering[i].vertex
        orderingArray[i].orderNumber = ordering[i].orderNumber
        orderingArray[i].laterDegree = length(ordering[i].later)
        orderingArray[i].later = \
            <int*> calloc(orderingArray[i].laterDegree, sizeof(int))

        j = 0
        curr = ordering[i].later.head.next # get first link in ordering
        while not isTail(curr):
            orderingArray[i].later[j] = curr.data
            curr = curr.next
            j += 1

        orderingArray[i].earlierDegree = length(ordering[i].earlier)
        orderingArray[i].earlier = \
            <int*> calloc(orderingArray[i].earlierDegree, sizeof(int))

        j = 0
        curr = ordering[i].earlier.head.next
        while not isTail(curr):
            orderingArray[i].earlier[j] = curr.data
            curr = curr.next
            j += 1

    for i in range(size):
        free(ordering[i])
        destroyLinkedList(verticesByDegree[i])

    free(vertexLocator)
    free(verticesByDegree)
    free(degree)

    return orderingArray
