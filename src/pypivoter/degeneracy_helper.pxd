cdef struct s_Link:
    int data # arbitrary data stored in the link
    s_Link* next # the previous link in the chain
    s_Link* prev # the next link in the chain

cdef struct s_LinkedList:
    s_Link* head # head of the linked list, a dummy sentinel
    s_Link* tail # tail of the linked list, a dummy sentinel

ctypedef s_Link Link
ctypedef s_LinkedList LinkedList

cdef struct s_NeighborList:
    int vertex # the vertex that owns this neighbor list
    LinkedList* earlier # a list of neighbors that are ordered before the vertex
    LinkedList* later # a list of neighbors that are ordered after the vertex
    int orderNumber # the position of the vertex in the ordering

cdef struct s_NeighborListArray:
    int vertex # the vertex that owns this neighbor list
    int* earlier # an array of neighbors that are ordered before the vertex
    int earlierDegree # the number of neighbors in earlier
    int* later # an array of neighbors that are ordered after the vertex
    int laterDegree # the number of neighbors in later
    int orderNumber # the position of the vertex in the ordering

ctypedef s_NeighborList NeighborList
ctypedef s_NeighborListArray NeighborListArray

cdef int isHead(Link* list)
cdef int isTail(Link* list)
cdef Link* addAfter(Link* list, int data)
cdef Link* addBefore(Link* list, int data)
cdef Link* removeLink(Link* list)
cdef int deleteLink(Link* list)

cdef LinkedList* createLinkedList()
cdef void destroyLinkedList(LinkedList* linkedList)
cdef void copyLinkedList(LinkedList* destination, LinkedList* source)
cdef Link* addFirst(LinkedList* linkedList, int data)
cdef Link* addLast(LinkedList* linkedList, int data)
cdef int isEmpty(LinkedList* linkedList)
cdef int getFirst(LinkedList* linkedList)
cdef Link* removeFirst(LinkedList* linkedList)
cdef Link* removeLast(LinkedList* linkedList)
cdef void deleteLast(LinkedList* linkedList)
cdef int length(LinkedList* linkedList)

cdef int computeDegeneracy(LinkedList** list, int size)
cdef NeighborList** computeDegeneracyOrderList(LinkedList** list, int size)
cdef NeighborListArray** computeDegeneracyOrderArray(LinkedList** list, int size)
