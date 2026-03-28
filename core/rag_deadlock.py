"""
rag_deadlock.py
---------------
Resource Allocation Graph (RAG) for deadlock detection.

Graph structure
---------------
  Request edge  : process  → resource   (process is WAITING for resource)
  Assignment edge: resource → process   (resource is HELD by process)

Deadlock exists if and only if there is a cycle in this directed graph.

A cycle requires at minimum:
  P_A holds R_X  →  R_X → P_A
  P_A waits for R_Y  →  P_A → R_Y
  P_B holds R_Y  →  R_Y → P_B
  P_B waits for R_X  →  P_B → R_X
  Cycle: P_A → R_Y → P_B → R_X → P_A
"""

from collections import defaultdict


class ResourceAllocationGraph:

    def __init__(self):
        # adjacency list: node → [neighbours]
        self.graph = defaultdict(list)

        # track what each process holds and wants (for explanation)
        self.held_by   = defaultdict(list)   # resource → [processes holding it]
        self.waited_by = defaultdict(list)   # resource → [processes waiting for it]

    def add_request(self, process, resource):
        """
        Add a REQUEST edge: process is WAITING for resource.
        process → resource
        """
        self.graph[process].append(resource)
        self.waited_by[resource].append(process)

    def add_assignment(self, resource, process):
        """
        Add an ASSIGNMENT edge: resource is HELD by process.
        resource → process
        """
        self.graph[resource].append(process)
        self.held_by[resource].append(process)

    def detect_cycle(self):
        """
        DFS-based cycle detection.
        Returns True if a cycle exists (deadlock), False otherwise.
        """
        visited = set()
        stack   = set()

        def dfs(node):
            if node in stack:
                return True
            if node in visited:
                return False

            visited.add(node)
            stack.add(node)

            for neighbour in self.graph.get(node, []):
                if dfs(neighbour):
                    return True

            stack.remove(node)
            return False

        for node in list(self.graph.keys()):
            if node not in visited:
                if dfs(node):
                    return True

        return False

    def get_cycles(self):
        """
        Return all simple cycles in the graph (for UI display).
        Uses networkx internally.
        """
        try:
            import networkx as nx
            G = nx.DiGraph()
            for node, neighbours in self.graph.items():
                for nb in neighbours:
                    G.add_edge(node, nb)
            return list(nx.simple_cycles(G))
        except ImportError:
            return []


def build_rag_from_processes(processes):
    """
    Build a RAG correctly from a process list.

    Each process dict must contain:
        pid              : int
        resource         : str   e.g. "R2"   ← resource this process HOLDS
        wanted_resource  : str   e.g. "R3"   ← resource this process WANTS
                                               (optional — if absent, no
                                                request edge is added)

    Rules
    -----
    - Assignment edge added for every process (it holds its resource).
    - Request edge added only if wanted_resource exists AND
      wanted_resource != resource (a process waiting for what it
      already holds is not a real wait).
    """
    rag = ResourceAllocationGraph()

    for p in processes:
        pid      = p['pid']
        held     = str(p['resource'])
        wanted   = str(p.get('wanted_resource', ''))

        # Assignment: resource is held by this process
        rag.add_assignment(held, pid)

        # Request: process is waiting for a DIFFERENT resource
        if wanted and wanted != held:
            rag.add_request(pid, wanted)

    return rag