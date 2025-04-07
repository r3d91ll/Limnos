from typing import Dict, Any, Hashable, Optional, Union, overload
import networkx as nx

# Full definition with correct default values
def best_partition(
    graph: nx.Graph,
    partition: Optional[Dict[Any, int]] = None,
    weight: str = 'weight',
    resolution: float = 1.,
    randomize: bool = False,
    random_state: Optional[int] = None
) -> Dict[Hashable, int]:
    """
    Type stub for community.best_partition function
    
    Compute the partition of the graph using the Louvain algorithm.
    
    Returns:
        Dict mapping node to community id
    """
    ...
