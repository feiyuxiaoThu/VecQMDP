# Copyright (c) 2026 VecQMDP Contributors.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import deque
from typing import Dict, List, Optional, Tuple, Union

from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject

class BreadthFirstSearchRoadBlock:
    """
    A class that performs iterative breadth first search. The class operates on the roadblock graph.
    """

    def __init__(
        self,
        start_roadblock_id: int,
        map_api: Optional[AbstractMap],
        forward_search: str = True,
    ):
        """
        Initialize BFS from a starting roadblock.

        Args:
            start_roadblock_id: roadblock id where the graph search starts.
            map_api: nuPlan map API instance.
            forward_search: whether to search in driving direction.
        """
        self._map_api: Optional[AbstractMap] = map_api
        self._queue = deque([self.id_to_roadblock(start_roadblock_id), None])
        self._parent: Dict[str, Optional[RoadBlockGraphEdgeMapObject]] = dict()
        self._forward_search = forward_search

        #  lazy loaded
        self._target_roadblock_ids: List[str] = None

    def search(
        self, target_roadblock_id: Union[str, List[str]], max_depth: int
    ) -> Tuple[List[RoadBlockGraphEdgeMapObject], bool]:
        """
        Apply BFS to find a route to the target roadblock.

        Args:
            target_roadblock_id: id or list of ids of target roadblocks.
            max_depth: maximum search depth.

        Returns:
            Tuple[Tuple[List[RoadBlockGraphEdgeMapObject], List[str]], bool]:
                (path, path_ids) and whether a path was found.
        """

        if isinstance(target_roadblock_id, str):
            target_roadblock_id = [target_roadblock_id]
        self._target_roadblock_ids = target_roadblock_id

        start_edge = self._queue[0]

        # Initial search states
        path_found: bool = False
        end_edge: RoadBlockGraphEdgeMapObject = start_edge
        end_depth: int = 1
        depth: int = 1

        self._parent[start_edge.id + f"_{depth}"] = None

        while self._queue:
            current_edge = self._queue.popleft()

            # Early exit condition
            if self._check_end_condition(depth, max_depth):
                break

            # Depth tracking
            if current_edge is None:
                depth += 1
                self._queue.append(None)
                if self._queue[0] is None:
                    break
                continue

            # Goal condition
            if self._check_goal_condition(current_edge, depth, max_depth):
                end_edge = current_edge
                end_depth = depth
                path_found = True
                break

            neighbors = (
                current_edge.outgoing_edges
                if self._forward_search
                else current_edge.incoming_edges
            )

            # Populate queue
            for next_edge in neighbors:
                # if next_edge.id in self._candidate_lane_edge_ids_old:
                self._queue.append(next_edge)
                self._parent[next_edge.id + f"_{depth + 1}"] = current_edge
                end_edge = next_edge
                end_depth = depth + 1

        return self._construct_path(end_edge, end_depth), path_found

    def id_to_roadblock(self, id: str) -> RoadBlockGraphEdgeMapObject:
        """
        Retrieve a roadblock or roadblock connector from the map API.

        Args:
            id: roadblock id.

        Returns:
            RoadBlockGraphEdgeMapObject: the corresponding roadblock object.
        """
        block = self._map_api._get_roadblock(id)
        block = block or self._map_api._get_roadblock_connector(id)
        return block

    @staticmethod
    def _check_end_condition(depth: int, max_depth: int) -> bool:
        """
        Check if the search should terminate regardless of goal condition.

        Args:
            depth: current search depth.
            max_depth: maximum allowed depth.

        Returns:
            bool: True if depth exceeds max_depth.
        """
        return depth > max_depth

    def _check_goal_condition(
        self,
        current_edge: RoadBlockGraphEdgeMapObject,
        depth: int,
        max_depth: int,
    ) -> bool:
        """
        Check if the current edge matches a target roadblock within depth limit.

        Args:
            current_edge: edge to check.
            depth: current search depth.
            max_depth: maximum allowed depth.

        Returns:
            bool: True if the edge is a target roadblock and depth <= max_depth.
        """
        return current_edge.id in self._target_roadblock_ids and depth <= max_depth

    def _construct_path(
        self, end_edge: RoadBlockGraphEdgeMapObject, depth: int
    ) -> List[RoadBlockGraphEdgeMapObject]:
        """
        Back-propagate from the end edge to construct the full path.

        Args:
            end_edge: terminal edge to trace back from.
            depth: depth of the terminal edge.

        Returns:
            Tuple[List[RoadBlockGraphEdgeMapObject], List[str]]:
                ordered path of roadblocks and their ids.
        """
        path = [end_edge]
        path_id = [end_edge.id]

        while self._parent[end_edge.id + f"_{depth}"] is not None:
            path.append(self._parent[end_edge.id + f"_{depth}"])
            path_id.append(path[-1].id)
            end_edge = self._parent[end_edge.id + f"_{depth}"]
            depth -= 1

        if self._forward_search:
            path.reverse()
            path_id.reverse()

        return (path, path_id)
