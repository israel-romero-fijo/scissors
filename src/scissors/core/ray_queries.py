"""Deterministic NumPy BVH for bounded, double-sided finite ray segments."""

import heapq
import time
from dataclasses import dataclass

import numpy as np

from scissors.api.errors import ScissorsError


@dataclass
class WorkBudget:
    limit: int
    deadline: float
    triangle_tests: int = 0
    node_tests: int = 0

    def consume(self, count: int = 0, nodes: int = 0) -> None:
        if self.triangle_tests + count > self.limit or self.node_tests + nodes > 2 * self.limit:
            raise ScissorsError("RESOURCE_LIMIT", "Visibility ray work budget exhausted")
        if time.monotonic() > self.deadline:
            raise ScissorsError("OPERATION_TIMEOUT", "Visibility ray query timed out")
        self.triangle_tests += count
        self.node_tests += nodes


@dataclass
class Node:
    low: np.ndarray
    high: np.ndarray
    start: int
    end: int
    left: int = -1
    right: int = -1


def box_entry(
    origin: np.ndarray,
    direction: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    near: float,
    far: float,
    tolerance: float,
) -> float | None:
    """Inclusive slab intersection, including parallel axis directions."""
    entry, exit = near, far
    for axis in range(3):
        if abs(direction[axis]) < 1e-300:
            if origin[axis] < low[axis] - tolerance or origin[axis] > high[axis] + tolerance:
                return None
            continue
        first = (low[axis] - tolerance - origin[axis]) / direction[axis]
        second = (high[axis] + tolerance - origin[axis]) / direction[axis]
        entry, exit = max(entry, min(first, second)), min(exit, max(first, second))
        if entry > exit:
            return None
    return float(entry)


class TriangleBVH:
    def __init__(self, triangles: np.ndarray, deadline: float):
        self.triangles = triangles
        self.order = np.arange(len(triangles), dtype=np.int64)
        self.nodes: list[Node] = []
        self.lows = triangles.min(axis=1)
        self.highs = triangles.max(axis=1)
        centers = (self.lows + self.highs) / 2

        def build(start: int, end: int) -> int:
            if time.monotonic() > deadline:
                raise ScissorsError("OPERATION_TIMEOUT", "Visibility BVH construction timed out")
            selected = self.order[start:end]
            node = Node(
                self.lows[selected].min(axis=0), self.highs[selected].max(axis=0), start, end
            )
            index = len(self.nodes)
            self.nodes.append(node)
            if end - start > 16:
                axis = int(np.argmax(np.ptp(centers[selected], axis=0)))
                self.order[start:end] = selected[np.argsort(centers[selected, axis], kind="stable")]
                middle = (start + end) // 2
                node.left, node.right = build(start, middle), build(middle, end)
            return index

        if len(triangles):
            build(0, len(triangles))

    def query(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        near: float,
        far: float,
        tolerance: float,
        budget: WorkBudget,
    ) -> tuple[str, int | None, float | None]:
        if not self.nodes:
            budget.consume()
            return "clear", None, None
        budget.consume(nodes=1)
        first = box_entry(
            origin, direction, self.nodes[0].low, self.nodes[0].high, near, far, tolerance
        )
        if first is None:
            return "clear", None, None
        pending = [(first, 0)]
        closest, hit = far, None
        ambiguous_entry = float("inf")
        while pending:
            entry, index = heapq.heappop(pending)
            if hit is not None and entry > closest + tolerance:
                continue
            node = self.nodes[index]
            budget.consume()
            if node.left >= 0:
                for child in (node.left, node.right):
                    budget.consume(nodes=1)
                    value = self.nodes[child]
                    bound = box_entry(
                        origin, direction, value.low, value.high, near, closest, tolerance
                    )
                    if bound is not None:
                        heapq.heappush(pending, (bound, child))
                continue
            selected = self.order[node.start : node.end]
            budget.consume(count=len(selected))
            triangles = self.triangles[selected]
            edge1, edge2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            cross = np.cross(direction, edge2)
            determinant = np.einsum("ij,ij->i", edge1, cross)
            normals = np.cross(edge1, edge2)
            lengths = np.linalg.norm(normals, axis=1)
            parallel = np.abs(determinant) <= 1e-12 * lengths
            offset = origin - triangles[:, 0]
            inverse = np.divide(1.0, determinant, out=np.zeros_like(determinant), where=~parallel)
            u = np.einsum("ij,ij->i", offset, cross) * inverse
            q = np.cross(offset, edge1)
            v = (q @ direction) * inverse
            distance = np.einsum("ij,ij->i", edge2, q) * inverse
            valid = ~parallel & (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12)
            valid &= (distance >= near) & (distance <= closest)
            for local in np.flatnonzero(valid):
                candidate, triangle_id = float(distance[local]), int(selected[local])
                if hit is None or (candidate, triangle_id) < (closest, hit):
                    closest, hit = candidate, triangle_id
            # Coplanar or near-parallel contact is conservatively unresolved. A known
            # earlier hit can still prove the nearest obstruction, beyond these boxes.
            plane_at_start = np.einsum("ij,ij->i", offset + near * direction, normals)
            plane_at_end = np.einsum("ij,ij->i", offset + far * direction, normals)
            uncertain = parallel & (np.minimum(plane_at_start, plane_at_end) <= tolerance * lengths)
            uncertain &= np.maximum(plane_at_start, plane_at_end) >= -tolerance * lengths
            for local in np.flatnonzero(uncertain):
                triangle_id = int(selected[local])
                candidate_entry = box_entry(
                    origin,
                    direction,
                    self.lows[triangle_id],
                    self.highs[triangle_id],
                    near,
                    far,
                    tolerance,
                )
                if candidate_entry is not None:
                    ambiguous_entry = min(ambiguous_entry, candidate_entry)
        if ambiguous_entry <= (closest + tolerance if hit is not None else far):
            return "unresolved", None, None
        return ("blocked", hit, float(closest)) if hit is not None else ("clear", None, None)
