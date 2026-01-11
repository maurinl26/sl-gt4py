# -*- coding: utf-8 -*-
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import reduce
import sys
from time import perf_counter
from typing import TYPE_CHECKING

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = None

if TYPE_CHECKING:
    from typing import Optional


@dataclass
class Node:
    label: str
    parent: Node = None
    children: dict[str, Node] = field(default_factory=dict)
    level: int = 0
    tic: float = 0
    total_calls: int = 0
    total_runtime: float = 0


class Timer:
    """
    Utility class to time nested code sections.

    >>> from time import sleep
    >>>
    >>> Timer.start('outer_section')
    >>> sleep(0.5)
    >>> for _ in range(10):
    ...     Timer.start('inner_section')
    ...     sleep(0.1)
    ...     Timer.stop()
    >>> sleep(0.5)
    >>> Timer.stop()
    >>>
    >>> Timer.print('outer_section')  #doctest: +ELLIPSIS
    outer_section: 2... s
    >>> Timer.print('inner_section')  #doctest: +ELLIPSIS
    inner_section: 1... s
    >>>
    >>> Timer.reset()
    >>> Timer.print("outer_section")
    outer_section: 0.000 s

    """

    active: list[str] = []
    head: Optional[Node] = None
    tree: dict[str, Node] = {}

    @classmethod
    def start(cls, label: str, reset: bool = False) -> None:
        """Start timing the code section labelled as ``label``."""

        # optionally reset
        if reset:
            cls.reset()

        # sanity check
        if label in cls.active:
            return

        # add the node to the list of active nodes
        cls.active.append(label)

        # insert the node in the tree
        node_label = cls.active[0]
        node = cls.tree.setdefault(node_label, Node(node_label))
        for i, node_label in enumerate(cls.active[1:]):
            node = node.children.setdefault(node_label, Node(node_label, parent=node, level=i + 1))
        cls.head = node

        # tic
        if cp is not None:
            try:
                cp.cuda.Device(0).synchronize()
            except RuntimeError:
                pass
        cls.head.tic = perf_counter()

    @classmethod
    def stop(cls, label: Optional[str] = None) -> None:
        """Stop the timer for the code section ``label``."""

        # sanity check
        if len(cls.active) == 0:
            return

        # only nested timers are allowed!
        label = label or cls.active[-1]
        assert label == cls.active[-1], f"Cannot stop `{label}` before stopping `{cls.active[-1]}`"

        # toc
        if cp is not None:
            try:
                cp.cuda.Device(0).synchronize()
            except RuntimeError:
                pass
        toc = perf_counter()

        # update statistics
        cls.head.total_calls += 1
        cls.head.total_runtime += toc - cls.head.tic

        # remove the node from the list of active nodes
        cls.active = cls.active[:-1]

        # update head
        cls.head = cls.head.parent

    @classmethod
    def reset(cls) -> None:
        """Reset the tree."""

        cls.active = []
        cls.head = None

        def cb(node):
            node.total_calls = 0
            node.total_runtime = 0

        for root in cls.tree.values():
            cls.traverse(cb, root)

    @classmethod
    def get_time(cls, label) -> float:
        """Return the execution time for the code section ``label`` in seconds."""

        nodes = cls._get_nodes_from_label(label)
        assert len(nodes) > 0, f"{label} is not a valid timer identifier."

        return reduce(lambda x, node: x + node.total_runtime, nodes, 0)

    @classmethod
    def print(cls, label) -> None:
        """Print the execution time for the code section ``label``."""
        print(f"{label}: {cls.get_time(label):.3f} s")

    @classmethod
    def log(cls, logfile: str = "log.txt") -> None:
        """Write the accumulated execution time for all code sections."""

        # ensure all timers have been stopped
        assert len(cls.active) == 0, "Some timers are still running."

        # callback
        def cb(node, out, prefix="", has_peers=False):
            level = node.level
            prefix_now = prefix + "|- " if level > 0 else prefix
            out.write(f"{prefix_now}{node.label}: {node.total_runtime:.3f} s\n")

            prefix_new = prefix if level == 0 else prefix + "|  " if has_peers else prefix + "   "
            peers_new = len(node.children)
            has_peers_new = peers_new > 0
            for i, label in enumerate(node.children.keys()):
                cb(
                    node.children[label],
                    out,
                    prefix=prefix_new,
                    has_peers=has_peers_new and i < peers_new - 1,
                )

        # write to file
        with open(logfile, "w") as outfile:
            for root in cls.tree.values():
                cb(root, outfile)

    @staticmethod
    def traverse(cb, node, **kwargs) -> None:
        """Invoke the callback ``cb`` on all nodes in the tree."""

        cb(node, **kwargs)
        for child in node.children.values():
            Timer.traverse(cb, child, **kwargs)

    @classmethod
    def _get_nodes_from_label(cls, label) -> list[Node]:
        out = []

        def cb(node):
            if node.label == label:
                out.append(node)

        for root in cls.tree.values():
            Timer.traverse(cb, root)

        return out


@contextmanager
def timing(label: str, reset: bool = False):
    """Context manager encapsulating the code to be timed by ``Timer``.

    >>> from time import sleep
    >>>
    >>> with timing('outer_section', reset=True) as timer:
    ...     sleep(0.5)
    ...     for _ in range(10):
    ...         with timing('inner_section'):
    ...             sleep(0.1)
    ...     sleep(0.5)
    >>>
    >>> timer.print('outer_section')  #doctest: +ELLIPSIS
    outer_section: 2... s
    >>> timer.print('inner_section')  #doctest: +ELLIPSIS
    inner_section: 1... s
    """

    try:
        Timer.start(label, reset=reset)
        yield Timer
    finally:
        Timer.stop(label)
