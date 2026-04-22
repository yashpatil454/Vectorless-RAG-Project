from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class TreeNode:
    node_id: str
    node_type: str          # "leaf" | "subsection" | "section" | "document"
    text: str
    summary: str
    metadata: Dict[str, Any]
    children: List[str] = field(default_factory=list)   # child node_ids

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "text": self.text,
            "summary": self.summary,
            "metadata": self.metadata,
            "children": self.children,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TreeNode":
        return cls(
            node_id=d["node_id"],
            node_type=d["node_type"],
            text=d["text"],
            summary=d.get("summary", ""),
            metadata=d.get("metadata", {}),
            children=d.get("children", []),
        )


@dataclass
class CustomTreeIndex:
    root_ids: List[str]
    nodes: Dict[str, TreeNode]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "root_ids": self.root_ids,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CustomTreeIndex":
        data = json.loads(path.read_text(encoding="utf-8"))
        nodes = {nid: TreeNode.from_dict(nd) for nid, nd in data["nodes"].items()}
        return cls(root_ids=data["root_ids"], nodes=nodes)

    @staticmethod
    def make_node_id() -> str:
        return str(uuid.uuid4())
