"""Data-quality issue log -> machine-readable report."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


@dataclass
class Issue:
    file: str
    issue_type: str
    rows_affected: int
    action: str
    category: str = "error"  # error | real_event | info
    detail: str = ""
    examples: list = field(default_factory=list)


class DQLog:
    """Collects issues found while cleaning plus before/after row counts."""

    def __init__(self) -> None:
        self.issues: list[Issue] = []
        self.counts: dict[str, dict[str, int]] = {}

    def add(
        self,
        file: str,
        issue_type: str,
        rows: int | pd.Series,
        action: str,
        *,
        category: str = "error",
        detail: str = "",
        examples: list | pd.DataFrame | None = None,
    ) -> None:
        n = int(rows.sum()) if isinstance(rows, pd.Series) else int(rows)
        if n == 0:
            return
        if isinstance(examples, pd.DataFrame):
            examples = json.loads(examples.head(5).to_json(orient="records"))
        self.issues.append(
            Issue(file, issue_type, n, action, category, detail, list(examples or [])[:5])
        )

    def count(self, file: str, stage: str, n: int) -> None:
        self.counts.setdefault(file, {})[stage] = int(n)

    def to_frame(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(columns=[f for f in Issue.__dataclass_fields__])
        df = pd.DataFrame([asdict(i) for i in self.issues])
        df["examples"] = df["examples"].map(json.dumps)
        return df

    def write(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "row_counts": self.counts,
            "n_issue_types": len(self.issues),
            "issues": [asdict(i) for i in self.issues],
        }
        (out_dir / "dq_report.json").write_text(json.dumps(report, indent=2, default=str))
        self.to_frame().to_csv(out_dir / "dq_issues.csv", index=False)
