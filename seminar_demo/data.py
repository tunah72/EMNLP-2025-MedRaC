from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


NEGATIVE_DECIMAL = re.compile(r"^-\d+\.\d+$")


@dataclass(frozen=True)
class ExclusionPolicy:
    calculator_reasons: dict[str, str]
    row_reasons: dict[str, str]
    exclude_negative_decimal: bool
    negative_decimal_reason: str
    expected_source_rows: int
    expected_excluded_rows: int
    expected_retained_rows: int

    def reason_for(self, row: pd.Series) -> str | None:
        calculator_id = str(row["Calculator ID"]).strip()
        row_number = str(row["Row Number"]).strip()
        reasons: list[str] = []
        if calculator_id in self.calculator_reasons:
            reasons.append(self.calculator_reasons[calculator_id])
        if row_number in self.row_reasons:
            reasons.append(self.row_reasons[row_number])
        ground_truth = str(row["Ground Truth Answer"]).strip()
        if self.exclude_negative_decimal and NEGATIVE_DECIMAL.fullmatch(ground_truth):
            reasons.append(self.negative_decimal_reason)
        return "; ".join(reasons) if reasons else None


def load_exclusion_policy(path: str | Path) -> ExclusionPolicy:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExclusionPolicy(
        calculator_reasons=payload["exclude_calculator_ids"],
        row_reasons=payload["exclude_row_numbers"],
        exclude_negative_decimal=payload["exclude_negative_decimal_ground_truth"],
        negative_decimal_reason=payload["negative_decimal_reason"],
        expected_source_rows=int(payload["expected_source_rows"]),
        expected_excluded_rows=int(payload["expected_excluded_rows"]),
        expected_retained_rows=int(payload["expected_retained_rows"]),
    )


def apply_exclusion_policy(
    frame: pd.DataFrame, policy: ExclusionPolicy
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(frame) != policy.expected_source_rows:
        raise ValueError(
            f"Expected {policy.expected_source_rows} source rows, found {len(frame)}"
        )
    reasons = frame.apply(policy.reason_for, axis=1)
    excluded = frame[reasons.notna()].copy()
    excluded["Exclusion Reason"] = reasons[reasons.notna()]
    retained = frame[reasons.isna()].copy()
    if len(excluded) != policy.expected_excluded_rows:
        raise ValueError(
            f"Expected {policy.expected_excluded_rows} exclusions, found {len(excluded)}"
        )
    if len(retained) != policy.expected_retained_rows:
        raise ValueError(
            f"Expected {policy.expected_retained_rows} retained rows, found {len(retained)}"
        )
    return retained, excluded


def _normalize_indices(indices: Iterable[int]) -> list[int]:
    result = list(indices)
    if not result:
        raise ValueError("At least one row index is required")
    if len(result) != len(set(result)):
        raise ValueError("Row indices must be unique")
    return result


def select_samples(
    frame: pd.DataFrame,
    policy: ExclusionPolicy,
    *,
    row_indices: Sequence[int] | None = None,
    calculator_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    if row_indices is not None and calculator_ids is not None:
        raise ValueError("Choose row_indices or calculator_ids, not both")
    if row_indices is None and calculator_ids is None:
        row_indices = [0, 40]

    retained, _ = apply_exclusion_policy(frame, policy)

    if row_indices is not None:
        indices = _normalize_indices(row_indices)
        if min(indices) < 0 or max(indices) >= len(frame):
            raise IndexError("Row index is outside the source DataFrame")
        selected = frame.iloc[indices].copy()
        selected.insert(0, "DataFrame Index", indices)
        rejected = [
            int(index)
            for index, (_, row) in zip(indices, selected.iterrows())
            if policy.reason_for(row) is not None
        ]
        if rejected:
            raise ValueError(f"Selected rows are excluded by policy: {rejected}")
        return selected.reset_index(drop=True)

    requested = [str(value) for value in calculator_ids or []]
    if not requested:
        raise ValueError("At least one calculator ID is required")
    available = set(retained["Calculator ID"].astype(str))
    missing = sorted(set(requested) - available)
    if missing:
        raise ValueError(f"Calculator IDs unavailable after exclusions: {missing}")

    rows = []
    for calculator_id in requested:
        row = retained[retained["Calculator ID"].astype(str) == calculator_id].iloc[0]
        row = row.copy()
        row["DataFrame Index"] = int(row.name)
        rows.append(row)
    selected = pd.DataFrame(rows)
    columns = ["DataFrame Index"] + [
        column for column in selected.columns if column != "DataFrame Index"
    ]
    return selected[columns].reset_index(drop=True)
