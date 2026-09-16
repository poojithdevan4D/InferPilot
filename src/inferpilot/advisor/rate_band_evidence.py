from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from .._base import SchemaModel


class BoundaryCell(SchemaModel):
    run_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_rate_qps: float = Field(gt=0)
    arrival_seed: int
    max_num_seqs: int = Field(gt=0)
    ttft_p95_ms: float = Field(ge=0)
    tpot_p95_ms: float = Field(ge=0)


class ValidatedRateBand(SchemaModel):
    min_rate_qps: float = Field(gt=0)
    max_rate_qps: float = Field(gt=0)
    max_num_seqs: int = Field(gt=0)
    cells: list[BoundaryCell] = Field(min_length=6, max_length=6)

    @model_validator(mode="after")
    def check(self):
        if self.min_rate_qps >= self.max_rate_qps:
            raise ValueError("invalid rate band")
        if {cell.request_rate_qps for cell in self.cells} != {
            self.min_rate_qps,
            self.max_rate_qps,
        }:
            raise ValueError("cells must cover both band endpoints")
        endpoint_seeds = []
        for rate in (self.min_rate_qps, self.max_rate_qps):
            seeds = [cell.arrival_seed for cell in self.cells if cell.request_rate_qps == rate]
            if len(seeds) != 3 or len(set(seeds)) != 3:
                raise ValueError("each endpoint requires three distinct seeds")
            endpoint_seeds.append(set(seeds))
        if endpoint_seeds[0] != endpoint_seeds[1]:
            raise ValueError("band endpoints must use the same seed blocks")
        if any(cell.max_num_seqs != self.max_num_seqs for cell in self.cells):
            raise ValueError("cell action does not match band action")
        return self


class RateBandEvidenceReport(SchemaModel):
    report_version: Literal["0.1.0"] = "0.1.0"
    policy_id: str
    model: str
    revision: str
    gpu_name: str
    prompt_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=1)
    arrival_pattern: Literal["poisson-v1", "batched-poisson-v1"]
    fixed_engine_fields: dict[str, Any]
    ttft_p95_limit_ms: float = Field(gt=0)
    tpot_p95_limit_ms: float = Field(gt=0)
    bands: list[ValidatedRateBand] = Field(min_length=1)

    @model_validator(mode="after")
    def check(self):
        ordered = sorted(self.bands, key=lambda band: band.min_rate_qps)
        if any(
            left.max_rate_qps >= right.min_rate_qps
            for left, right in zip(ordered, ordered[1:])
        ):
            raise ValueError("rate bands overlap")
        cells = [cell for band in self.bands for cell in band.cells]
        if len({cell.run_id for cell in cells}) != len(cells):
            raise ValueError("boundary run ids must be distinct")
        if any(
            cell.ttft_p95_ms > self.ttft_p95_limit_ms
            or cell.tpot_p95_ms > self.tpot_p95_limit_ms
            for cell in cells
        ):
            raise ValueError("rate-band evidence violates SLO")
        return self
