"""Frozen multi-task one-shot policy benchmark contracts and scoring."""
from __future__ import annotations
from typing import Literal
from pydantic import Field,model_validator
from .._base import SchemaModel
from ..comparison.models import BlockedStudyReport
from .policy import candidate_order

class PolicyDefinition(SchemaModel):
 name:str; kind:Literal["task_map","static","seeded_random"]
 task_actions:dict[str,int]=Field(default_factory=dict); candidate_index:int|None=None
 random_seeds:list[int]=Field(default_factory=list)
 @model_validator(mode="after")
 def check(self):
  if self.kind=="task_map" and not self.task_actions:raise ValueError("task_map requires actions")
  if self.kind=="static" and self.candidate_index is None:raise ValueError("static requires candidate_index")
  if self.kind=="seeded_random" and not self.random_seeds:raise ValueError("random requires seeds")
  return self
class PolicyBenchmarkSpec(SchemaModel):
 spec_version:Literal["0.1.0"]="0.1.0"; benchmark_id:str; task_ids:list[str]; candidate_count:int=Field(ge=2); policies:list[PolicyDefinition]
 @model_validator(mode="after")
 def check(self):
  if len(set(self.task_ids))!=len(self.task_ids):raise ValueError("task ids must be distinct")
  if len({p.name for p in self.policies})!=len(self.policies):raise ValueError("policy names must be distinct")
  for p in self.policies:
   indices=list(p.task_actions.values()) if p.kind=="task_map" else [p.candidate_index] if p.kind=="static" else []
   if p.kind=="task_map" and set(p.task_actions)!=set(self.task_ids):raise ValueError("task_map must cover tasks exactly")
   if any(i is None or i<0 or i>=self.candidate_count for i in indices):raise ValueError("candidate index out of range")
  return self
class PolicyScore(SchemaModel):
 policy_name:str; evaluations:int; slo_success_rate:float; oracle_hit_rate:float; mean_simple_regret_ms:float|None
class PolicyBenchmarkReport(SchemaModel):
 report_version:Literal["0.1.0"]="0.1.0"; spec:PolicyBenchmarkSpec; scores:list[PolicyScore]
 interpretation:str="One-shot held-out policy scoring; no significance, production, or deployment-safety claim."
def evaluate_policy_benchmark(spec:PolicyBenchmarkSpec,reports:dict[str,BlockedStudyReport])->PolicyBenchmarkReport:
 if set(reports)!=set(spec.task_ids):raise ValueError("reports must exactly match benchmark tasks")
 scores=[]
 for policy in spec.policies:
  outcomes=[]
  seeds=policy.random_seeds if policy.kind=="seeded_random" else [None]
  for seed in seeds:
   for task in spec.task_ids:
    report=reports[task]
    if len(report.candidates)!=spec.candidate_count:raise ValueError("candidate count mismatch")
    index=(policy.task_actions[task] if policy.kind=="task_map" else policy.candidate_index if policy.kind=="static" else candidate_order("seeded_random-v1",spec.candidate_count,seed)[0])
    candidate=report.candidates[index];hit=index in report.best_candidate_indices
    regret=None if not candidate.robust_feasible or not report.best_candidate_indices else max(0.,candidate.mean_objective_mean-report.candidates[report.best_candidate_indices[0]].mean_objective_mean)
    outcomes.append((candidate.robust_feasible,hit,regret))
  regrets=[x[2] for x in outcomes if x[2] is not None]
  scores.append(PolicyScore(policy_name=policy.name,evaluations=len(outcomes),slo_success_rate=sum(x[0] for x in outcomes)/len(outcomes),oracle_hit_rate=sum(x[1] for x in outcomes)/len(outcomes),mean_simple_regret_ms=sum(regrets)/len(regrets) if regrets else None))
 return PolicyBenchmarkReport(spec=spec,scores=scores)
