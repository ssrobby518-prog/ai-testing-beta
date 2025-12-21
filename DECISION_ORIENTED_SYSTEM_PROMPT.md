# Decision-Oriented AI Detection System Prompt

**System**: TSAR_RAPTOR_V6 - Non-blocking Detection Subsystem

## Core Philosophy
- **Decision over Label**: Output routing decisions, NOT classifications
- **Uncertainty as Asset**: Track uncertainty reduction, not confidence maximization
- **Attention Governed**: Allocate compute by information gain, not all samples equally
- **Fail Open**: Degrade gracefully under resource pressure, never block pipeline

## 1. Scheduler Decision Logic

**State Machine** (NOT classification pipeline):
```
INIT → LOW_COST → {STOP | MID_COST} → {STOP | ROUTE | HIGH_COST} → {ROUTE | HUMAN_ESCALATION}
```

**Decision Criteria** (information theory, NOT accuracy):
- `LOW_COST`: If `information_gain < 0.10` → STOP (discard sample)
- `MID_COST`: If `uncertainty_remain < 0.15 AND disagreement < 0.35` → ROUTE
- `HIGH_COST`: If `disagreement >= 0.35` → HUMAN_ESCALATION (max 200/day)

**Output Format**:
```python
{
  "routing_decision": "ai_pool | real_pool | disagreement_pool",
  "uncertainty_state": 0.23,  # remaining uncertainty
  "disagreement_score": 0.12, # module conflict level
  "confidence": None  # operational metric only, NOT label confidence
}
```

## 2. Attention Budget Governance

**Priority Hierarchy** (resource allocation):
```yaml
core_integrity: {floor_gpu: 1, priority: 10}     # always protected
detection_layer: {ceiling_gpu: 1, priority: 7}   # throttleable
economic_layer: {ceiling_gpu: 2, priority: 5}    # first to degrade
```

**Risk Gradient Sorting** (NOT confidence sorting):
- HIGH COST modules run ONLY when `max_high_cost_rate < 0.05`
- Sort queue by: `(uncertainty_reduction / computational_cost)` ratio
- Discard bottom 20% samples when GPU utilization > 92%

**Degradation Order**:
1. Economic layer → disabled
2. Detection layer → LOW_COST only
3. Core integrity → BYPASS (fail open)

## 3. Human Feedback as Policy

**Human Role** (policy tuner, NOT labeler):
```python
human_output = {
  "hypothesis_adjustment": {...},      # refine generation hypothesis
  "module_trust_feedback": {...}       # adjust module weights
}
# NO "label" or "ground_truth" fields allowed
```

**Policy Learning Loop**:
- Maximize: `uncertainty_reduction_per_cost`
- Minimize: `wasted_high_cost_calls`
- Adjustable: `disagreement_high ∈ [0.25, 0.50]`, `min_continue ∈ [0.05, 0.25]`
- Safety: Max delta 0.05 per update, rollback on regression

## 4. Failure Modes & Safety Valves

**Auto Circuit Breakers**:
```yaml
gpu_pressure: {condition: "vram > 0.92", action: DEGRADED_LOW_ONLY}
latency_violation: {condition: "p95 > 300ms", action: BYPASS}
human_budget_exhausted: {condition: "escalations >= 200/day", action: DEGRADED_LOW_ONLY}
low_information_period: {condition: "mean_info_gain_1h < min_viable", action: DEGRADED_LOW_ONLY}
```

**Degradation Modes**:
- `NORMAL`: Full pipeline
- `DEGRADED_LOW_ONLY`: Skip MID/HIGH COST layers
- `BYPASS`: Output default routing, log for offline processing

**Recovery Protocol**:
- Manual override TTL: 180 minutes
- Audit log: All mode transitions
- SLA: `non_blocking: true`, `fail_mode: FAIL_OPEN`

---

**Critical Constraint**: System NEVER outputs labels during inference. Labels exist ONLY in stable dataset after long-term decision consistency (月度擴增).

**Generated with**: First Principles × Decision Theory × Resource Governance
