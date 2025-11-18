# Project 6.3: Auto-Scaling LLM Deployment

**Difficulty**: Advanced
**Estimated Time**: 8-10 hours
**Type**: Individual Project

---

## Project Overview

Design and implement a complete auto-scaling deployment system for LLM inference that responds to traffic patterns, optimizes costs, and maintains SLOs.

---

## Objectives

Build a system that:
1. Automatically scales based on multiple metrics
2. Optimizes for cost while meeting SLOs
3. Handles traffic spikes gracefully
4. Provides predictive scaling
5. Minimizes cold start latency

---

## Requirements

### Scaling Policies

1. **Reactive Scaling**:
   - CPU/Memory-based (HPA)
   - Queue depth-based
   - Request rate-based
   - Latency-based

2. **Predictive Scaling**:
   - Time-based patterns (traffic prediction)
   - ML-based prediction (optional)
   - Pre-warming for expected traffic

3. **Cost Optimization**:
   - Spot instance management
   - Right-sizing instances
   - Schedule-based scaling (off-peak reduction)
   - Resource bin-packing

4. **SLO Maintenance**:
   - Availability > 99.9%
   - P95 latency < 2s
   - Error rate < 0.1%

---

## Architecture Components

```
┌─────────────────────────────────────────────┐
│         Scaling Controller                   │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
│  │ Metrics  │  │ Predictor│  │ Policy    │ │
│  │ Collector│──▶│          │──▶│ Engine    │ │
│  └──────────┘  └──────────┘  └─────┬─────┘ │
└────────────────────────────────────┼────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │ K8s HPA      │ │ Cluster      │ │ Cloud        │
            │              │ │ Autoscaler   │ │ Autoscaling  │
            └──────────────┘ └──────────────┘ └──────────────┘
```

---

## Implementation Guide

### Part 1: Multi-Metric Auto-Scaling (3 hours)

**Implement HPA with custom metrics**:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llama-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llama-server
  minReplicas: 2
  maxReplicas: 50
  metrics:
  # CPU
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  # Memory
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  # Custom: Queue depth
  - type: Pods
    pods:
      metric:
        name: queue_depth
      target:
        type: AverageValue
        averageValue: "10"
  # Custom: Request rate
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "50"

  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 4
        periodSeconds: 30
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

### Part 2: Queue-Based Scaling (2 hours)

**Implement queue monitoring and scaling**:

```python
import asyncio
from kubernetes import client, config

class QueueBasedScaler:
    def __init__(self, namespace="llm-inference"):
        config.load_kube_config()
        self.api = client.AppsV1Api()
        self.namespace = namespace

    async def scale_based_on_queue(self):
        while True:
            queue_depth = await self.get_queue_depth()
            current_replicas = self.get_current_replicas()

            desired = self.calculate_desired_replicas(
                queue_depth,
                current_replicas
            )

            if desired != current_replicas:
                self.scale_deployment(desired)

            await asyncio.sleep(15)

    def calculate_desired_replicas(
        self,
        queue_depth: int,
        current: int
    ) -> int:
        # 1 replica per 50 queued requests
        target = max(2, min(queue_depth // 50, 50))

        # Gradual scaling
        if target > current:
            return min(current + 2, target)
        elif target < current:
            return max(current - 1, target)

        return current
```

### Part 3: Predictive Scaling (3 hours)

**Implement time-based traffic prediction**:

```python
from datetime import datetime, timedelta
import pandas as pd

class PredictiveScaler:
    def __init__(self, historical_data_days=30):
        self.data = self.load_historical_data(historical_data_days)

    def predict_next_hour_traffic(self) -> float:
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()

        # Get historical traffic for this hour/day
        similar_periods = self.data[
            (self.data['hour'] == hour) &
            (self.data['day_of_week'] == day_of_week)
        ]

        # Average with recent trend
        historical_avg = similar_periods['requests_per_second'].mean()
        recent_trend = self.get_recent_trend()

        # Weighted average
        prediction = 0.7 * historical_avg + 0.3 * recent_trend

        return prediction

    def pre_scale_for_peak(self):
        \"\"\"Scale up 15 minutes before predicted peak\"\"\"
        prediction = self.predict_next_hour_traffic()

        if prediction > threshold:
            # Scale up preemptively
            desired_replicas = int(prediction / 50)
            self.scale_deployment(desired_replicas)
```

### Part 4: Cost Optimization (2 hours)

**Implement spot instance management**:

```python
class CostOptimizer:
    def __init__(self):
        self.on_demand_cost = 1.0  # per hour
        self.spot_cost = 0.3  # per hour

    def select_instance_type(
        self,
        required_capacity: int,
        latency_requirement: float,
        budget: float
    ) -> str:
        \"\"\"Select optimal instance mix\"\"\"

        # Use spot instances for base load
        spot_capacity = int(required_capacity * 0.7)
        on_demand_capacity = required_capacity - spot_capacity

        # Calculate cost
        hourly_cost = (
            spot_capacity * self.spot_cost +
            on_demand_capacity * self.on_demand_cost
        )

        if hourly_cost > budget:
            # Reduce spot, increase on-demand
            spot_capacity = int(required_capacity * 0.5)
            on_demand_capacity = required_capacity - spot_capacity

        return {
            "spot": spot_capacity,
            "on_demand": on_demand_capacity,
            "cost": hourly_cost
        }

    def schedule_scaling(self):
        \"\"\"Reduce capacity during off-peak hours\"\"\"
        schedule = {
            "peak": (9, 17),     # 9 AM - 5 PM: full capacity
            "medium": (6, 21),   # 6 AM - 9 PM: 70% capacity
            "low": (0, 24),      # Night: 30% capacity
        }

        hour = datetime.now().hour

        if 9 <= hour <= 17:
            return 1.0  # 100%
        elif 6 <= hour <= 21:
            return 0.7  # 70%
        else:
            return 0.3  # 30%
```

---

## Deliverables

1. **Scaling System**:
   - HPA configurations
   - Custom metrics exporter
   - Predictive scaling service
   - Cost optimizer

2. **Documentation**:
   - Scaling architecture
   - Configuration guide
   - Tuning recommendations
   - Cost analysis

3. **Testing**:
   - Load testing scenarios
   - Scaling behavior tests
   - Cost comparison analysis

4. **Dashboard**:
   - Real-time scaling visualization
   - Cost tracking
   - SLO monitoring

---

## Testing Scenarios

1. **Traffic Spike**: 10x traffic increase over 5 minutes
2. **Gradual Growth**: Steady 2x growth over 1 hour
3. **Periodic Pattern**: Daily traffic cycle
4. **Sudden Drop**: Traffic drops to 10%
5. **Mixed Workload**: Varying request sizes

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Scale-up latency | < 2 minutes |
| Scale-down latency | < 5 minutes |
| Availability | > 99.9% |
| P95 latency | < 2s |
| Cost optimization | > 30% savings vs baseline |
| Over-provisioning | < 20% |

---

## Bonus Challenges

1. Implement ML-based traffic prediction
2. Multi-region auto-scaling
3. Automated canary deployments
4. Budget-aware scaling
5. SLO-based auto-scaling

---

**Recommended Duration**: 1-2 weeks part-time
