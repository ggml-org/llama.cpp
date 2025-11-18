# Lab 6.4: Kubernetes Deployment

**Difficulty**: Advanced
**Estimated Time**: 90 minutes
**Prerequisites**: Lab 6.3, Kubernetes basics, kubectl installed

---

## Objectives

1. Deploy llama-server to Kubernetes
2. Configure auto-scaling
3. Set up Ingress
4. Implement health probes
5. Monitor with Prometheus

---

## Part 1: Basic Deployment (25 min)

**deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: llama-server
  template:
    metadata:
      labels:
        app: llama-server
    spec:
      containers:
      - name: llama
        image: llama-server:v1
        args:
          - -m
          - /models/llama-2-7b-chat.Q4_K_M.gguf
          - -ngl
          - "35"
          - --metrics
        ports:
        - containerPort: 8080
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: 16Gi
          limits:
            nvidia.com/gpu: 1
            memory: 32Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
```

**Deploy**:
```bash
kubectl create namespace llm-inference
kubectl apply -f deployment.yaml -n llm-inference
kubectl get pods -n llm-inference
kubectl logs -f deployment/llama-server -n llm-inference
```

---

## Part 2: Service & Ingress (20 min)

**service.yaml**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: llama-server
spec:
  selector:
    app: llama-server
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
```

**ingress.yaml**:
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: llama-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  rules:
  - host: llm-api.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: llama-server
            port:
              number: 80
```

**Apply**:
```bash
kubectl apply -f service.yaml -n llm-inference
kubectl apply -f ingress.yaml -n llm-inference

# Test
kubectl port-forward svc/llama-server 8080:80 -n llm-inference
curl http://localhost:8080/health
```

---

## Part 3: Auto-Scaling (20 min)

**hpa.yaml**:
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
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

**Apply and test**:
```bash
kubectl apply -f hpa.yaml -n llm-inference
kubectl get hpa -n llm-inference

# Generate load
kubectl run -it load-generator --rm --image=busybox -n llm-inference -- /bin/sh
while true; do wget -q -O- http://llama-server/v1/chat/completions; done

# Watch scaling
kubectl get hpa -n llm-inference -w
```

---

## Part 4: Monitoring (25 min)

**ServiceMonitor for Prometheus Operator**:
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: llama-server
spec:
  selector:
    matchLabels:
      app: llama-server
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
```

**View metrics in Prometheus**:
```bash
kubectl port-forward -n monitoring svc/prometheus 9090:9090

# Visit http://localhost:9090
# Query: rate(llama_requests_total[5m])
```

---

## Deliverables

1. All YAML manifests
2. Screenshot of running pods
3. Screenshot of HPA scaling in action
4. Prometheus metrics screenshot

---

## Challenge

Implement PodDisruptionBudget and test rolling updates

**Next Lab**: [Lab 6.5 - Monitoring Setup](./lab-05-monitoring-setup.md)
