#pragma once

// Polling interval in milliseconds used for backend health checks during startup.
#define ROUTER_BACKEND_HEALTH_POLL_MS 100

// Polling interval in milliseconds used for process exit checks and shutdown loops.
#define ROUTER_PROCESS_POLL_INTERVAL_MS 100

// Maximum time in milliseconds to wait for a process to terminate gracefully before forcing shutdown.
#define ROUTER_PROCESS_SHUTDOWN_TIMEOUT_MS 2000

// Maximum time in milliseconds to wait for a backend to report readiness.
#define ROUTER_BACKEND_READY_TIMEOUT_MS 60000
