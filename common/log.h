#pragma once

// Back-compat shim. The real header now lives in common/log/log.h and
// is built by the `llama-log` static target. This file exists so
// existing `#include "log.h"` call sites keep working without changes.

#include "log/log.h"
