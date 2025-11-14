#pragma once

/* the rest of this file must match virglrenderer/src/apir-protocol.h */

#include <unistd.h>

#define VENUS_COMMAND_TYPE_LENGTH 331

#define APIR_PROTOCOL_MAJOR 0
#define APIR_PROTOCOL_MINOR 1

#define APIR_HANDSHAKE_MAGIC 0xab1e

typedef enum {
    APIR_COMMAND_TYPE_HandShake = 0,
    APIR_COMMAND_TYPE_LoadLibrary = 1,
    APIR_COMMAND_TYPE_Forward = 2,

    APIR_COMMAND_TYPE_LENGTH = 3,
} ApirCommandType;

typedef uint64_t ApirCommandFlags;

typedef enum {
    APIR_LOAD_LIBRARY_SUCCESS = 0,
    APIR_LOAD_LIBRARY_HYPERCALL_INITIALIZATION_ERROR = 1,
    APIR_LOAD_LIBRARY_ALREADY_LOADED = 2,
    APIR_LOAD_LIBRARY_ENV_VAR_MISSING = 3,
    APIR_LOAD_LIBRARY_CANNOT_OPEN = 4,
    APIR_LOAD_LIBRARY_SYMBOL_MISSING = 5,
    APIR_LOAD_LIBRARY_INIT_BASE_INDEX = 6, // anything above this is a APIR backend library initialization return code
} ApirLoadLibraryReturnCode;

typedef enum {
    APIR_FORWARD_SUCCESS = 0,
    APIR_FORWARD_NO_DISPATCH_FCT = 1,
    APIR_FORWARD_TIMEOUT = 2,

    APIR_FORWARD_BASE_INDEX = 3, // anything above this is a APIR backend library forward return code
} ApirForwardReturnCode;

__attribute__((unused))
static inline const char *apir_command_name(ApirCommandType type)
{
  switch (type) {
  case APIR_COMMAND_TYPE_HandShake: return "HandShake";
  case APIR_COMMAND_TYPE_LoadLibrary: return "LoadLibrary";
  case APIR_COMMAND_TYPE_Forward: return "Forward";
  default: return "unknown";
  }
}

__attribute__((unused))
static const char *apir_load_library_error(ApirLoadLibraryReturnCode code) {
#define APIR_LOAD_LIBRARY_ERROR(code_name) \
  do {						 \
    if (code == code_name) return #code_name;	 \
  } while (0)					 \

  APIR_LOAD_LIBRARY_ERROR(APIR_LOAD_LIBRARY_SUCCESS);
  APIR_LOAD_LIBRARY_ERROR(APIR_LOAD_LIBRARY_HYPERCALL_INITIALIZATION_ERROR);
  APIR_LOAD_LIBRARY_ERROR(APIR_LOAD_LIBRARY_ALREADY_LOADED);
  APIR_LOAD_LIBRARY_ERROR(APIR_LOAD_LIBRARY_ENV_VAR_MISSING);
  APIR_LOAD_LIBRARY_ERROR(APIR_LOAD_LIBRARY_CANNOT_OPEN);
  APIR_LOAD_LIBRARY_ERROR(APIR_LOAD_LIBRARY_SYMBOL_MISSING);
  APIR_LOAD_LIBRARY_ERROR(APIR_LOAD_LIBRARY_INIT_BASE_INDEX);

  return "Unknown APIR_COMMAND_TYPE_LoadLibrary error";

#undef APIR_LOAD_LIBRARY_ERROR
}

__attribute__((unused))
static const char *apir_forward_error(ApirForwardReturnCode code) {
#define APIR_FORWARD_ERROR(code_name) \
  do {						 \
    if (code == code_name) return #code_name;	 \
  } while (0)					 \

  APIR_FORWARD_ERROR(APIR_FORWARD_SUCCESS);
  APIR_FORWARD_ERROR(APIR_FORWARD_NO_DISPATCH_FCT);
  APIR_FORWARD_ERROR(APIR_FORWARD_TIMEOUT);
  APIR_FORWARD_ERROR(APIR_FORWARD_BASE_INDEX);

  return "Unknown APIR_COMMAND_TYPE_Forward error";

#undef APIR_FORWARD_ERROR
}
