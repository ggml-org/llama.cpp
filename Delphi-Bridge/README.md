# Delphi Bridge for llama.cpp

A cross-platform library (DLL/dylib) built with Delphi 12 FMX that provides audio streaming, JSON command dispatching, and a fuzzy correction UI for AI-powered applications.

## Features

### Audio Streamer
- **TAudioCaptureDevice Handler**: Captures and streams raw PCM audio data
- **Network Streaming**: Sends audio to localhost:9000 via TCP socket
- **Real-time Processing**: Low-latency audio capture and transmission

### JSON Command Dispatcher
- **TCommandRouter Class**: Listens for JSON commands on configurable port (default: 9001)
- **Structured Commands**: Processes JSON with format:
  ```json
  {
    "timestamp": 1234567890,
    "group": "CAMERA_CONTROL",
    "verb": "PAN",
    "subject": "main_camera",
    "params": { "angle": 25.0 }
  }
  ```

### Subsystem Routing
The `RouteCommand` method handles three command groups:

1. **CAMERA_CONTROL**: Routes to `FF_HandleCameraCommand(AVerb, AParams)`
   - Handles PAN, TILT, ZOOM commands
   - C-exported function for external integration

2. **ACTOR_POSE**: Updates thread-safe `TAIPoseBuffer`
   - Parses `pose_data` array from params
   - Maintains pose history with timestamps

3. **OBJECT_MGMT**: Routes to `FF_HandleObjectCommand(AVerb, AParams)`
   - Handles CREATE, DELETE, MODIFY commands
   - C-exported function for external integration

### Fuzzy Correction UI
- **Floating FMX Form**: Displays recognized intents to the user
  - Example: "Recognized Intent: CAMERA_CONTROL: PAN main_camera"
- **Undo Button**: Sends cancellation signal back to Sidecar
- **Auto-hide**: Form disappears after 5 seconds
- **Thread-safe**: Synchronized UI updates from background threads

### Temporal Consistency (Race Condition Handling)
- **2-Second Command Buffer**: Maintains timestamped history of all commands
- **Automatic Rollback**: When a command arrives with an earlier timestamp:
  1. Identifies all commands executed after that timestamp
  2. Calls undo handlers for affected commands
  3. Rolls back pose buffer to the specified timestamp
  4. Executes the new command
- **Thread-safe Operations**: All buffer operations use critical sections

## Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Audio Capture  │─────>│   TCP Socket     │─────>│  localhost:9000 │
│     Device      │      │   (PCM Stream)   │      │   (Receiver)    │
└─────────────────┘      └──────────────────┘      └─────────────────┘

┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Command Source │─────>│   TCP Socket     │─────>│ Command Router  │
│   (JSON Stream) │      │  localhost:9001  │      │  RouteCommand() │
└─────────────────┘      └──────────────────┘      └─────────────────┘
                                                            │
                    ┌───────────────────────────────────────┼──────────┐
                    │                                       │          │
                    ▼                                       ▼          ▼
        ┌─────────────────────┐              ┌──────────────────────────┐
        │ FF_HandleCameraCmd  │              │  TAIPoseBuffer           │
        │ (C-exported)        │              │  (Thread-safe)           │
        └─────────────────────┘              └──────────────────────────┘
                    │
                    ▼                         ┌──────────────────────────┐
        ┌─────────────────────┐              │  FF_HandleObjectCmd      │
        │   TCommandBuffer    │              │  (C-exported)            │
        │   (2s Rollback)     │              └──────────────────────────┘
        └─────────────────────┘
                    │
                    ▼
        ┌─────────────────────┐
        │    Fuzzy UI Form    │
        │  (Intent Display)   │
        └─────────────────────┘
```

## Building

### Prerequisites
- Delphi 12 (or later) with FMX support
- Windows, macOS, or Linux target platform

### Build Steps
1. Open `DelphiBridge.dpr` in Delphi IDE
2. Select target platform (Win32/Win64/OSX64/Linux64)
3. Build → Project → Build DelphiBridge
4. Output DLL/dylib will be in `Build/` directory

### Command Line Build (Windows)
```bash
cd Delphi-Bridge
msbuild DelphiBridge.dproj /p:Config=Release /p:Platform=Win64
```

### Command Line Build (macOS/Linux)
```bash
cd Delphi-Bridge
dcc64 DelphiBridge.dpr
```

## Usage

### C/C++ Integration

```c
// Function prototypes
typedef int (*DB_Initialize_t)(const char* audioHost, int audioPort, 
                                int commandPort, const char* sidecarHost, 
                                int sidecarPort);
typedef int (*DB_Start_t)();
typedef int (*DB_Stop_t)();
typedef void (*DB_Shutdown_t)();
typedef int (*DB_RouteCommand_t)(const char* json);

// Load library
#ifdef _WIN32
  HMODULE lib = LoadLibrary("DelphiBridge.dll");
#else
  void* lib = dlopen("libDelphiBridge.dylib", RTLD_LAZY);
#endif

// Get function pointers
DB_Initialize_t DB_Initialize = (DB_Initialize_t)GetProcAddress(lib, "DB_Initialize");
DB_Start_t DB_Start = (DB_Start_t)GetProcAddress(lib, "DB_Start");

// Initialize
DB_Initialize("localhost", 9000, 9001, "localhost", 9002);
DB_Start();

// Route a command
const char* json = "{\"timestamp\":1234567890,\"group\":\"CAMERA_CONTROL\","
                   "\"verb\":\"PAN\",\"subject\":\"main_camera\","
                   "\"params\":{\"angle\":25.0}}";
DB_RouteCommand(json);

// Cleanup
DB_Stop();
DB_Shutdown();
```

### Python Integration

```python
import ctypes
import json

# Load library
lib = ctypes.CDLL('./DelphiBridge.dll')  # or .dylib/.so

# Initialize
lib.DB_Initialize(b"localhost", 9000, 9001, b"localhost", 9002)
lib.DB_Start()

# Send command
cmd = {
    "timestamp": 1234567890,
    "group": "CAMERA_CONTROL",
    "verb": "PAN",
    "subject": "main_camera",
    "params": {"angle": 25.0}
}
lib.DB_RouteCommand(json.dumps(cmd).encode())

# Cleanup
lib.DB_Stop()
lib.DB_Shutdown()
```

## API Reference

### Exported Functions

#### `DB_Initialize`
```c
int DB_Initialize(const char* audioHost, int audioPort, 
                   int commandPort, const char* sidecarHost, 
                   int sidecarPort);
```
Initializes all subsystems. Returns 1 on success, 0 on failure.

#### `DB_Start`
```c
int DB_Start();
```
Starts audio streaming and command listening. Returns 1 on success, 0 on failure.

#### `DB_Stop`
```c
int DB_Stop();
```
Stops all active streaming and listening. Returns 1 on success, 0 on failure.

#### `DB_Shutdown`
```c
void DB_Shutdown();
```
Cleans up all resources. Call before unloading library.

#### `DB_RouteCommand`
```c
int DB_RouteCommand(const char* json);
```
Routes a JSON command string. Returns 1 on success, 0 on failure, -1 on error.

#### `DB_GetPoseCount`
```c
int DB_GetPoseCount();
```
Returns the number of poses currently in the buffer.

#### `DB_GetCommandCount`
```c
int DB_GetCommandCount();
```
Returns the number of commands in the 2-second buffer.

#### `FF_HandleCameraCommand`
```c
int FF_HandleCameraCommand(const char* verb, const char* params);
```
Handles camera control commands. Returns 1 on success, -1 for unknown verb, -2 on error.

#### `FF_HandleObjectCommand`
```c
int FF_HandleObjectCommand(const char* verb, const char* params);
```
Handles object management commands. Returns 1 on success, -1 for unknown verb, -2 on error.

## Thread Safety

- **TAIPoseBuffer**: Protected by critical sections, safe for concurrent access
- **TCommandBuffer**: Thread-safe command history with rollback support
- **UI Updates**: Synchronized to main thread via `TThread.Synchronize`
- **Socket Operations**: Each client connection runs in its own task

## Race Condition Handling

The library handles out-of-order command arrival:

1. **Command arrives at T=100ms**
   - Added to buffer
   - Executed
   - Marked as executed

2. **Command arrives at T=50ms** (earlier timestamp)
   - System detects timestamp < latest
   - Rolls back all commands after T=50ms
   - Undoes executed commands (calls undo handlers)
   - Rolls back pose buffer
   - Executes new command at T=50ms

3. **Buffer maintains 2-second window**
   - Automatically removes commands older than 2 seconds
   - Reduces memory usage
   - Maintains temporal consistency

## Customization

### Extending Command Groups
Add new command groups in `CommandRouter.pas`:

```pascal
else if Group = 'NEW_GROUP' then
begin
  // Handle new command group
  Result := True;
end
```

### Modifying Buffer Duration
Change `BUFFER_DURATION_MS` in `CommandBuffer.pas`:

```pascal
const
  BUFFER_DURATION_MS = 5000; // 5 seconds instead of 2
```

### Custom Undo Logic
Implement undo handlers in `ExportedFunctions.pas`:

```pascal
if Verb = 'UNDO_PAN' then
begin
  // Reverse the pan operation
  // Extract original angle and negate it
  Result := 1;
end
```

## Examples

See the `Examples/` directory for:
- Simple command sender application
- Audio receiver example
- Integration test suite

## License

This component is part of the llama.cpp project. See main repository LICENSE for details.

## Contributing

Follow the llama.cpp contribution guidelines. See [CONTRIBUTING.md](../CONTRIBUTING.md) in the root directory.

## Notes

- Audio streaming requires appropriate permissions on the host OS
- FMX forms require platform-specific UI frameworks (AppKit on macOS, GTK on Linux)
- The library is designed to be called from C/C++ applications primarily
- For best performance, use a dedicated thread for command routing in the calling application
