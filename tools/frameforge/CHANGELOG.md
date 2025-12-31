# FrameForge Voice Command Integration - JSON Configuration Update

## Summary

This update transforms the FrameForge Voice Command Integration from hard-coded verb definitions to a flexible JSON-based configuration system. It also updates the command structure for Delphi Bridge compatibility and adds support for master verbs.

## Key Changes

### 1. JSON-Based Verb Definitions

**File:** `tools/frameforge/verb-definitions.json`

Verb definitions are now loaded from an external JSON file, allowing customization without code changes:

```json
{
  "verbs": [
    {
      "name": "PAN",
      "action_group": "CAMERA_CONTROL",
      "required_parameters": ["direction"],
      "optional_parameters": ["speed", "degrees", "target", "subject"],
      "aliases": ["PIN"],
      "is_master_verb": false,
      "description": "Pan the camera left or right"
    }
  ]
}
```

**Benefits:**
- Easy to add/modify verbs without recompiling
- Support for verb aliases (PIN→PAN, ROOM→ZOOM, PUSH→DOLLY)
- Required and optional parameters for better validation
- Fallback to hard-coded defaults if JSON not provided

### 2. Delphi Bridge Compatible JSON Format

**OLD Format:**
```json
{
  "verb": "PAN",
  "subject": "Camera1",
  "action_group": "CAMERA_CONTROL",
  "parameters": {
    "direction": "LEFT"
  }
}
```

**NEW Format:**
```json
{
  "verb": "PAN",
  "action_group": "CAMERA_CONTROL",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "parameters": {
    "subject": "Camera1",
    "direction": "LEFT"
  }
}
```

**Changes:**
- ✅ `subject` moved into `parameters` object
- ✅ Added `timestamp` field (ISO 8601 format)
- ✅ Added optional `master_verb` field for compound commands
- ✅ Removed `subject` from root level

### 3. Master Verbs

Master verbs require a secondary verb to form complete commands:

**Implemented Master Verbs:**
- `START` / `BEGIN`: Initiates an action
- `HAVE` / `MAKE`: Commands an actor/object  
- `STOP`: Stops an ongoing action

**Examples:**
- "START PANNING LEFT" → `{"verb": "PAN", "master_verb": "START", ...}`
- "HAVE TOM WALK FORWARD" → `{"verb": "MOVE", "master_verb": "HAVE", ...}`
- "BEGIN ZOOMING IN" → `{"verb": "ZOOM", "master_verb": "BEGIN", ...}`

### 4. Required vs Optional Parameters

Each verb now has both required and optional parameters:

| Verb | Required | Optional |
|------|----------|----------|
| PAN | direction | speed, degrees, target, subject |
| ZOOM | direction | speed, degrees, target, subject |
| HAVE | subject | target, direction, speed, degrees, pose_description |
| START | - | subject, target, direction, speed, degrees |

**Benefits:**
- Better validation
- Helps with verb disambiguation (e.g., "ROOM LEFT" doesn't match ZOOM's pattern)

### 5. Verb Disambiguation via Aliases

Aliases help correct common speech recognition errors:

| Heard | Alias | Corrected To |
|-------|-------|--------------|
| PIN | → | PAN |
| ROOM | → | ZOOM |
| PUSH | → | DOLLY |
| WALK / RUN | → | MOVE |
| TURN | → | ROTATE |
| REMOVE | → | DELETE |

## Code Changes

### Modified Files

1. **frameforge-schema.h / .cpp**
   - Added `load_verb_definitions()` function
   - Added `is_master_verb()` function
   - Added `get_optional_parameters()` function
   - Added `get_current_timestamp()` function
   - Updated `Command` structure (removed `subject`, added `master_verb`, `timestamp`)
   - Added master verb enums (START, BEGIN, HAVE, MAKE, STOP)
   - Added MASTER_VERB action group

2. **frameforge-validator.cpp**
   - Updated to handle `subject` in parameters
   - Updated validation to check for subject as a parameter
   - Removed subject validation from root level

3. **frameforge-json.cpp**
   - Updated `command_to_json()` to include timestamp and master_verb
   - Updated to serialize subject from parameters
   - Updated parsing to handle new format

4. **frameforge-sidecar.cpp**
   - Added `--verb-defs` command-line option
   - Updated INTENT_SYSTEM_PROMPT to reflect new JSON format
   - Added verb definitions loading on startup

5. **README.md**
   - Updated documentation with new JSON format
   - Added master verb examples
   - Added parameter tables
   - Added verb disambiguation explanation

### New Files

1. **verb-definitions.json** - Complete verb definitions for all 19 verbs
2. **test-frameforge-new-features.cpp** - Comprehensive tests for new features

### Test Coverage

All tests passing:
- ✅ `test-frameforge-validator` - Core validation tests  
- ✅ `test-frameforge-json-loader` - JSON loading tests
- ✅ `test-frameforge-new-features` - New features tests (master verbs, timestamps, etc.)

## Usage

### Loading Custom Verb Definitions

```bash
./frameforge-sidecar \
  --whisper-model whisper-base.en.bin \
  --llama-model llama-3-8b-instruct.gguf \
  --verb-defs /path/to/custom-verbs.json \
  --audio test.wav
```

### Example Voice Commands

**Camera Control:**
- "Pan the camera left"
- "Start panning slowly to the right"
- "Begin zooming in"

**With Master Verbs:**
- "Have Tom walk to the door"
- "Make Rachel turn around"
- "Start tilting up"

**With Aliases:**
- "Room out" (ROOM → ZOOM OUT)
- "Push in slowly" (PUSH → DOLLY IN)

## Backward Compatibility

✅ **Fully backward compatible** - If no JSON file is provided, the system falls back to hard-coded defaults that match the previous behavior.

## Migration Guide

### For Delphi Bridge Integration

Update your JSON parsing to:
1. Look for `subject` in `parameters` instead of root level
2. Include `timestamp` field in all commands
3. Handle optional `master_verb` field
4. Generate timestamps using ISO 8601 format

Example migration:
```cpp
// OLD
std::string subject = json["subject"];

// NEW  
std::string subject = json["parameters"]["subject"];
```

## Testing

Run all tests:
```bash
cmake --build build --target test-frameforge-validator test-frameforge-json-loader test-frameforge-new-features
./build/bin/test-frameforge-validator
./build/bin/test-frameforge-json-loader
./build/bin/test-frameforge-new-features
```

## Future Enhancements

Potential future additions:
- Verb disambiguation algorithm using optional parameters
- Dynamic verb loading/reloading without restart
- Verb precedence rules for ambiguous commands
- Custom parameter validation rules per verb
- Multi-language verb aliases

## Files Changed

- `tools/frameforge/frameforge-schema.h`
- `tools/frameforge/frameforge-schema.cpp`
- `tools/frameforge/frameforge-validator.cpp`
- `tools/frameforge/frameforge-json.cpp`
- `tools/frameforge/frameforge-sidecar.cpp`
- `tools/frameforge/README.md`
- `tools/frameforge/verb-definitions.json` (NEW)
- `tests/test-frameforge-validator.cpp`
- `tests/test-frameforge-json-loader.cpp`
- `tests/test-frameforge-new-features.cpp` (NEW)
- `tests/CMakeLists.txt`
