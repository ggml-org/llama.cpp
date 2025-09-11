# LlamaCPP Fork Management Strategy

## Problem Statement

The team maintains a fork of llama.cpp with custom modifications but needs to stay synchronized with the rapidly evolving upstream repository. Traditional PR-based merging creates merge commits or squashes that modify git history, making future upstream synchronization difficult.

## Proposed Solution

### Branch Structure

- **temp-upstream**: Tracks the official upstream llama.cpp repository (pure upstream)
- **temp-latest**: Contains team's custom changes rebased on top of upstream
- **Feature branches**: Built on top of temp-latest for new changes

### Synchronization Process

#### Initial Setup (one-time)
To make things clear, will start from a blank state. You can skip steps you've already done:

1. **Fork the repository**: Fork `qvac-ext-lib-llama.cpp` repo in GitHub (e.g., `https://github.com/olyasir/qvac-ext-lib-llama.cpp`)

2. **Clone locally**:
   ```bash
   git clone git@github.com:olyasir/qvac-ext-lib-llama.cpp.git
   cd qvac-ext-lib-llama.cpp
   ```

3. **Configure remotes**:
   ```bash
   # Add upstream ggml remote
   git remote add ggml git@github.com:ggml-org/llama.cpp.git
   git fetch ggml
   
   # Add tether remote
   git remote add tether git@github.com:tetherto/qvac-ext-lib-llama.cpp.git
   git fetch tether
   ```

#### Regular Synchronization Process

1. **Prepare temp-latest branch**:
   ```bash
   git checkout temp-latest
   git pull
   ```

2. **Rebase onto new upstream tag** (e.g., b6789):
   ```bash
   git rebase b6789
   ```

3. **Resolve conflicts**: Git will stop if it finds conflicts. Resolve them as appropriate (may need to check with original writers)

4. **Push rebased changes**:
   ```bash
   git push -f
   ```

5. **Create and push new tag**:
   ```bash
   git tag b6789.0.0
   # Add description like "Sync with upstream version b6789"
   git push tether tag b6789.0.0
   ```

6. **Test and publish**: Test the new tag, and if it works properly, publish to vcpkg

#### Testing Process

1. **Get test project**: Download the test project from [vcpkg-test-llama-cpp](https://drive.google.com/file/d/1Fm47_QsPsjp-kjPnQpQiRTE5KIrxMh_G/view?usp=sharing) (simple project that depends on the llama-cpp port)

2. **Update vcpkg port**:
   ```bash
   # Copy latest ports/llama-cpp folder from qvac-registry-vcpkg 
   cp -r qvac-registry-vcpkg/ports/llama-cpp vcpkg/ports/llama-cpp
   ```

3. **Update version**: In `vcpkg/ports/llama-cpp/vcpkg.json`, update version number to new tag (without 'b' prefix)
   - For tag `b6789.0.0` → version should be `6789.0.0`

4. **Initial build attempt**:
   ```bash
   bare-make generate
   ```

5. **Fix SHA512 hash**: Configuration will fail with hash mismatch error:
   ```
   error: download from https://github.com/tetherto/qvac-ext-lib-llama.cpp/archive/b6435.0.0.tar.gz had an unexpected hash
   note: Expected: 9baedc3c4ff681222d8fe16ac10346af9cd7fd5a4a6047448f8a3ad0712ba8e35dbd06af16b3a8c6c8b25518b43fd3b152275e90969f0c33cf851cdb44484eb0
   note: Actual  : c869a45e809c367cae6122bfc26c26f16767b010f2da804eb6d20eab8fc9ee8a6fa9c35d04792d0dc1e7483a1b552441027a96ebd30cfb8ac455a3da52801f59
   ```
   Update `vcpkg/ports/llama-cpp/portfile.cmake` - replace SHA512 line with the "Actual" value

6. **Final verification**:
   ```bash
   bare-make generate
   bare-make build
   ```
   If successful, the sync worked properly and you can publish the new vcpkg port version

### Version Management
*(on temp-latest branch)*

- Base versions follow upstream tags (e.g., b5932)
- Extended versions add incremental numbers:
  - **b5932.0.0**: temp-upstream + mtmd changes
  - **b5932.1.0**: temp-upstream + mtmd + load-from-buffer changes
  - And so on...

## Development Workflow

1. **New PRs**: Create against temp-latest (which contains existing custom changes)
2. **After synchronization**: New PRs pointed to temp-latest should include all our changes and the new upstream version
3. **vcpkg integration**: vcpkg registry points to specific commit hashes, not branch names
4. **Testing**: Test new tags before publishing to vcpkg

## Benefits

This strategy ensures the team can maintain their custom modifications while staying current with upstream llama.cpp development.

- Maintains clean git history aligned with upstream
- Enables easy future synchronization with upstream
- Protects against accidental direct merges to main
- Allows multiple teams (including external collaborators like Collabora) to build on top of stable versions
- Custom changes accumulate incrementally while staying current with upstream
