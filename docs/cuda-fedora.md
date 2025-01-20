# Setting Up CUDA on Fedora

In this guide we setup [Nvidia CUDA](https://docs.nvidia.com/cuda/) in a toolbox container. This guide is applicable for:
- [Fedora Workstation](https://fedoraproject.org/workstation/)
- [Atomic Desktops for Fedora](https://fedoraproject.org/atomic-desktops/)
- [Fedora Spins](https://fedoraproject.org/spins)
- [Other Distributions](https://containertoolbx.org/distros/), including `Red Hat Enterprise Linux >= 8.`, `Arch Linux`, and `Ubuntu`.


## Table of Contents

- [Prerequisites](#prerequisites)
- [Monitoring NVIDIA CUDA Repositories](#monitoring-nvidia-cuda-repositories)
- [Using the Fedora 39 CUDA Repository](#using-the-fedora-39-cuda-repository)
- [Creating a Fedora Toolbox Environment](#creating-a-fedora-toolbox-environment)
- [Installing Essential Development Tools](#installing-essential-development-tools)
- [Adding the CUDA Repository](#adding-the-cuda-repository)
- [Installing `nvidia-driver-libs`](#installing-nvidia-driver-libs)
- [Manually Resolving Package Conflicts](#manually-resolving-package-conflicts)
- [Finalizing the Installation of `nvidia-driver-libs`](#finalizing-the-installation-of-nvidia-driver-libs)
- [Installing the CUDA Meta-Package](#installing-the-cuda-meta-package)
- [Configuring the Environment](#configuring-the-environment)
- [Verifying the Installation](#verifying-the-installation)
- [Conclusion](#conclusion)
- [Troubleshooting](#troubleshooting)
- [Additional Notes](#additional-notes)
- [References](#references)

## Prerequisites

- **Toolbox Installed on the Host System** `Fedora Silverblue` and `Fedora Workstation` both have toolbox by default, other distributions may need to install the [toolbox package](https://containertoolbx.org/install/).
- **NVIDIA Drivers and Graphics Card installed on Host System (optional)** To run CUDA program, such as `llama.cpp`, the host should be setup to access your NVIDIA hardware. Fedora Hosts can use the [RPM Fusion Repository](https://rpmfusion.org/Howto/NVIDIA).
- **Internet connectivity** to download packages.

### Monitoring NVIDIA CUDA Repositories

Before proceeding, it is advisable to check if NVIDIA has updated their CUDA repositories for your Fedora version. NVIDIA's repositories can be found at:

- [Fedora 40 CUDA Repository](https://developer.download.nvidia.com/compute/cuda/repos/fedora40/x86_64/)
- [Fedora 41 CUDA Repository](https://developer.download.nvidia.com/compute/cuda/repos/fedora41/x86_64/)

As of the latest update, these repositories do not contain the `cuda` meta-package or are missing essential components.

### Using the Fedora 39 CUDA Repository

Since the newer repositories are incomplete, we'll use the Fedora 39 repository:

- [Fedora 39 CUDA Repository](https://developer.download.nvidia.com/compute/cuda/repos/fedora39/x86_64/)

**Note:** Fedora 39 is no longer maintained, so we recommend using a toolbox environment to prevent system conflicts.

## Creating a Fedora Toolbox Environment

This guide focuses on Fedora hosts, but with small adjustments, it can work for other hosts. Using a Fedora 39 toolbox allows us to install the necessary packages without affecting the host system.

**Note:** Toolbox is available for other systems, and even without Toolbox, it is possible to use Podman or Docker.

We do not recommend installing on the host system, as Fedora 39 is out-of-maintenance, and instead you should upgrade to a maintained version of Fedora for your host.

1. **Create a Fedora 39 Toolbox:**

   ```bash
   toolbox create --image registry.fedoraproject.org/fedora-toolbox:39 --container fedora-toolbox-39-cuda
   ```

2. **Enter the Toolbox:**

   ```bash
   toolbox enter --container fedora-toolbox-39-cuda
   ```

   Inside the toolbox, you have root privileges and can install packages without affecting the host system.

## Installing Essential Development Tools

1. **Synchronize the DNF Package Manager:**

   ```bash
   sudo dnf distro-sync
   ```

2. **Install the Default Text Editor (Optional):**

   ```bash
   sudo dnf install vim-default-editor --allowerasing
   ```

   The `--allowerasing` flag resolves any package conflicts.

3. **Install Development Tools and Libraries:**

   ```bash
   sudo dnf install @c-development @development-tools cmake
   ```

   This installs essential packages for compiling software, including `gcc`, `make`, and other development headers.

## Adding the CUDA Repository

Add the NVIDIA CUDA repository to your DNF configuration:

```bash
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora39/x86_64/cuda-fedora39.repo
```

After adding the repository, synchronize the package manager again:

```bash
sudo dnf distro-sync
```

## Installing `nvidia-driver-libs`

Attempt to install `nvidia-driver-libs`:

```bash
sudo dnf install nvidia-driver-libs
```

**Explanation:**

- `nvidia-driver-libs` contains necessary NVIDIA driver libraries required by CUDA.
- This step might fail due to conflicts with existing NVIDIA drivers on the host system.

## Manually Resolving Package Conflicts

If the installation fails due to conflicts, we'll manually download and install the required packages, excluding conflicting files.

### 1. Download the `nvidia-driver-libs` RPM

```bash
sudo dnf download --arch x86_64 nvidia-driver-libs
```

You should see a file similar to:

```
nvidia-driver-libs-560.35.05-1.fc39.x86_64.rpm
```

### 2. Attempt to Install the RPM

```bash
sudo dnf install nvidia-driver-libs-560.35.05-1.fc39.x86_64.rpm
```

**Expected Error:**

Installation may fail with errors pointing to conflicts with `egl-gbm` and `egl-wayland`.

**Note: It is important to carefully read the error messages to identify the exact paths that need to be excluded.**

### 3. Download Dependencies

```bash
sudo dnf download --arch x86_64 egl-gbm egl-wayland
```

### 4. Install `egl-gbm` with Excluded Paths

Exclude conflicting files during installation:

```bash
sudo rpm --install --verbose --hash \
  --excludepath=/usr/lib64/libnvidia-egl-gbm.so.1.1.2 \
  --excludepath=/usr/share/egl/egl_external_platform.d/15_nvidia_gbm.json \
  egl-gbm-1.1.2^20240919gitb24587d-3.fc39.x86_64.rpm
```

**Explanation:**

- The `--excludepath` option skips installing files that conflict with existing files.
- Adjust the paths based on the error messages you receive.

### 5. Install `egl-wayland` with Excluded Paths

```bash
sudo rpm --install --verbose --hash \
  --excludepath=/usr/share/egl/egl_external_platform.d/10_nvidia_wayland.json \
  egl-wayland-1.1.17^20241118giteeb29e1-5.fc39.x86_64.rpm
```

### 6. Install `nvidia-driver-libs` with Excluded Paths

```bash
sudo rpm --install --verbose --hash \
  --excludepath=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
  --excludepath=/usr/share/nvidia/nvoptix.bin \
  nvidia-driver-libs-560.35.05-1.fc39.x86_64.rpm
```

**Note:**

- Replace the paths with the ones causing conflicts in your installation if they differ.
- The `--verbose` and `--hash` options provide detailed output during installation.

## Finalizing the Installation of `nvidia-driver-libs`

After manually installing the dependencies, run:

```bash
sudo dnf install nvidia-driver-libs
```

You should receive a message indicating the package is already installed:

```
Package nvidia-driver-libs-3:560.35.05-1.fc39.x86_64 is already installed.
Dependencies resolved.
Nothing to do.
Complete!
```

## Installing the CUDA Meta-Package

Now that the driver libraries are installed, proceed to install CUDA:

```bash
sudo dnf install cuda
```

This installs the CUDA toolkit and associated packages.

## Configuring the Environment

To use CUDA, add its binary directory to your system's `PATH`.

1. **Create a Profile Script:**

   ```bash
   sudo sh -c 'echo "export PATH=\$PATH:/usr/local/cuda/bin" >> /etc/profile.d/cuda.sh'
   ```

   **Explanation:**

   - We add to  `/etc/profile.d/` as the `/etc/` folder is unique to this particular container, and is not shared with other containers or the host system.
   - The backslash `\` before `$PATH` ensures the variable is correctly written into the script.

2. **Make the Script Executable:**

   ```bash
   sudo chmod +x /etc/profile.d/cuda.sh
   ```

3. **Source the Script to Update Your Environment:**

   ```bash
   source /etc/profile.d/cuda.sh
   ```

   **Note:** This command updates your current shell session with the new `PATH`. The `/etc/profile.d/cuda.sh` script ensures that the CUDA binaries are available in your `PATH` for all future sessions.

## Verifying the Installation

To confirm that CUDA is correctly installed and configured, check the version of the NVIDIA CUDA Compiler (`nvcc`):

```bash
nvcc --version
```

You should see output similar to:

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Oct_29_23:50:19_PDT_2024
Cuda compilation tools, release 12.6, V12.6.85
Build cuda_12.6.r12.6/compiler.35059454_0
```

This output confirms that the CUDA compiler is accessible and indicates the installed version.

## Conclusion

You have successfully set up CUDA on Fedora within a toolbox environment using the Fedora 39 CUDA repository. By manually resolving package conflicts and configuring the environment, you can develop CUDA applications without affecting your host system.

## Troubleshooting

- **Installation Failures:**
  - If you encounter errors during installation, carefully read the error messages. They often indicate conflicting files or missing dependencies.
  - Use the `--excludepath` option with `rpm` to exclude conflicting files during manual installations.

- **Driver Conflicts:**
  - Since the host system may already have NVIDIA drivers installed, conflicts can arise. Using the toolbox environment helps isolate these issues.

- **Environment Variables Not Set:**
  - If `nvcc` is not found after installation, ensure that `/usr/local/cuda/bin` is in your `PATH`.
  - Run `echo $PATH` to check if the path is included.
  - Re-source the profile script or open a new terminal session.

## Additional Notes

- **Updating CUDA in the Future:**
  - Keep an eye on the official NVIDIA repositories for updates to your Fedora version.
  - When an updated repository becomes available, adjust your `dnf` configuration accordingly.

- **Building `llama.cpp`:**
  - With CUDA installed, you can follow these [build instructions for `llama.cpp`](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md) to compile it with CUDA support.
  - Ensure that any CUDA-specific build flags or paths are correctly set in your build configuration.

- **Using the Toolbox Environment:**
  - The toolbox environment is isolated from your host system, which helps prevent conflicts.
  - Remember that system files and configurations inside the toolbox are separate from the host. By default the home directory of the user is shared between the host and the toolbox.

---

**Disclaimer:** Manually installing and modifying system packages can lead to instability of the container. The above steps are provided as a guideline and may need adjustments based on your specific system configuration. Always back up important data before making significant system changes, especially as your home folder is writable and shared with he toolbox.

**Acknowledgments:** Special thanks to the Fedora community and NVIDIA documentation for providing resources that assisted in creating this guide.

## References

- [Fedora Toolbox Documentation](https://docs.fedoraproject.org/en-US/fedora-silverblue/toolbox/)
- [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [Podman Documentation](https://podman.io/get-started)

---
