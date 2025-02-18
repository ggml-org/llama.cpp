# Setting Up CUDA on Fedora

In this guide we setup [Nvidia CUDA](https://docs.nvidia.com/cuda/) in a toolbox container. This guide is applicable for:

- [Fedora Workstation](https://fedoraproject.org/workstation/)
- [Atomic Desktops for Fedora](https://fedoraproject.org/atomic-desktops/)
- [Fedora Spins](https://fedoraproject.org/spins)
- [Other Distributions](https://containertoolbx.org/distros/), including `Red Hat Enterprise Linux >= 8.5`, `Arch Linux`, and `Ubuntu`.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Using the Fedora 41 CUDA Repository](#using-the-fedora-41-cuda-repository)
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
- **NVIDIA Drivers and Graphics Card installed on Host System (recommended)** To run CUDA program, such as `llama.cpp`, the host should be setup to access your NVIDIA hardware. Fedora Hosts can use the [RPM Fusion Repository](https://rpmfusion.org/Howto/NVIDIA).
- **Internet connectivity** to download packages.

### Using the Fedora 41 CUDA Repository

The latest release is 41.

- [Fedora 41 CUDA Repository](https://developer.download.nvidia.com/compute/cuda/repos/fedora41/x86_64/)

**Note:** We recommend using a toolbox environment to prevent system conflicts.

## Creating a Fedora Toolbox Environment

This guide focuses on Fedora hosts, but with small adjustments, it can work for other hosts. Using the Fedora Toolbox allows us to install the necessary packages without affecting the host system.

**Note:** Toolbox is available for other systems, and even without Toolbox, it is possible to use Podman or Docker.

1. **Create a Fedora 41 Toolbox:**

   ```bash
   toolbox create --image registry.fedoraproject.org/fedora-toolbox:41 --container fedora-toolbox-41-cuda
   ```

2. **Enter the Toolbox:**

   ```bash
   toolbox enter --container fedora-toolbox-41-cuda
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

   The `--allowerasing` flag will allow the removal of the conflicting `nano-default-editor` package.

3. **Install Development Tools and Libraries:**

   ```bash
   sudo dnf install @c-development @development-tools cmake
   ```

   This installs essential packages for compiling software, including `gcc`, `make`, and other development headers.

## Adding the CUDA Repository

Add the NVIDIA CUDA repository to your DNF configuration:

```bash
sudo dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora41/x86_64/cuda-fedora41.repo
```

After adding the repository, synchronize the package manager again:

```bash
sudo dnf distro-sync
```

## Installing `nvidia-driver-libs` and `nvidia-driver-cuda-libs`

We need to detect if the host is supplying the [NVIDIA driver libraries into the toolbox](https://github.com/containers/toolbox/blob/main/src/pkg/nvidia/nvidia.go).

```bash
ls -la /usr/lib64/libcuda.so.1
```

**Explanation:**

- `nvidia-driver-libs` and `nvidia-driver-cuda-libs` contains necessary NVIDIA driver libraries required by CUDA,
  on hosts with NVIDIA drivers installed the Fedora Container will supply the host libraries.

### Install Nvidia Driver Libraries on Guest (if `libcuda.so.1` was NOT found).

```bash
sudo dnf install nvidia-driver-libs nvidia-driver-cuda-libs
```

### Manually Updating the RPM database for host-supplied NVIDIA drivers (if `libcuda.so.1` was found).

If the installation fails due to conflicts, we'll manually download and install the required packages, excluding conflicting files.

#### 1. Download `nvidia-driver-libs` and `nvidia-driver-cuda-libs` RPM's (with dependencies)

```bash
sudo dnf download --destdir=/tmp/nvidia-driver-libs --resolve --arch x86_64 nvidia-driver-libs nvidia-driver-cuda-libs
```

#### 2. Update the RPM database to assume the installation of these packages.

```bash
sudo rpm --install --verbose --hash --justdb /tmp/nvidia-driver-libs/*
```

**Note:**

- The `--justdb` option only updates the RPM database, without touching the filesystem.

#### Finalizing the Installation of `nvidia-driver-libs` and `nvidia-driver-cuda-libs`

After manually installing the dependencies, run:

```bash
sudo dnf install nvidia-driver-libs nvidia-driver-cuda-libs
```

You should receive a message indicating the package is already installed:

```
Updating and loading repositories:
Repositories loaded.
Package "nvidia-driver-libs-3:570.86.10-1.fc41.x86_64" is already installed.
Package "nvidia-driver-cuda-libs-3:570.86.10-1.fc41.x86_64" is already installed.

Nothing to do.
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

   - We add to `/etc/profile.d/` as the `/etc/` folder is unique to this particular container, and is not shared with other containers or the host system.
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
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Jan_15_19:20:09_PST_2025
Cuda compilation tools, release 12.8, V12.8.61
Build cuda_12.8.r12.8/compiler.35404655_0
```

This output confirms that the CUDA compiler is accessible and indicates the installed version.

## Conclusion

You have successfully set up CUDA on Fedora within a toolbox environment using the Fedora 41 CUDA repository. By manually updating the RPM db and configuring the environment, you can develop CUDA applications without affecting your host system.

## Troubleshooting

- **Installation Failures:**

  - If you encounter errors during installation, carefully read the error messages. They often indicate conflicting files or missing dependencies.
  - You may use the `--excludepath` option with `rpm` to exclude conflicting files during manual RPM installations.

- **Rebooting the Container:**

  - Sometimes there may be a bug in the NVIDIA driver host passthrough (such as missing a shared library). Rebooting the container may solve this issue:

  ```bash
  # on the host system
  podman container restart --all
  ```

- **Environment Variables Not Set:**
  - If `nvcc` is not found after installation, ensure that `/usr/local/cuda/bin` is in your `PATH`.
  - Run `echo $PATH` to check if the path is included.
  - Re-source the profile script or open a new terminal session.

## Additional Notes

- **Updating CUDA in the Future:**

  - Keep an eye on the official NVIDIA repositories for updates to your Fedora version.
  - When an updated repository becomes available, adjust your `dnf` configuration accordingly.

- **Building `llama.cpp`:**

  - With CUDA installed, you can follow these [build instructions for `llama.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) to compile it with CUDA support.
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
