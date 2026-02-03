# SRPM for building from source and packaging an RPM for RPM-based distros.
# https://docs.fedoraproject.org/en-US/quick-docs/creating-rpm-packages
# Built and maintained by John Boero - boeroboy@gmail.com
# In honor of Seth Vidal https://www.redhat.com/it/blog/thank-you-seth-vidal

# Notes for llama.cpp:
# 1. Tags are currently based on hash - which will not sort asciibetically.
#    We need to declare standard versioning if people want to sort latest releases.
#    In the meantime, YYYYMMDD format will be used.
# 2. Builds for CUDA/OpenCL support are separate, with different depenedencies.
# 3. NVidia's developer repo must be enabled with nvcc, cublas, clblas, etc installed.
#    Example: https://developer.download.nvidia.com/compute/cuda/repos/fedora37/x86_64/cuda-fedora37.repo
# 4. OpenCL/CLBLAST support simply requires the ICD loader and basic opencl libraries.
#    It is up to the user to install the correct vendor-specific support.

Name:           llama.cpp
Version:        %( date "+%%Y%%m%%d" )
Release:        1%{?dist}
Summary:        CPU Inference of LLaMA model in pure C/C++ (no CUDA/OpenCL)
License:        MIT
Source0:        https://github.com/ggml-org/llama.cpp/archive/refs/heads/master.tar.gz
BuildRequires:  coreutils make gcc-c++ git libstdc++-devel
Requires:       libstdc++
URL:            https://github.com/ggml-org/llama.cpp

%define debug_package %{nil}
%define source_date_epoch_from_changelog 0

%description
CPU inference for Meta's Lllama2 models using default options.
Models are not included in this package and must be downloaded separately.

%prep
%setup -n llama.cpp-master

%build
make -j

%install
mkdir -p %{buildroot}%{_bindir}/
cp -p llama-cli %{buildroot}%{_bindir}/llama-cli
cp -p llama-completion %{buildroot}%{_bindir}/llama-completion
cp -p llama-server %{buildroot}%{_bindir}/llama-server
cp -p llama-simple %{buildroot}%{_bindir}/llama-simple

mkdir -p %{buildroot}/usr/lib/systemd/system
%{__cat} <<EOF  > %{buildroot}/usr/lib/systemd/system/llama.service
[Unit]
Description=Llama.cpp server, CPU only (no GPU support in this build).
After=syslog.target network.target local-fs.target remote-fs.target nss-lookup.target

[Service]
Type=simple
EnvironmentFile=/etc/sysconfig/llama
ExecStart=/usr/bin/llama-server $LLAMA_ARGS
ExecReload=/bin/kill -s HUP $MAINPID
Restart=never

[Install]
WantedBy=default.target
EOF

mkdir -p %{buildroot}/etc/sysconfig
%{__cat} <<EOF  > %{buildroot}/etc/sysconfig/llama
LLAMA_ARGS="-m /opt/llama2/ggml-model-f32.bin"
EOF

%clean
rm -rf %{buildroot}
rm -rf %{_builddir}/*

%files
%{_bindir}/llama-cli
%{_bindir}/llama-completion
%{_bindir}/llama-server
%{_bindir}/llama-simple
/usr/lib/systemd/system/llama.service
%config /etc/sysconfig/llama

%pre

%post

%preun
%postun

%changelog
