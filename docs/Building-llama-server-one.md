## Build Instructions

Brad Hutchings<br/>
brad@bradhutchings.com

This file contains instructions for building llama.cpp with cosmocc to yield a `llama-server` executable that will run on multiple platforms.

---
### Build Dependencies
I build with a freshly installed Ubuntu 24.04 VM. Here are some packages that are helpful in creating a working build system. You may need to install more.
```
sudo apt install -y git python3-pip build-essential zlib1g-dev \
    libffi-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev \
    liblzma-dev tk-dev python3-tk cmake zip
```

### Clone this Repo Locally
Clone this repo into a `~\llama.cpp` directory.
```
cd ~
git clone https://github.com/BradHutchings/llama-server-one.git llama.cpp
```

Use the `changes-1` branch while I test. **Delete this before merging!**
```
cd ~/llama.cpp
git checkout changes-1
```

### Make llama.cpp
We use the old `Makefile` rather than CMake. We've updated the `Makefile` in this repo to build llama.cpp correctly.
```
cd ~/llama.cpp
export LLAMA_MAKEFILE=1
make
```

### Install Cosmo
```
mkdir -p cosmocc
cd cosmocc
wget https://cosmo.zip/pub/cosmocc/cosmocc.zip
unzip cosmocc.zip
rm cosmocc.zip
cd ..
```

### Prepare to make llama.cpp with Cosmo
```
export PATH="$(pwd)/cosmocc/bin:$PATH"
export CC="cosmocc -I$(pwd)/cosmocc/include -L$(pwd)/cosmocc/lib"
export CXX="cosmocc -I$(pwd)/cosmocc/include \
    -I$(pwd)/cosmocc/include/third_party/libcxx \
    -L$(pwd)/cosmocc/lib"
export UNAME_S="cosmocc"
export UNAME_P="cosmocc"
export UNAME_M="cosmocc"
```

### Make llama.cpp with Cosmo
```
make clean
make
```

At this point, you should see llama-server and other built binaries in the directort listing.
```
ls -al
```

`llama-server` is actually a zip acrhive with an APE loader prefix. Let's verify the zip archive part:
```
unzip -l llama-server
```

---
### Packaging llama-server-one

Now that you've built `llama-server`, you're ready to package it as `llama-server-one`. Follow instructions in [Packaging-llama-server-one.md](Packaging-llama-server-one.md).

