## Build Instructions

Brad Hutchings<br/>
brad@bradhutchings.com

This file contains instructions for building llama.cpp with cosmocc, then customizing the `llama-server` executable to create a ready-to-deploy `llama-server-one` executable.

---

### Clone this Repo Locally
Clone this repo into a `~\llama.cpp` directory.
```
cd ~
git clone https://github.com/BradHutchings/llama-server-one.git llama.cpp
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
export PATH="~/llama.cpp/cosmocc/bin:$PATH"
export CC="cosmocc -I~/llama.cpp/cosmocc/include -L~/llama.cpp/cosmocc/lib"
export CXX="cosmocc -I~/llama.cpp/cosmocc/include \
    -I~/llama.cpp/cosmocc/include/third_party/libcxx \
    -L~/llama.cpp/cosmocc/lib"
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





