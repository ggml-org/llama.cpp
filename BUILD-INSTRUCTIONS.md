## Build Instructions

Brad Hutchings<br/>
brad@bradhutchings.com

This file contains instructions for building llama.cpp with cosmocc, then customizing the `llama-server` executable to create a ready-to-deploy `llama-server-one` executable.

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
### Package llama-server-one Executable

Let's define some environment variables:
```
LLAMA_CPP_DIR="llama.cpp"
LLAMA_SERVER_ONE_DIR="llama-server-one"

LLAMA_SERVER="llama-server"
LLAMA_SERVER_ONE="llama-server-one"
LLAMA_SERVER_ONE_ZIP="llama-server-one.zip"
LLAMA_SERVER_ONE_ARGS="llama-server-one-args"
```

Next, let's create a directory where we'll package up `llama-server-one`:
```
cd ~
mkdir -p $LLAMA_SERVER_ONE_DIR
rm -r ~/$LLAMA_SERVER_ONE_DIR/*
cp ~/$LLAMA_CPP_DIR/$LLAMA_SERVER \
    ~/$LLAMA_SERVER_ONE_DIR/$LLAMA_SERVER_ONE_ZIP

cd ~/$LLAMA_SERVER_ONE_DIR
```

Look at the contents of the `llama-server-one` zip archive:
```
unzip -l $LLAMA_SERVER_ONE_ZIP 
```

You should notice a bunch of extraneous timezone related files in `/usr/*`. Let's get rid of those:
```
zip -d $LLAMA_SERVER_ONE_ZIP "/usr/*"
```

Verify that these files are no longer in the archive:
```
unzip -l $LLAMA_SERVER_ONE_ZIP 
```

**Optional:** `llama.cpp` has a built in chat UI. If you'd like to provide a custom UI, you should add a `website` directory to the `llama-server-one` archive. `llama.cpp`'s chat UI is optimized for serving inside the project's source code. But we can copy the unoptimized source:
```
mkdir -p website
cp -r /mnt/hyperv/web-apps/completion-tool/* website
zip -0 -r $LLAMA_SERVER_ONE_ZIP website/*
```
**Optional:** You don't have source for my Completion Tool UI, but if you did, and it were on a mounted share like on my build system, you would include it like this:
```
mkdir -p website
cp -r /mnt/hyperv/web-apps/completion-tool/* website
rm website/*.txt
rm website/images/*.psd
zip -0 -r $LLAMA_SERVER_ONE_ZIP website/*
```

**Optional:** Verify that the archive has your website:
```
unzip -l $LLAMA_SERVER_ONE_ZIP 
```


Here are some more raw notes that need to be organized.
```
cat << EOF > $LLAMA_SERVER_ONE_ARGS
-m
model.gguf
--host
127.0.0.1
--port
8080
--ctx-size
8192
--path
/zip/website
...
EOF

zip -0 -r $LLAMA_SERVER_ONE_ZIP $LLAMA_SERVER_ONE_ARGS

# archive contents after adding args
unzip -l $LLAMA_SERVER_ONE_ZIP 

mv $LLAMA_SERVER_ONE_ZIP $LLAMA_SERVER_ONE
# maybe get the model from my HF repo?
# Do I need to be logged in to HF?
cp /mnt/hyperv/models/Google-Gemma-1B-Instruct-v3-q8_0.gguf \
    ~/$LLAMA_SERVER_ONE_DIR/model.gguf

# Test launch. It should load the model and listen.
./$LLAMA_SERVER_ONE
```


