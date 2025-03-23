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

---
### Package llama-server-one Executable

Here are some raw notes that need to be organized.
```
LLAMA_CPP_DIR="llama.cpp"
LLAMA_SERVER_ONE_DIR="llama-server-one"

LLAMA_SERVER="llama-server"
LLAMA_SERVER_ONE="llama-server-one"
LLAMA_SERVER_ONE_ZIP="llama-server-one.zip"
LLAMA_SERVER_ONE_ARGS="llama-server-one-args"

cd ~
mkdir -p $LLAMA_SERVER_ONE_DIR
rm -r ~/$LLAMA_SERVER_ONE_DIR/*
cp ~/$LLAMA_CPP_DIR/$LLAMA_SERVER ~/$LLAMA_SERVER_ONE_DIR/$LLAMA_SERVER_ONE_ZIP

cd ~/$LLAMA_SERVER_ONE_DIR

# delete the /usr directory with all the timezone crap.
zip -d $LLAMA_SERVER_ONE_ZIP "/usr/*"

# archive contents after delete /usr/*.
unzip -l $LLAMA_SERVER_ONE_ZIP 

# add the completion tool to website -- need to decide on front end to add to this repo.
mkdir -p website
cp -r /mnt/hyperv/web-apps/completion-tool/* website
rm website/*.txt
rm website/images/*.psd

ls -al --recursive website

zip -0 -r $LLAMA_SERVER_ONE_ZIP website/*

# archive contents after adding website
unzip -l $LLAMA_SERVER_ONE_ZIP 

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
cp /mnt/hyperv/models/Google-Gemma-1B-Instruct-v3-q8_0.gguf ~/$LLAMA_SERVER_ONE_DIR/model.gguf

$LLAMA_SERVER_ONE
```


