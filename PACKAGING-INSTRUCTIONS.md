## Packaging llama-server-one

Brad Hutchings<br/>
brad@bradhutchings.com

This file contains instructions for packaging the `llama-server-one` executable to make it ready to deploy on multiple platforms.

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

A `llama-server-one-args` file in the archive can specify sane default parameters. The format of the file is parameter name on a line, parameter value on a line, rinse, repeat. End the file with a `...` line to include user specified parameters.

We don't yet support including the model inside the zip archive (yet). That has a size limitation on Windows anyway. So let's use an adjacent file called `model.gguf`.

We will server on localhost, port 8080 by default for safety. We'll include the optional website we defined above. If you didn't include a custom UI, omit the `--path` and `/zip/website` lines.

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
```

Verify that the archive contains the `llama-server-one-args` file:
```
unzip -l $LLAMA_SERVER_ONE_ZIP 
```



Here are some more raw notes that need to be organized.
```

# Link to download: https://huggingface.co/bradhutchings/Brads-LLMs/resolve/main/models/Apple-OpenELM-1.1B-Instruct-q8_0.gguf?download=true

mv $LLAMA_SERVER_ONE_ZIP $LLAMA_SERVER_ONE
# maybe get the model from my HF repo?
# Do I need to be logged in to HF?
cp /mnt/hyperv/models/Google-Gemma-1B-Instruct-v3-q8_0.gguf \
    ~/$LLAMA_SERVER_ONE_DIR/model.gguf

# Test launch. It should load the model and listen.
./$LLAMA_SERVER_ONE
```
