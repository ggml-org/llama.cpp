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
rm -r -f ~/$LLAMA_SERVER_ONE_DIR
mkdir -p $LLAMA_SERVER_ONE_DIR
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
cp -r ~/$LLAMA_CPP_DIR/examples/server/public_legacy/* website
zip -0 -r $LLAMA_SERVER_ONE_ZIP website/*
```

**Optional:** Verify that the archive has your website:
```
unzip -l $LLAMA_SERVER_ONE_ZIP 
```

A `llama-server-one-args` file in the archive can specify sane default parameters. The format of the file is parameter name on a line, parameter value on a line, rinse, repeat. End the file with a `...` line to include user specified parameters.

We don't yet support including the model inside the zip archive (yet). That has a 4GB size limitation on Windows anyway, as `.exe` files cannot exceed 4GB. So let's use an adjacent file called `model.gguf`.

We will serve on localhost, port 8080 by default for safety. The `--ctx-size` parameter is the size of the context window. This is kinda screwy to have as a set size rather than a maximum because the `.gguf` files now have the training context size in metadata. We set it to 8192 to be sensible.
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
...
EOF
```

**Optional:** If you added a website to the archive, use this instead:
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
```

Add the `llama-server-one-args` file to the archive:
```
zip -0 -r $LLAMA_SERVER_ONE_ZIP $LLAMA_SERVER_ONE_ARGS
```

Verify that the archive contains the `llama-server-one-args` file:
```
unzip -l $LLAMA_SERVER_ONE_ZIP 
```

Remove the `.zip` from our working file:
```
mv $LLAMA_SERVER_ONE_ZIP $LLAMA_SERVER_ONE
```

Let's download a small model. We'll use Google Gemma 1B Instruct v3, a surprisingly capable tiny model.
```
MODEL_FILE="Google-Gemma-1B-Instruct-v3-q8_0.gguf"
wget https://huggingface.co/bradhutchings/Brads-LLMs/resolve/main/models/$MODEL_FILE?download=true \
    --show-progress --quiet -O model.gguf
```

Now we can test run `llama-server-one`, listening on localhost:8080.
```
./$LLAMA_SERVER_ONE
```

Hit `ctrl-C` to stop it.

If you'd like it to listen on all available interfaces, so you can connect from a browser on another computer:
```
./$LLAMA_SERVER_ONE --host 0.0.0.0
```
---
Congratulations! You are ready to deploy your `llams-server-one` executable. Follow instructions in [Deploying-ls1.md](Deploying-ls1.md).
