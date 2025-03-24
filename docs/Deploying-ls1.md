## Deploying llama-server-one

Brad Hutchings<br/>
brad@bradhutchings.com

This file contains instructions for deploying the `llama-server-one` executable. I'm using Ubuntu 24.04.

---
### Deployment Folder
Assuming you packaged as instructed in the [Packaging-ls1.md](Packaging-ls1.md) instructions file, let's create a folder with everything you need to deploy. You can zip this folder to distribute your `llama-server-one`, model, and arguments file for use on any platform. 

Let's define some environment variables:
```
LLAMA_CPP_DIR="llama.cpp"
LLAMA_SERVER_ONE_DIR="llama-server-one"
DEPLOY_DIR="llama-server-one-deploy"
DEPLOY_ZIP="llama-server-one-deploy.zip"

LLAMA_SERVER="llama-server"
LLAMA_SERVER_ONE="llama-server-one"
LLAMA_SERVER_ONE_EXE="llama-server-one.exe"
LLAMA_SERVER_ONE_ARGS="llama-server-one-args"
```

Create a folder and copy `llama-server-one` into the new folder.
```
# This should use variables for paths and filenames. So should the packaging instructions.
cd ~
rm -r -f $DEPLOY_DIR $DEPLOY_ZIP
mkdir -p $DEPLOY_DIR
cd $DEPLOY_DIR
cp ~/$LLAMA_SERVER_ONE_DIR/$LLAMA_SERVER_ONE .
```

On Windows, this executable will need to be renamed to a `.exe` file. Since our executable is small, let's just make a copy of `llama-server-one` with the `.exe` extension.

```
cp $LLAMA_SERVER_ONE $LLAMA_SERVER_ONE_EXE
```

Let's download a small model. We'll use Google Gemma 1B Instruct v3, a surprisingly capable tiny model. We'll keep it's filename and make that work with the `llama-server-args` file (below).
```
MODEL_FILE="Google-Gemma-1B-Instruct-v3-q8_0.gguf"
wget https://huggingface.co/bradhutchings/Brads-LLMs/resolve/main/models/$MODEL_FILE?download=true \
    --show-progress --quiet -O $MODEL_FILE
```

Let's create a `llama-server-one-args` file. These parameters can override or augment the parameters you previously embedded in you `llama-server-one` archive. This file could be edited by the end user to configure llama-file-one without having to construct and type a long command line. Notice that we've overridden the `-m`, `--host`, and `--port` parameters.
```
cat << EOF > $LLAMA_SERVER_ONE_ARGS
-m
$MODEL_FILE
--host
0.0.0.0
--port
8888
...
EOF
```

Now we can test run `llama-server-one`, listening on all network interfaces, port 8888. Note that these are different from the default args you built into `llama-server-one`. You can connect to it from another web browser.
```
./$LLAMA_SERVER_ONE
```

Hit `ctrl-C` to stop it.

Finally, let's zip up the files into a `.zip` file you can share and move it to your home directory. The model won't compress much, so we're turning compression off with the `-0` parameter.

```
zip -0 $DEPLOY_ZIP *
mv $DEPLOY_ZIP ~
cd ~
```



