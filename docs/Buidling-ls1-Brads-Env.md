## Building llama-server

Brad Hutchings<br/>
brad@bradhutchings.com

This file contains instructions for building `llama.cpp` with `cosmocc` to yield a `llama-server` executable that will run on multiple platforms.

### Environment Variables

Let's define some environment variables, resetting those that affect the Makefile:
```
BUILDING_DIR="1-BUILDING-llama.cpp"
export CC=""
export CXX=""
export AR=""
export UNAME_S=""
export UNAME_P=""
export UNAME_M=""
printf "\n**********\n*\n* FINISHED: Environment Variables.\n*\n**********\n\n"
```

_Note that if you copy each code block from the guide and paste it into your terminal, each block ends with a message so you won't lose your place in this guide._

---
### Build Dependencies
I build with a freshly installed Ubuntu 24.04 VM. Here are some packages that are helpful in creating a working build system. You may need to install more.
```
sudo apt install -y git python3-pip build-essential zlib1g-dev \
    libffi-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev \
    liblzma-dev tk-dev python3-tk cmake zip npm
printf "\n**********\n*\n* FINISHED: Build Dependencies.\n*\n**********\n\n"
```

---
### Clone this Repo Locally
Clone this repo into a `~\llama.cpp` directory.
```
cd ~
git clone https://github.com/BradHutchings/llama-server-one.git $BUILDING_DIR
printf "\n**********\n*\n* FINISHED: Clone this Repo Locally.\n*\n**********\n\n"
```

**Optional:** Use the `work-in-progress` branch where I implement and test my own changes and where I test upstream changes from `llama.cpp`.
```
cd ~/$BUILDING_DIR
git checkout work-in-progress
printf "\n**********\n*\n* FINISHED: Checkout work-in-progress.\n*\n**********\n\n"
```

---
### Customize WebUI
```
APP_NAME='Mmojo Chat'
sed -i -e "s/<title>.*<\/title>/<title>$APP_NAME<\/title>/g" examples/server/webui/index.html
sed -i -e "s/>llama.cpp<\/div>/>$APP_NAME<\/div>/g" examples/server/webui/src/components/Header.tsx
cd examples/server/webui
npm i
npm run build
cd ~/$BUILDING_DIR
printf "\n**********\n*\n* FINISHED: Customize WebUI.\n*\n**********\n\n"
```

---
### Make llama.cpp
We use the old `Makefile` rather than CMake. We've updated the `Makefile` in this repo to build llama.cpp correctly.
```
cd ~/$BUILDING_DIR
export LLAMA_MAKEFILE=1
export LLAMA_SERVER_SSL=ON
make clean
make
printf "\n**********\n*\n* FINISHED: Make llama.cpp.\n*\n**********\n\n"
```

If the build is successful, it will end with this message:

&nbsp;&nbsp;&nbsp;&nbsp;**NOTICE: The 'server' binary is deprecated. Please use 'llama-server' instead.**

If the build fails and you've checked out the `work-in-progress` branch, well, it's in progess, so switch back to the `master` branch and build that.

If the build fails on the `master` branch, please post a note in the [Discussions](https://github.com/BradHutchings/llama-server-one/discussions) area.

#### List Directory

At this point, you should see `llama-server` and other built binaries in the directory listing.
```
ls -al
printf "\n**********\n*\n* FINISHED: List Directory.\n*\n**********\n\n"
```

---
### Install Cosmo
```
mkdir -p cosmocc
cd cosmocc
wget https://cosmo.zip/pub/cosmocc/cosmocc.zip
unzip cosmocc.zip
rm cosmocc.zip
cd ..
printf "\n**********\n*\n* FINISHED: Install Cosmo.\n*\n**********\n\n"
```

---
### Prepare to make llama.cpp with Cosmo
```
export PATH="$(pwd)/cosmocc/bin:$PATH"
export CC="cosmocc -I$(pwd)/cosmocc/include -L$(pwd)/cosmocc/lib"
export CXX="cosmocc -I$(pwd)/cosmocc/include \
    -I$(pwd)/cosmocc/include/third_party/libcxx \
    -L$(pwd)/cosmocc/lib"
export AR="cosmoar"
export UNAME_S="cosmocc"
export UNAME_P="cosmocc"
export UNAME_M="cosmocc"
printf "\n**********\n*\n* FINISHED: Prepare to make llama.cpp with Cosmo.\n*\n**********\n\n"
```

---
### Make llama.cpp with Cosmo
```
make clean
make
printf "\n**********\n*\n* FINISHED: Make llama.cpp with Cosmo\n*\n**********\n\n"
```

If the build is successful, it will end with this message:

&nbsp;&nbsp;&nbsp;&nbsp;**NOTICE: The 'server' binary is deprecated. Please use 'llama-server' instead.**

If the build fails and you've checked out the `work-in-progress` branch, well, it's in progess, so switch back to the `master` branch and build that.

If the build fails on the `master` branch, please post a note in the [Discussions](https://github.com/BradHutchings/llama-server-one/discussions) area.

#### List Directory

At this point, you should see `llama-server` and other built binaries in the directory listing.
```
ls -al
printf "\n**********\n*\n* FINISHED: List Directory.\n*\n**********\n\n"
```

#### Verify Zip Archive

`llama-server` is actually a zip acrhive with an "Actually Portable Executable" (APE) loader prefix. Let's verify the zip archive part:
```
unzip -l llama-server
printf "\n**********\n*\n* FINISHED: Verify Zip Archive.\n*\n**********\n\n"
```

---
### Configuring llama-server-one

Now that you've built `llama-server`, you're ready to configure it as `llama-server-one`. Follow instructions in [Configuring-ls1-Brads-Env.md](Configuring-ls1-Brads-Env.md).
