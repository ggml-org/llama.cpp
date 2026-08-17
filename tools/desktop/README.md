# llama-desktop

A desktop app for llama.cpp with the goal of making it easy to use for non-technical users.

Implemented as a thin wrapper around `llama-server`, based on Electron.

It eliminates the need to be familiar with the command line.

---

## Table of Contents

- [Usage](#usage)
- [How it works](#how-it-works)
- [Future work](#future-work)
- [Building](#building)

---

## Usage

1. Launch the app
2. Select your models directory
3. [`llama-ui`](../ui/README.md) opens

Optionally: pass CLI flags to the app. They will be forwarded to `llama-server`.

---

## How it works

1. The app starts [`llama-server`](../server/README.md)
2. It loads [`llama-ui`](../ui/README.md) in an Electron `BrowserWindow`

---

## Future work

- Improve the UX of selecting the models directory
    - Persist the user's choice across app launches
    - Maybe start with a default directory (e.g., `app.getPath('userData')`) and let the user change it
    - Have a settings UI to change it later
- Improve app icon design
- [Packaging for Linux](https://www.electronforge.io/config/makers)
- [Code signing](https://www.electronforge.io/guides/code-signing)
- CI builds (require code signing to be useful)
- [Auto updates](https://www.electronforge.io/advanced/auto-update)
- Allow downloading models from Hugging Face to the models directory through `llama-ui`

---

## Building

### Development Build

```bash
cd tools/desktop
npm start -- -- --build-dir <cmake-build-dir>
```

The app will run `llama-server` from the `bin` directory of the specified build directory.

### Production Build

```bash
cmake -B build -DLLAMA_BUILD_DESKTOP=ON
cmake --build build --config Release
```

The build output will be in `<build-dir>/tools/desktop`.
