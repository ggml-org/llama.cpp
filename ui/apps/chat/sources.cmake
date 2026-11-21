# Inputs used to decide whether the npm build output is up-to-date.

set(UI_SOURCE_GLOBS
    src/*
    static/*
    # Llama UI, which the app compiles from source
    ../../lib/ui/src/lib/*
    # Llama Voice, also compiled from source
    ../../lib/voice/src/*
)

set(UI_SOURCE_FILES
    package.json
    package-lock.json
    src/.gitignore
    vite.config.ts
    tsconfig.json
    scripts/llama-ui.js
    scripts/vite-plugin-llama-ui.ts
    ../../lib/ui/package.json
    scripts/vite-plugin-llama-cpp-build.ts
)
