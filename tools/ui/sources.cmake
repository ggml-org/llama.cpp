# Inputs used to decide whether the npm build output is up-to-date.

set(UI_SOURCE_GLOBS
    src/*
    static/*
)

set(UI_SOURCE_FILES
    package.json
    package-lock.json
    vite.config.ts
    svelte.config.js
    tsconfig.json
    scripts/vite-plugin-llama-cpp-build.ts
)
