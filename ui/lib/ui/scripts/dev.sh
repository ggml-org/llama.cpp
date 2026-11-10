#!/bin/bash

# Development script for @ggml/llama-ui (a Svelte library, not an app).
#
# Storybook on :6006 is the primary surface for building and testing components.
# The SvelteKit sandbox (src/routes) runs via `vp dev` on :5173 as a demo harness.
# To develop against the built package output from another repo, also run:
#   npm run watch:package
#
# Usage:
#   npm run dev

echo "Starting development servers..."

storybook dev -p 6006 --ci &
vp dev &

# Wait for all background processes
wait
