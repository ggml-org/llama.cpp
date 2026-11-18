#!/usr/bin/env node
// `npx @ggml-org/llama-chat` and `llama-chat`, see cli/server.js
import { main } from '../cli/server.js';

await main(process.argv.slice(2));
