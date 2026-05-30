import { Llama } from "./index.js";

const modelPath = process.argv[2];

if (!modelPath) {
  console.log("Usage: bun example.js <path-to-model>");
  console.log("Example: bun example.js /path/to/llama-2-7b-chat.Q4_K_M.gguf");
  process.exit(1);
}

const llama = new Llama({
  nCtx: 2048,
  nBatch: 512,
  nThreads: 4,
  nGpuLayers: 100,
  logLevel: "silent"
});

console.log("Loading model...");
llama.loadModel(modelPath);
console.log("Model loaded successfully!");

const prompt = "你好";
console.log(`\nPrompt: ${prompt}`);
console.log("\nResponse:");

let fullResponse = "";

for (const token of llama.generate(prompt, { maxTokens: 100 })) {
  process.stdout.write(token);
  fullResponse += token;
}

console.log("\n\n---\nStreaming response with callback:");

await llama.generateStream("你好，请简单介绍一下你自己：", (token) => {
  process.stdout.write(token);
}, { maxTokens: 50 });

console.log("\n\nUnloading model...");
llama.unloadModel();
console.log("Model unloaded!");

Llama.free();
