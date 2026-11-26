// What `@ggml-org/llama-harness` resolves to in a chat-only build
// (VITE_PUBLIC_AGENTIC=false, see vite.config.ts): the trace phrasing, which
// still draws the tool calls of conversations made by a full build, and no
// loop. The chat store never starts a flow in such a build.

export * from '@ggml-org/llama-harness/trace';

export function createHarness(): never {
	throw new Error('This build of Llama Chat has no agentic loop.');
}
