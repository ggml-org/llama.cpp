// Llama Chat (ui/apps/chat) imports the helpers its stories share with these (the Editorial, the state gallery), so
// its svelte-check reads them too and looks upwards from each file for a Svelte config. Without this one it would load
// vite.config.ts two levels up, which needs this package's own dependencies installed (see src/svelte.config.js).
// This package's Storybook is configured in vite.config.ts and .storybook.
export default {};
