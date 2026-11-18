// Apps in ui/apps compile these sources themselves, so their svelte-check and editors read
// them too and look upwards from each file for a Svelte config. Without this one they would
// load vite.config.ts a level up, which needs this package's own dependencies installed.
// The package itself is configured in vite.config.ts.
export default {};
