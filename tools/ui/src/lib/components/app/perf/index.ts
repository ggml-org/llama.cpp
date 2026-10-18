/**
 *
 * PERF
 *
 * The live performance panel (Display settings > Display performance).
 *
 */

/**
 * **PerfPanel** - CPU, GPU, memory, generation / prompt speed, NVMe bandwidth and expert prediction figures
 *
 * Reads perfStore, which polls GET /perf. `variant="card"`: the desktop dashboard in the bottom-right stack,
 * hover or click "details" for the expanded figures; `variant="strip"`: one compact line above the phone
 * composer, tap to expand.
 */
export { default as PerfPanel } from './PerfPanel.svelte';

/**
 * **PerfDock** - Mounted once by +layout.svelte: switches perfStore with the setting, and shows the card in the
 * right margin of wide screens (`deviceStore.isWide`), below dialogs (z-40)
 */
export { default as PerfDock } from './PerfDock.svelte';
