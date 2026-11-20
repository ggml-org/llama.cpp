// lowlight without its bundled language sets, for lazy-highlight.ts to load with the first code block: a static
// re-export of createLowlight alone lets the build drop lowlight's `common` and `all`
export { createLowlight } from 'lowlight';
