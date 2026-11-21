/**
 * Model memory estimation.
 *
 * Mirrors the app's compatibility check (Model+Compatibility.swift): the
 * runtime budget is RAM x 0.75 minus a fixed overhead, and a file fits when
 * its size with headroom stays under that budget. The result is the smallest
 * memory tier that can run the model, so the UI presents an honest machine
 * requirement instead of a raw file size. Context length and
 * device-specific budgets are deliberately ignored - callers present the
 * requirement and let the user judge.
 */
export { minMemoryTierGb } from '@ggml-org/llama-ui/models';
