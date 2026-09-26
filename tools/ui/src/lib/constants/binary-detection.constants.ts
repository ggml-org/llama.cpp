import type { BinaryDetectionOptions } from '$lib/types';

export const DEFAULT_BINARY_DETECTION_OPTIONS: BinaryDetectionOptions = {
	maxAbsoluteNullBytes: 2,
	prefixLength: 1024 * 10, // Check the first 10KB of the string
	suspiciousCharThresholdRatio: 0.15 // Allow up to 15% suspicious chars
};
