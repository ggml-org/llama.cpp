export interface BinaryDetectionOptions {
	/** Number of characters to check from the beginning of the file */
	prefixLength: number;
	/** Maximum ratio of suspicious characters allowed (0.0 to 1.0) */
	suspiciousCharThresholdRatio: number;
	/** Maximum absolute number of null bytes allowed */
	maxAbsoluteNullBytes: number;
}

export const DEFAULT_BINARY_DETECTION_OPTIONS: BinaryDetectionOptions = {
	prefixLength: 1024 * 10, // Check the first 10KB of the string
	suspiciousCharThresholdRatio: 0.15, // Allow up to 15% suspicious chars
	maxAbsoluteNullBytes: 2
};
