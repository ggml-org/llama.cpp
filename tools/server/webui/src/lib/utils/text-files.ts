/**
 * Text file processing utilities
 * Handles text file detection, reading, and validation
 */

import { DEFAULT_BINARY_DETECTION_OPTIONS } from '$lib/constants/binary-detection';
import type { BinaryDetectionOptions } from '$lib/types';
import { FileExtensionText } from '$lib/enums';

/**
 * Check if a filename indicates a text file based on its extension
 * @param filename - The filename to check
 * @returns True if the filename has a recognized text file extension
 */
export function isTextFileByName(filename: string): boolean {
	const textExtensions = Object.values(FileExtensionText);

	return textExtensions.some((ext: FileExtensionText) => filename.toLowerCase().endsWith(ext));
}

/**
 * Read a file's content as text
 * @param file - The file to read
 * @returns Promise resolving to the file's text content
 */
export async function readFileAsText(file: File): Promise<string> {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();

		reader.onload = (event) => {
			if (event.target?.result !== null && event.target?.result !== undefined) {
				resolve(event.target.result as string);
			} else {
				reject(new Error('Failed to read file'));
			}
		};

		reader.onerror = () => reject(new Error('File reading error'));

		reader.readAsText(file);
	});
}

/**
 * Heuristic check to determine if content is likely from a text file
 * Detects binary files by counting suspicious characters and null bytes
 * @param content - The file content to analyze
 * @param options - Optional configuration for detection parameters
 * @returns True if the content appears to be text-based
 */
export function isLikelyTextFile(
	content: string,
	options: Partial<BinaryDetectionOptions> = {}
): boolean {
	if (!content) return true;

	const config = { ...DEFAULT_BINARY_DETECTION_OPTIONS, ...options };
	const sample = content.substring(0, config.prefixLength);

	let nullCount = 0;
	let suspiciousControlCount = 0;

	for (let i = 0; i < sample.length; i++) {
		const charCode = sample.charCodeAt(i);

		// Count null bytes - these are strong indicators of binary files
		if (charCode === 0) {
			nullCount++;

			continue;
		}

		// Count suspicious control characters
		// Allow common whitespace characters: tab (9), newline (10), carriage return (13)
		if (charCode < 32 && charCode !== 9 && charCode !== 10 && charCode !== 13) {
			// Count most suspicious control characters
			if (charCode < 8 || (charCode > 13 && charCode < 27)) {
				suspiciousControlCount++;
			}
		}

		// Count replacement characters (indicates encoding issues)
		if (charCode === 0xfffd) {
			suspiciousControlCount++;
		}
	}

	// Reject if too many null bytes
	if (nullCount > config.maxAbsoluteNullBytes) return false;

	// Reject if too many suspicious characters
	if (suspiciousControlCount / sample.length > config.suspiciousCharThresholdRatio) return false;

	return true;
}
