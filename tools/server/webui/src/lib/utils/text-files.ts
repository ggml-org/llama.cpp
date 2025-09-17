/**
 * Text file processing utilities
 * Handles text file detection, reading, and validation
 */

import { FileExtensionText } from '$lib/enums/files';

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
 * @returns True if the content appears to be text-based
 */
export function isLikelyTextFile(content: string): boolean {
	if (!content) return true;

	const sample = content.substring(0, 1000);

	let suspiciousCount = 0;
	let nullCount = 0;

	for (let i = 0; i < sample.length; i++) {
		const charCode = sample.charCodeAt(i);

		// Count null bytes
		if (charCode === 0) {
			nullCount++;
			suspiciousCount++;

			continue;
		}

		// Count suspicious control characters (excluding common ones like tab, newline, carriage return)
		if (charCode < 32 && charCode !== 9 && charCode !== 10 && charCode !== 13) {
			suspiciousCount++;
		}

		// Count replacement characters (indicates encoding issues)
		if (charCode === 0xfffd) {
			suspiciousCount++;
		}
	}

	// Reject if too many null bytes or suspicious characters
	if (nullCount > 2) return false;
	if (suspiciousCount / sample.length > 0.1) return false;

	return true;
}
