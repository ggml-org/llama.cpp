/**
 * File validation utilities based on model modalities
 * Ensures only compatible file types are processed based on model capabilities
 */

import { getFileTypeCategory } from '$lib/utils';
import { FileTypeCategory } from '$lib/enums';
import type { ModalityCapabilities } from '$lib/types';

/**
 * Check if a file type is supported by the given modalities
 * @param filename - The filename to check
 * @param mimeType - The MIME type of the file
 * @param capabilities - The modality capabilities to check against
 * @returns true if the file type is supported
 */
export function isFileTypeSupportedByModel(
	filename: string,
	mimeType: string | undefined,
	capabilities: ModalityCapabilities
): boolean {
	const category = mimeType ? getFileTypeCategory(mimeType) : null;

	// If we can't determine the category from MIME type, fall back to general support check
	if (!category) {
		// For unknown types, only allow if they might be text files
		// This is a conservative approach for edge cases
		return true; // Let the existing isFileTypeSupported handle this
	}

	switch (category) {
		case FileTypeCategory.TEXT:
			// Text files are always supported
			return true;

		case FileTypeCategory.PDF:
			// PDFs are always supported (will be processed as text for non-vision models)
			return true;

		case FileTypeCategory.IMAGE:
			// Images require vision support
			return capabilities.hasVision;

		case FileTypeCategory.AUDIO:
			// Audio files require audio support
			return capabilities.hasAudio;

		default:
			// Unknown categories - be conservative and allow
			return true;
	}
}

/**
 * Filter files based on model modalities and return supported/unsupported lists
 * @param files - Array of files to filter
 * @param capabilities - The modality capabilities to check against
 * @returns Object with supportedFiles and unsupportedFiles arrays
 */
export function filterFilesByModalities(
	files: File[],
	capabilities: ModalityCapabilities
): {
	supportedFiles: File[];
	unsupportedFiles: File[];
	modalityReasons: Record<string, string>;
} {
	const supportedFiles: File[] = [];
	const unsupportedFiles: File[] = [];
	const modalityReasons: Record<string, string> = {};

	const { hasVision, hasAudio } = capabilities;

	for (const file of files) {
		const category = getFileTypeCategory(file.type);
		let isSupported = true;
		let reason = '';

		switch (category) {
			case FileTypeCategory.IMAGE:
				if (!hasVision) {
					isSupported = false;
					reason = 'Images require a vision-capable model';
				}
				break;

			case FileTypeCategory.AUDIO:
				if (!hasAudio) {
					isSupported = false;
					reason = 'Audio files require an audio-capable model';
				}
				break;

			case FileTypeCategory.TEXT:
			case FileTypeCategory.PDF:
				// Always supported
				break;

			default:
				// For unknown types, check if it's a generally supported file type
				// This handles edge cases and maintains backward compatibility
				break;
		}

		if (isSupported) {
			supportedFiles.push(file);
		} else {
			unsupportedFiles.push(file);
			modalityReasons[file.name] = reason;
		}
	}

	return { supportedFiles, unsupportedFiles, modalityReasons };
}

/**
 * Generate a user-friendly error message for unsupported files
 * @param unsupportedFiles - Array of unsupported files
 * @param modalityReasons - Reasons why files are unsupported
 * @param capabilities - The modality capabilities to check against
 * @returns Formatted error message
 */
export function generateModalityErrorMessage(
	unsupportedFiles: File[],
	modalityReasons: Record<string, string>,
	capabilities: ModalityCapabilities
): string {
	if (unsupportedFiles.length === 0) return '';

	const { hasVision, hasAudio } = capabilities;

	let message = '';

	if (unsupportedFiles.length === 1) {
		const file = unsupportedFiles[0];
		const reason = modalityReasons[file.name];
		message = `The file "${file.name}" cannot be uploaded: ${reason}.`;
	} else {
		const fileNames = unsupportedFiles.map((f) => f.name).join(', ');
		message = `The following files cannot be uploaded: ${fileNames}.`;
	}

	// Add helpful information about what is supported
	const supportedTypes: string[] = ['text files', 'PDFs'];
	if (hasVision) supportedTypes.push('images');
	if (hasAudio) supportedTypes.push('audio files');

	message += ` This model supports: ${supportedTypes.join(', ')}.`;

	return message;
}

/**
 * Generate file input accept string based on model modalities
 * @param capabilities - The modality capabilities to check against
 * @returns Accept string for HTML file input element
 */
