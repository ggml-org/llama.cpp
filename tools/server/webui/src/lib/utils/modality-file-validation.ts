/**
 * File validation utilities based on model modalities
 * Ensures only compatible file types are processed based on model capabilities
 */

import { getFileTypeCategory } from '$lib/utils/file-type';
import { supportsVision, supportsAudio } from '$lib/stores/server.svelte';
import {
	FileExtensionAudio,
	FileExtensionImage,
	FileExtensionPdf,
	FileExtensionText,
	MimeTypeAudio,
	MimeTypeImage,
	MimeTypeApplication,
	MimeTypeText,
	FileTypeCategory
} from '$lib/enums/files';

/**
 * Check if a file type is supported by the current model's modalities
 * @param filename - The filename to check
 * @param mimeType - The MIME type of the file
 * @returns true if the file type is supported by the current model
 */
export function isFileTypeSupportedByModel(filename: string, mimeType?: string): boolean {
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
			return supportsVision();

		case FileTypeCategory.AUDIO:
			// Audio files require audio support
			return supportsAudio();

		default:
			// Unknown categories - be conservative and allow
			return true;
	}
}

/**
 * Filter files based on model modalities and return supported/unsupported lists
 * @param files - Array of files to filter
 * @returns Object with supportedFiles and unsupportedFiles arrays
 */
export function filterFilesByModalities(files: File[]): {
	supportedFiles: File[];
	unsupportedFiles: File[];
	modalityReasons: Record<string, string>;
} {
	const supportedFiles: File[] = [];
	const unsupportedFiles: File[] = [];
	const modalityReasons: Record<string, string> = {};

	const hasVision = supportsVision();
	const hasAudio = supportsAudio();

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
 * @returns Formatted error message
 */
export function generateModalityErrorMessage(
	unsupportedFiles: File[],
	modalityReasons: Record<string, string>
): string {
	if (unsupportedFiles.length === 0) return '';

	const hasVision = supportsVision();
	const hasAudio = supportsAudio();

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
 * Generate file input accept string based on current model modalities
 * @returns Accept string for HTML file input element
 */
export function generateModalityAwareAcceptString(): string {
	const hasVision = supportsVision();
	const hasAudio = supportsAudio();

	const acceptedExtensions: string[] = [];
	const acceptedMimeTypes: string[] = [];

	// Always include text files and PDFs
	acceptedExtensions.push(...Object.values(FileExtensionText));
	acceptedMimeTypes.push(...Object.values(MimeTypeText));
	acceptedExtensions.push(...Object.values(FileExtensionPdf));
	acceptedMimeTypes.push(...Object.values(MimeTypeApplication));

	// Include images only if vision is supported
	if (hasVision) {
		acceptedExtensions.push(...Object.values(FileExtensionImage));
		acceptedMimeTypes.push(...Object.values(MimeTypeImage));
	}

	// Include audio only if audio is supported
	if (hasAudio) {
		acceptedExtensions.push(...Object.values(FileExtensionAudio));
		acceptedMimeTypes.push(...Object.values(MimeTypeAudio));
	}

	return [...acceptedExtensions, ...acceptedMimeTypes].join(',');
}
