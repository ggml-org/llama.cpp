import {
	AUDIO_FILE_TYPES,
	IMAGE_FILE_TYPES,
	PDF_FILE_TYPES,
	TEXT_FILE_TYPES
} from '$lib/constants/supported-file-types';
import { FileTypeCategory } from '$lib/enums/files';

export function getFileTypeCategory(mimeType: string): FileTypeCategory | null {
	if (
		Object.values(IMAGE_FILE_TYPES).some((type) =>
			(type.mimeTypes as readonly string[]).includes(mimeType)
		)
	) {
		return FileTypeCategory.IMAGE;
	}

	if (
		Object.values(AUDIO_FILE_TYPES).some((type) =>
			(type.mimeTypes as readonly string[]).includes(mimeType)
		)
	) {
		return FileTypeCategory.AUDIO;
	}

	if (
		Object.values(PDF_FILE_TYPES).some((type) =>
			(type.mimeTypes as readonly string[]).includes(mimeType)
		)
	) {
		return FileTypeCategory.PDF;
	}

	if (
		Object.values(TEXT_FILE_TYPES).some((type) =>
			(type.mimeTypes as readonly string[]).includes(mimeType)
		)
	) {
		return FileTypeCategory.TEXT;
	}

	return null;
}

export function getFileTypeByExtension(filename: string): string | null {
	const extension = filename.toLowerCase().substring(filename.lastIndexOf('.'));

	for (const [key, type] of Object.entries(IMAGE_FILE_TYPES)) {
		if ((type.extensions as readonly string[]).includes(extension)) {
			return `${FileTypeCategory.IMAGE}:${key}`;
		}
	}

	for (const [key, type] of Object.entries(AUDIO_FILE_TYPES)) {
		if ((type.extensions as readonly string[]).includes(extension)) {
			return `${FileTypeCategory.AUDIO}:${key}`;
		}
	}

	for (const [key, type] of Object.entries(PDF_FILE_TYPES)) {
		if ((type.extensions as readonly string[]).includes(extension)) {
			return `${FileTypeCategory.PDF}:${key}`;
		}
	}

	for (const [key, type] of Object.entries(TEXT_FILE_TYPES)) {
		if ((type.extensions as readonly string[]).includes(extension)) {
			return `${FileTypeCategory.TEXT}:${key}`;
		}
	}

	return null;
}

export function isFileTypeSupported(filename: string, mimeType?: string): boolean {
	if (mimeType && getFileTypeCategory(mimeType)) {
		return true;
	}

	return getFileTypeByExtension(filename) !== null;
}
