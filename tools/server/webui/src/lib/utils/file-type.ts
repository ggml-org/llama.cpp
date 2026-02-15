import {
	AUDIO_FILE_TYPES,
	IMAGE_FILE_TYPES,
	PDF_FILE_TYPES,
	TEXT_FILE_TYPES
} from '$lib/constants/supported-file-types';
import {
	FileExtensionAudio,
	FileExtensionImage,
	FileExtensionPdf,
	FileExtensionText,
	FileTypeCategory,
	MimeTypeApplication,
	MimeTypeAudio,
	MimeTypeImage,
	MimeTypeText
} from '$lib/enums';

export function getFileTypeCategory(mimeType: string): FileTypeCategory | null {
	switch (mimeType) {
		// Images
		case MimeTypeImage.JPEG:
		case MimeTypeImage.PNG:
		case MimeTypeImage.GIF:
		case MimeTypeImage.WEBP:
		case MimeTypeImage.SVG:
			return FileTypeCategory.IMAGE;

		// Audio
		case MimeTypeAudio.MP3_MPEG:
		case MimeTypeAudio.MP3:
		case MimeTypeAudio.MP4:
		case MimeTypeAudio.WAV:
		case MimeTypeAudio.WEBM:
		case MimeTypeAudio.WEBM_OPUS:
			return FileTypeCategory.AUDIO;

		// PDF
		case MimeTypeApplication.PDF:
			return FileTypeCategory.PDF;

		// Text
		case MimeTypeText.PLAIN:
		case MimeTypeText.MARKDOWN:
		case MimeTypeText.ASCIIDOC:
		case MimeTypeText.JAVASCRIPT:
		case MimeTypeText.JAVASCRIPT_APP:
		case MimeTypeText.TYPESCRIPT:
		case MimeTypeText.JSX:
		case MimeTypeText.TSX:
		case MimeTypeText.CSS:
		case MimeTypeText.HTML:
		case MimeTypeText.JSON:
		case MimeTypeText.XML_TEXT:
		case MimeTypeText.XML_APP:
		case MimeTypeText.YAML_TEXT:
		case MimeTypeText.YAML_APP:
		case MimeTypeText.CSV:
		case MimeTypeText.PYTHON:
		case MimeTypeText.JAVA:
		case MimeTypeText.CPP_SRC:
		case MimeTypeText.C_SRC:
		case MimeTypeText.C_HDR:
		case MimeTypeText.PHP:
		case MimeTypeText.RUBY:
		case MimeTypeText.GO:
		case MimeTypeText.RUST:
		case MimeTypeText.SHELL:
		case MimeTypeText.BAT:
		case MimeTypeText.SQL:
		case MimeTypeText.R:
		case MimeTypeText.SCALA:
		case MimeTypeText.KOTLIN:
		case MimeTypeText.SWIFT:
		case MimeTypeText.DART:
		case MimeTypeText.VUE:
		case MimeTypeText.SVELTE:
		case MimeTypeText.LATEX:
		case MimeTypeText.BIBTEX:
		case MimeTypeText.CUDA:
		case MimeTypeText.CPP_HDR:
		case MimeTypeText.CSHARP:
		case MimeTypeText.HASKELL:
		case MimeTypeText.PROPERTIES:
		case MimeTypeText.TEX:
		case MimeTypeText.TEX_APP:
			return FileTypeCategory.TEXT;

		default:
			return null;
	}
}

export function getFileTypeCategoryByExtension(filename: string): FileTypeCategory | null {
	const extension = filename.toLowerCase().substring(filename.lastIndexOf('.'));

	switch (extension) {
		// Images
		case FileExtensionImage.JPG:
		case FileExtensionImage.JPEG:
		case FileExtensionImage.PNG:
		case FileExtensionImage.GIF:
		case FileExtensionImage.WEBP:
		case FileExtensionImage.SVG:
			return FileTypeCategory.IMAGE;

		// Audio
		case FileExtensionAudio.MP3:
		case FileExtensionAudio.WAV:
			return FileTypeCategory.AUDIO;

		// PDF
		case FileExtensionPdf.PDF:
			return FileTypeCategory.PDF;

		// Text
		case FileExtensionText.TXT:
		case FileExtensionText.MD:
		case FileExtensionText.ADOC:
		case FileExtensionText.JS:
		case FileExtensionText.TS:
		case FileExtensionText.JSX:
		case FileExtensionText.TSX:
		case FileExtensionText.CSS:
		case FileExtensionText.HTML:
		case FileExtensionText.HTM:
		case FileExtensionText.JSON:
		case FileExtensionText.XML:
		case FileExtensionText.YAML:
		case FileExtensionText.YML:
		case FileExtensionText.CSV:
		case FileExtensionText.LOG:
		case FileExtensionText.PY:
		case FileExtensionText.JAVA:
		case FileExtensionText.CPP:
		case FileExtensionText.C:
		case FileExtensionText.H:
		case FileExtensionText.PHP:
		case FileExtensionText.RB:
		case FileExtensionText.GO:
		case FileExtensionText.RS:
		case FileExtensionText.SH:
		case FileExtensionText.BAT:
		case FileExtensionText.SQL:
		case FileExtensionText.R:
		case FileExtensionText.SCALA:
		case FileExtensionText.KT:
		case FileExtensionText.SWIFT:
		case FileExtensionText.DART:
		case FileExtensionText.VUE:
		case FileExtensionText.SVELTE:
		case FileExtensionText.TEX:
		case FileExtensionText.BIB:
		case FileExtensionText.COMP:
		case FileExtensionText.CU:
		case FileExtensionText.CUH:
		case FileExtensionText.HPP:
		case FileExtensionText.HS:
		case FileExtensionText.PROPERTIES:
			return FileTypeCategory.TEXT;

		default:
			return null;
	}
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
	// Images are detected and handled separately for vision models
	if (mimeType) {
		const category = getFileTypeCategory(mimeType);
		if (
			category === FileTypeCategory.IMAGE ||
			category === FileTypeCategory.AUDIO ||
			category === FileTypeCategory.PDF
		) {
			return true;
		}
	}

	// Check extension for known types (especially images without MIME)
	const extCategory = getFileTypeCategoryByExtension(filename);
	if (
		extCategory === FileTypeCategory.IMAGE ||
		extCategory === FileTypeCategory.AUDIO ||
		extCategory === FileTypeCategory.PDF
	) {
		return true;
	}

	// Fallback: treat everything else as text (inclusive by default)
	return true;
}
