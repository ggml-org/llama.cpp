import { FileExtensionImage, MimeTypeImage } from '$lib/enums';

export function isJxlMimeType(mimeType: string): boolean {
	return mimeType.trim().toLowerCase() === MimeTypeImage.JXL;
}

export function isJxlFile(file: File | { name: string; type: string }): boolean {
	return isJxlMimeType(file.type) || file.name.toLowerCase().endsWith(FileExtensionImage.JXL);
}

/** Force image/jxl MIME on data URLs (browsers often omit it for .jxl files). */
export function normalizeJxlDataUrl(dataUrl: string, filename: string): string {
	if (!filename.toLowerCase().endsWith(FileExtensionImage.JXL)) {
		return dataUrl;
	}

	const comma = dataUrl.indexOf(',');

	if (comma < 0) {
		return dataUrl;
	}

	return `data:${MimeTypeImage.JXL};base64,${dataUrl.slice(comma + 1)}`;
}
