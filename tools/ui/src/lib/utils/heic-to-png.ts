import { heicTo } from 'heic-to';
import { MimeTypeImage } from '$lib/enums';

/**
 * Convert a HEIC/HEIF file to a PNG data URL
 * @param file - The HEIC/HEIF file to convert
 * @returns Promise resolving to PNG data URL
 */
export async function heicFileToPngDataURL(file: File | Blob): Promise<string> {
	const pngBlob = await heicTo({
		blob: file,
		type: MimeTypeImage.PNG
	});

	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = () => resolve(reader.result as string);
		reader.onerror = () => reject(reader.error);
		reader.readAsDataURL(pngBlob);
	});
}

/**
 * Check if a MIME type represents a HEIC/HEIF image
 * @param mimeType - The MIME type to check
 * @returns True if the MIME type is image/heic or image/heif
 */
export function isHeicMimeType(mimeType: string): boolean {
	const normalized = mimeType.trim().toLowerCase();

	return normalized === MimeTypeImage.HEIC || normalized === MimeTypeImage.HEIF;
}
