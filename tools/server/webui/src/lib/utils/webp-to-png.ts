import { FileExtensionImage, MimeTypeImage } from '$lib/enums/files';

/**
 * Convert a WebP base64 data URL to a PNG data URL
 * @param base64UrlWebp - The WebP base64 data URL to convert
 * @param backgroundColor - Background color for the PNG (default: 'white')
 * @returns Promise resolving to PNG data URL
 */
export function webpBase64UrlToPngDataURL(
	base64UrlWebp: string,
	backgroundColor: string = 'white'
): Promise<string> {
	return new Promise((resolve, reject) => {
		try {
			const img = new Image();

			img.onload = () => {
				const canvas = document.createElement('canvas');
				const ctx = canvas.getContext('2d');

				if (!ctx) {
					reject(new Error('Failed to get 2D canvas context.'));
					return;
				}

				const targetWidth = img.naturalWidth || 300;
				const targetHeight = img.naturalHeight || 300;

				canvas.width = targetWidth;
				canvas.height = targetHeight;

				if (backgroundColor) {
					ctx.fillStyle = backgroundColor;
					ctx.fillRect(0, 0, canvas.width, canvas.height);
				}
				ctx.drawImage(img, 0, 0, targetWidth, targetHeight);

				resolve(canvas.toDataURL(MimeTypeImage.PNG));
			};

			img.onerror = () => {
				reject(new Error('Failed to load WebP image. Ensure the WebP data is valid.'));
			};

			img.src = base64UrlWebp;
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			const errorMessage = `Error converting WebP to PNG: ${message}`;
			console.error(errorMessage, error);
			reject(new Error(errorMessage));
		}
	});
}

/**
 * Check if a file is a WebP based on its MIME type
 * @param file - The file to check
 * @returns True if the file is a WebP
 */
export function isWebpFile(file: File): boolean {
	return (
		file.type === MimeTypeImage.WEBP || file.name.toLowerCase().endsWith(FileExtensionImage.WEBP)
	);
}

/**
 * Check if a MIME type represents a WebP
 * @param mimeType - The MIME type to check
 * @returns True if the MIME type is image/webp
 */
export function isWebpMimeType(mimeType: string): boolean {
	return mimeType === MimeTypeImage.WEBP;
}
