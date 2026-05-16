import { MimeTypeImage } from '$lib/enums';

/**
 * Converts a PNG base64 data URL to another PNG data URL with capped dimensions to reduce file size.
 * @param base64UrlPng - The PNG base64 data URL to convert
 * @param maxMegapixels - The maximum megapixels for the output PNG (default: 1.048576 MP for 1024x1024)
 * @returns Promise resolving to PNG data URL
 */
export function capPngDataURLSize(
	base64UrlPng: string,
	maxMegapixels: number = 1.048576
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

				const targetWidth = img.naturalWidth;
				const targetHeight = img.naturalHeight;
				const totalPixels = targetWidth * targetHeight;
				const maxPixels = Math.floor(maxMegapixels * 1_000_000);

				if (maxPixels > 0 && totalPixels > maxPixels) {
					const scaleFactor = Math.sqrt(maxPixels / totalPixels);
					canvas.width = Math.floor(targetWidth * scaleFactor);
					canvas.height = Math.floor(targetHeight * scaleFactor);
				} else {
					canvas.width = targetWidth;
					canvas.height = targetHeight;
				}

				ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
				resolve(canvas.toDataURL(MimeTypeImage.PNG));
			};

			img.onerror = () => {
				reject(new Error('Failed to load PNG image.'));
			};

			img.src = base64UrlPng;
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			const errorMessage = `Error resizing PNG: ${message}`;
			console.error(errorMessage, error);
			reject(new Error(errorMessage));
		}
	});
}
