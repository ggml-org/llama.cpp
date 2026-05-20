import { MEGAPIXELS_TO_PIXELS } from '$lib/constants/image-size';
import { BASE64_IMAGE_URI_REGEX } from '$lib/constants/uri-template';
import { MimeTypeImage } from '$lib/enums';

/**
 * Converts an Image base64 data URL to another Image data URL with capped dimensions to reduce file size.
 * @param base64UrlImage - The Image base64 data URL to convert
 * @param maxMegapixels - The maximum image size in megapixels for the output Image
 * @returns Promise resolving to Image data URL
 */
export function capImageDataURLSize(
	base64UrlImage: string,
	maxMegapixels: number
): Promise<string> {
	return new Promise((resolve, reject) => {
		try {
			
			const mimeMatch = base64UrlImage.match(BASE64_IMAGE_URI_REGEX);
			
			if (!mimeMatch) {
				return reject(new Error('Invalid data URL format.'));
			}

			const mimeType = mimeMatch[1] as MimeTypeImage;

			if (!Object.values(MimeTypeImage).includes(mimeType)) {
				return reject(new Error(`Unsupported image MIME type: ${mimeType}`));
			}

			const img = new Image();

			img.onload = () => {
				try {
					const canvas = document.createElement('canvas');
					const ctx = canvas.getContext('2d');

					if (!ctx) {
						throw new Error('Failed to get 2D canvas context.');
					}

					const targetWidth = img.naturalWidth;
					const targetHeight = img.naturalHeight;
					const totalPixels = targetWidth * targetHeight;
					const maxPixels = Math.floor(maxMegapixels * MEGAPIXELS_TO_PIXELS);

					if (maxPixels > 0 && totalPixels > maxPixels) {
						const scaleFactor = Math.sqrt(maxPixels / totalPixels);
						canvas.width = Math.floor(targetWidth * scaleFactor);
						canvas.height = Math.floor(targetHeight * scaleFactor);
					} else {
						return resolve(base64UrlImage);
					}

					ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
					resolve(canvas.toDataURL(mimeType));
				} catch (err) {
					reject(err instanceof Error ? err : new Error(String(err)));
				}
			};

			img.onerror = () => {
				reject(new Error('Failed to load image.'));
			};

			img.src = base64UrlImage;
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			const errorMessage = `Error resizing image: ${message}`;
			console.error(errorMessage, error);
			reject(new Error(errorMessage));
		}
	});
}
