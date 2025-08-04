export function svgBase64UrlToPngDataURL(
	base64UrlSvg: string,
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
				
				resolve(canvas.toDataURL('image/png'));
			};

			img.onerror = () => {
				reject(
					new Error('Failed to load SVG image. Ensure the SVG data is valid.')
				);
			};

			img.src = base64UrlSvg;
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			const errorMessage = `Error converting SVG to PNG: ${message}`;
			console.error(errorMessage, error);
			reject(new Error(errorMessage));
		}
	});
}

export function isSvgFile(file: File): boolean {
	return file.type === 'image/svg+xml';
}

export function isSvgMimeType(mimeType: string): boolean {
	return mimeType === 'image/svg+xml';
}
