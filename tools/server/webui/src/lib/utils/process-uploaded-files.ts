import { isPdfMimeType } from './pdf-processing';
import { isSvgMimeType, svgBase64UrlToPngDataURL } from './svg-to-png';
import { isTextFileByName } from './text-files';
import { isWebpMimeType, webpBase64UrlToPngDataURL } from './webp-to-png';

function readFileAsDataURL(file: File): Promise<string> {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = () => resolve(reader.result as string);
		reader.onerror = () => reject(reader.error);
		reader.readAsDataURL(file);
	});
}

function readFileAsUTF8(file: File): Promise<string> {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = () => resolve(reader.result as string);
		reader.onerror = () => reject(reader.error);
		reader.readAsText(file);
	});
}

export async function processFilesToChatUploaded(files: File[]): Promise<ChatUploadedFile[]> {
	const results: ChatUploadedFile[] = [];

	for (const file of files) {
		const id = Date.now().toString() + Math.random().toString(36).substr(2, 9);
		const base: ChatUploadedFile = {
			id,
			name: file.name,
			size: file.size,
			type: file.type,
			file
		};

		try {
			if (file.type.startsWith('image/')) {
				let preview = await readFileAsDataURL(file);

				// Normalize SVG and WebP to PNG in previews
				if (isSvgMimeType(file.type)) {
					try {
						preview = await svgBase64UrlToPngDataURL(preview);
					} catch (err) {
						console.error('Failed to convert SVG to PNG:', err);
					}
				} else if (isWebpMimeType(file.type)) {
					try {
						preview = await webpBase64UrlToPngDataURL(preview);
					} catch (err) {
						console.error('Failed to convert WebP to PNG:', err);
					}
				}

				results.push({ ...base, preview });
			} else if (file.type.startsWith('text/') || isTextFileByName(file.name)) {
				try {
					const textContent = await readFileAsUTF8(file);
					results.push({ ...base, textContent });
				} catch (err) {
					console.warn('Failed to read text file, adding without content:', err);
					results.push(base);
				}
			} else if (isPdfMimeType(file.type)) {
				// PDFs handled later when building extras; keep metadata only
				results.push(base);
			} else {
				// Other files: add as-is (audio, etc.)
				results.push(base);
			}
		} catch (error) {
			console.error('Error processing file', file.name, error);
			results.push(base);
		}
	}

	return results;
}
