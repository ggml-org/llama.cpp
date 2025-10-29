import { isSvgMimeType, svgBase64UrlToPngDataURL } from './svg-to-png';
import { isTextFileByName } from './text-files';
import { isWebpMimeType, webpBase64UrlToPngDataURL } from './webp-to-png';
import { FileTypeCategory } from '$lib/enums/files';
import { getFileTypeCategory } from '$lib/utils/file-type';
import { supportsVision } from '$lib/stores/server.svelte';
import { settingsStore } from '$lib/stores/settings.svelte';
import { toast } from 'svelte-sonner';

/**
 * Read a file as a data URL (base64 encoded)
 * @param file - The file to read
 * @returns Promise resolving to the data URL string
 */
function readFileAsDataURL(file: File): Promise<string> {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = () => resolve(reader.result as string);
		reader.onerror = () => reject(reader.error);
		reader.readAsDataURL(file);
	});
}

/**
 * Read a file as UTF-8 text
 * @param file - The file to read
 * @returns Promise resolving to the text content
 */
function readFileAsUTF8(file: File): Promise<string> {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = () => resolve(reader.result as string);
		reader.onerror = () => reject(reader.error);
		reader.readAsText(file);
	});
}

/**
 * Process uploaded files into ChatUploadedFile format with previews and content
 *
 * This function processes various file types and generates appropriate previews:
 * - Images: Base64 data URLs with format normalization (SVG/WebP → PNG)
 * - Text files: UTF-8 content extraction
 * - PDFs: Metadata only (processed later in conversion pipeline)
 * - Audio: Base64 data URLs for preview
 *
 * @param files - Array of File objects to process
 * @returns Promise resolving to array of ChatUploadedFile objects
 */
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
			if (getFileTypeCategory(file.type) === FileTypeCategory.IMAGE) {
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
			} else if (
				getFileTypeCategory(file.type) === FileTypeCategory.TEXT ||
				isTextFileByName(file.name)
			) {
				try {
					const textContent = await readFileAsUTF8(file);
					results.push({ ...base, textContent });
				} catch (err) {
					console.warn('Failed to read text file, adding without content:', err);
					results.push(base);
				}
			} else if (getFileTypeCategory(file.type) === FileTypeCategory.PDF) {
				// PDFs handled later when building extras; keep metadata only
				results.push(base);

				// Show suggestion toast if vision model is available but PDF as image is disabled
				const hasVisionSupport = supportsVision();
				const currentConfig = settingsStore.config;
				if (hasVisionSupport && !currentConfig.pdfAsImage) {
					toast.info(`You can enable parsing PDF as images with vision models.`, {
						duration: 8000,
						action: {
							label: 'Enable PDF as Images',
							onClick: () => {
								settingsStore.updateConfig('pdfAsImage', true);
								toast.success('PDF parsing as images enabled!', {
									duration: 3000
								});
							}
						}
					});
				}
			} else if (getFileTypeCategory(file.type) === FileTypeCategory.AUDIO) {
				// Generate preview URL for audio files
				const preview = await readFileAsDataURL(file);
				results.push({ ...base, preview });
			} else {
				// Other files: add as-is
				results.push(base);
			}
		} catch (error) {
			console.error('Error processing file', file.name, error);
			results.push(base);
		}
	}

	return results;
}
