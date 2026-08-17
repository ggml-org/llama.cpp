import { heicFileToJpegDataURL, isHeicMimeType } from './heic-to-jpeg';
import { isJxlFile, normalizeJxlDataUrl } from './jxl';
import { convertPDFToText } from './pdf-processing';
import { isSvgMimeType, svgBase64UrlToPngDataURL } from './svg-to-png';
import { isWebpMimeType, webpBase64UrlToPngDataURL } from './webp-to-png';
import { SETTINGS_KEYS } from '$lib/constants';
import { FileTypeCategory, MimeTypeImage } from '$lib/enums';
import { modelsStore } from '$lib/stores/models.svelte';
import { settingsStore } from '$lib/stores/settings.svelte';
import { getUploadedFileCategory } from '$lib/utils';
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
 * - JPEG XL: kept as image/jxl data URLs (not converted)
 * - Text files: UTF-8 content extraction
 * - PDFs: Metadata only (processed later in conversion pipeline)
 * - Audio: Base64 data URLs for preview
 *
 * @param files - Array of File objects to process
 * @returns Promise resolving to array of ChatUploadedFile objects
 */
export async function processFilesToChatUploaded(
	files: File[],
	activeModelId?: string
): Promise<ChatUploadedFile[]> {
	const results: ChatUploadedFile[] = [];

	for (const file of files) {
		const id = Date.now().toString() + Math.random().toString(36).substr(2, 9);
		const base: ChatUploadedFile = {
			file,
			id,
			name: file.name,
			size: file.size,
			type: file.type
		};

		try {
			if (getUploadedFileCategory(file) === FileTypeCategory.IMAGE || isJxlFile(file)) {
				let preview = await readFileAsDataURL(file);

				// browsers often report an empty type for .jxl, so set it here
				const imageType = isJxlFile(file) ? MimeTypeImage.JXL : file.type;

				if (isJxlFile(file)) {
					preview = normalizeJxlDataUrl(preview, file.name);
				} else if (isSvgMimeType(file.type)) {
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
				} else if (isHeicMimeType(file.type)) {
					try {
						preview = await heicFileToJpegDataURL(file);
					} catch (err) {
						console.error('Failed to convert HEIC to PNG:', err);

						continue;
					}
				}

				results.push({ ...base, preview, type: imageType });
			} else if (getUploadedFileCategory(file) === FileTypeCategory.PDF) {
				// Extract text content from PDF for preview
				try {
					const textContent = await convertPDFToText(file);

					results.push({ ...base, textContent });
				} catch (err) {
					console.warn('Failed to extract text from PDF, adding without content:', err);
					results.push(base);
				}

				// Show suggestion toast if vision model is available but PDF as image is disabled
				const hasVisionSupport = activeModelId
					? modelsStore.modelSupportsVision(activeModelId)
					: false;
				const currentConfig = settingsStore.config;

				if (hasVisionSupport && !currentConfig.pdfAsImage) {
					toast.info(`You can enable parsing PDF as images with vision models.`, {
						action: {
							label: 'Enable PDF as Images',
							onClick: () => {
								settingsStore.updateConfig(SETTINGS_KEYS.PDF_AS_IMAGE, true);
								toast.success('PDF parsing as images enabled!', {
									duration: 3000
								});
							}
						},
						duration: 8000
					});
				}
			} else if (getUploadedFileCategory(file) === FileTypeCategory.AUDIO) {
				// Generate preview URL for audio files
				const preview = await readFileAsDataURL(file);

				results.push({ ...base, preview });
			} else if (getUploadedFileCategory(file) === FileTypeCategory.VIDEO) {
				// Generate preview URL for video files
				const preview = await readFileAsDataURL(file);

				results.push({ ...base, preview });
			} else {
				// Fallback: treat unknown files as text
				try {
					const textContent = await readFileAsUTF8(file);

					results.push({ ...base, textContent });
				} catch (err) {
					console.warn('Failed to read file as text, adding without content:', err);
					results.push(base);
				}
			}
		} catch (error) {
			console.error('Error processing file', file.name, error);
			results.push(base);
		}
	}

	return results;
}
