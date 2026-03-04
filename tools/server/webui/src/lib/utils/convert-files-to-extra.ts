import { convertPDFToImage, convertPDFToText } from './pdf-processing';
import { isSvgMimeType, svgBase64UrlToPngDataURL } from './svg-to-png';
import { isWebpMimeType, webpBase64UrlToPngDataURL } from './webp-to-png';
import { FileTypeCategory, AttachmentType } from '$lib/enums';
import { config, settingsStore } from '$lib/stores/settings.svelte';
import { modelsStore } from '$lib/stores/models.svelte';
import { getFileTypeCategory } from '$lib/utils';
import { readFileAsText, isLikelyTextFile } from './text-files';
import { toast } from 'svelte-sonner';
import type { FileProcessingResult, ChatUploadedFile, DatabaseMessageExtra } from '$lib/types';

function readFileAsBase64(file: File): Promise<string> {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();

		reader.onload = () => {
			// Extract base64 data without the data URL prefix
			const dataUrl = reader.result as string;
			const base64 = dataUrl.split(',')[1];
			resolve(base64);
		};

		reader.onerror = () => reject(reader.error);

		reader.readAsDataURL(file);
	});
}

export async function parseFilesToMessageExtras(
	files: ChatUploadedFile[],
	activeModelId?: string
): Promise<FileProcessingResult> {
	const extras: DatabaseMessageExtra[] = [];
	const emptyFiles: string[] = [];

	for (const file of files) {
		if (getFileTypeCategory(file.type) === FileTypeCategory.IMAGE) {
			if (file.preview) {
				let base64Url = file.preview;

				if (isSvgMimeType(file.type)) {
					try {
						base64Url = await svgBase64UrlToPngDataURL(base64Url);
					} catch (error) {
						console.error('Failed to convert SVG to PNG for database storage:', error);
					}
				} else if (isWebpMimeType(file.type)) {
					try {
						base64Url = await webpBase64UrlToPngDataURL(base64Url);
					} catch (error) {
						console.error('Failed to convert WebP to PNG for database storage:', error);
					}
				}

				extras.push({
					type: AttachmentType.IMAGE,
					name: file.name,
					base64Url
				});
			}
		} else if (getFileTypeCategory(file.type) === FileTypeCategory.AUDIO) {
			// Process audio files (MP3 and WAV)
			try {
				const base64Data = await readFileAsBase64(file.file);

				extras.push({
					type: AttachmentType.AUDIO,
					name: file.name,
					base64Data: base64Data,
					mimeType: file.type
				});
			} catch (error) {
				console.error(`Failed to process audio file ${file.name}:`, error);
			}
		} else if (getFileTypeCategory(file.type) === FileTypeCategory.PDF) {
			try {
				// Always get base64 data for preview functionality
				const base64Data = await readFileAsBase64(file.file);
				const currentConfig = config();
				// Use per-model vision check for router mode
				const hasVisionSupport = activeModelId
					? modelsStore.modelSupportsVision(activeModelId)
					: false;

				// Force PDF-to-text for non-vision models
				let shouldProcessAsImages = Boolean(currentConfig.pdfAsImage) && hasVisionSupport;

				// If user had pdfAsImage enabled but model doesn't support vision, update setting and notify
				if (currentConfig.pdfAsImage && !hasVisionSupport) {
					console.log('Non-vision model detected: forcing PDF-to-text mode and updating settings');

					// Update the setting in localStorage
					settingsStore.updateConfig('pdfAsImage', false);

					// Show toast notification to user
					toast.warning(
						'PDF setting changed: Non-vision model detected, PDFs will be processed as text instead of images.',
						{
							duration: 5000
						}
					);

					shouldProcessAsImages = false;
				}

				if (shouldProcessAsImages) {
					// Process PDF as images (only for vision models)
					try {
						const images = await convertPDFToImage(file.file);

						// Show success toast for PDF image processing
						toast.success(
							`PDF "${file.name}" processed as ${images.length} images for vision model.`,
							{
								duration: 3000
							}
						);

						extras.push({
							type: AttachmentType.PDF,
							name: file.name,
							content: `PDF file with ${images.length} pages`,
							images: images,
							processedAsImages: true,
							base64Data: base64Data
						});
					} catch (imageError) {
						console.warn(
							`Failed to process PDF ${file.name} as images, falling back to text:`,
							imageError
						);

						// Fallback to text processing
						const content = await convertPDFToText(file.file);

						extras.push({
							type: AttachmentType.PDF,
							name: file.name,
							content: content,
							processedAsImages: false,
							base64Data: base64Data
						});
					}
				} else {
					// Process PDF as text (default or forced for non-vision models)
					const content = await convertPDFToText(file.file);

					// Show success toast for PDF text processing
					toast.success(`PDF "${file.name}" processed as text content.`, {
						duration: 3000
					});

					extras.push({
						type: AttachmentType.PDF,
						name: file.name,
						content: content,
						processedAsImages: false,
						base64Data: base64Data
					});
				}
			} catch (error) {
				console.error(`Failed to process PDF file ${file.name}:`, error);
			}
		} else {
			try {
				const content = await readFileAsText(file.file);

				// Check if file is empty
				if (content.trim() === '') {
					console.warn(`File ${file.name} is empty and will be skipped`);
					emptyFiles.push(file.name);
				} else if (isLikelyTextFile(content)) {
					extras.push({
						type: AttachmentType.TEXT,
						name: file.name,
						content: content
					});
				} else {
					console.warn(`File ${file.name} appears to be binary and will be skipped`);
				}
			} catch (error) {
				console.error(`Failed to read file ${file.name}:`, error);
			}
		}
	}

	return { extras, emptyFiles };
}
