import { convertPDFToImage, convertPDFToText } from './pdf-processing';
import { isSvgMimeType, svgBase64UrlToPngDataURL } from './svg-to-png';
import { isLikelyTextFile, readFileAsText } from './text-files';
import { isWebpMimeType, webpBase64UrlToPngDataURL } from './webp-to-png';
import { SETTINGS_KEYS } from '$lib/constants';
import { AttachmentType, FileTypeCategory, SpecialFileType } from '$lib/enums';
import { modelsStore } from '$lib/stores/models/index.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import type { ChatUploadedFile, DatabaseMessageExtra, FileProcessingResult } from '$lib/types';
import { getFileTypeCategory } from '$lib/utils';
import { toast } from 'svelte-sonner';

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
		if (file.type === SpecialFileType.MCP_PROMPT && file.mcpPrompt) {
			extras.push({
				arguments: file.mcpPrompt.arguments,
				content: file.textContent ?? '',
				name: file.name,
				promptName: file.mcpPrompt.promptName,
				serverName: file.mcpPrompt.serverName,
				size: file.size,
				type: AttachmentType.MCP_PROMPT
			});

			continue;
		}

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
					base64Url,
					name: file.name,
					size: file.size,
					type: AttachmentType.IMAGE
				});
			}
		} else if (getFileTypeCategory(file.type) === FileTypeCategory.AUDIO) {
			// Process audio files (MP3 and WAV)
			try {
				const base64Data = await readFileAsBase64(file.file);

				extras.push({
					base64Data: base64Data,
					mimeType: file.type,
					name: file.name,
					size: file.size,
					type: AttachmentType.AUDIO
				});
			} catch (error) {
				console.error(`Failed to process audio file ${file.name}:`, error);
			}
		} else if (getFileTypeCategory(file.type) === FileTypeCategory.VIDEO) {
			// Process video files (MP4, etc)
			try {
				const base64Data = await readFileAsBase64(file.file);

				extras.push({
					base64Data: base64Data,
					mimeType: file.type,
					name: file.name,
					size: file.size,
					type: AttachmentType.VIDEO
				});
			} catch (error) {
				console.error(`Failed to process video file ${file.name}:`, error);
			}
		} else if (getFileTypeCategory(file.type) === FileTypeCategory.PDF) {
			try {
				// Always get base64 data for preview functionality
				const base64Data = await readFileAsBase64(file.file);
				const currentConfig = settingsStore.config;
				// Use per-model vision check for router mode
				const hasVisionSupport = activeModelId
					? modelsStore.props.modelSupportsVision(activeModelId)
					: false;

				// Force PDF-to-text for non-vision models
				let shouldProcessAsImages = Boolean(currentConfig.pdfAsImage) && hasVisionSupport;

				// If user had pdfAsImage enabled but model doesn't support vision, update setting and notify
				if (currentConfig.pdfAsImage && !hasVisionSupport) {
					console.log('Non-vision model detected: forcing PDF-to-text mode and updating settings');

					// Update the setting in localStorage
					settingsStore.updateConfig(SETTINGS_KEYS.PDF_AS_IMAGE, false);

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
							base64Data: base64Data,
							content: `PDF file with ${images.length} pages`,
							images: images,
							name: file.name,
							processedAsImages: true,
							size: file.size,
							type: AttachmentType.PDF
						});
					} catch (imageError) {
						console.warn(
							`Failed to process PDF ${file.name} as images, falling back to text:`,
							imageError
						);

						// Fallback to text processing
						const content = await convertPDFToText(file.file);

						extras.push({
							base64Data: base64Data,
							content: content,
							name: file.name,
							processedAsImages: false,
							size: file.size,
							type: AttachmentType.PDF
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
						base64Data: base64Data,
						content: content,
						name: file.name,
						processedAsImages: false,
						size: file.size,
						type: AttachmentType.PDF
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
						content: content,
						name: file.name,
						size: file.size,
						type: AttachmentType.TEXT
					});
				} else {
					console.warn(`File ${file.name} appears to be binary and will be skipped`);
				}
			} catch (error) {
				console.error(`Failed to read file ${file.name}:`, error);
			}
		}
	}

	return { emptyFiles, extras };
}
