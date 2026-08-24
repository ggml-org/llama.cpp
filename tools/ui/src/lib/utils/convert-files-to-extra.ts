import { convertPDFToImage, convertPDFToText } from './pdf-processing';
import { isSvgMimeType, svgBase64UrlToPngDataURL } from './svg-to-png';
import { isLikelyTextFile, readFileAsText } from './text-files';
import { isWebpMimeType, webpBase64UrlToPngDataURL } from './webp-to-png';
import { AttachmentType, FileTypeCategory, SpecialFileType } from '$lib/enums';
import { modelsStore } from '$lib/stores/models/index.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import type { ChatUploadedFile, DatabaseMessageExtra, FileProcessingResult } from '$lib/types';
import { getFileTypeCategory, getPdfParseMode } from '$lib/utils';
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
				const parseMode = getPdfParseMode(settingsStore.config);
				const hasVisionSupport = activeModelId
					? modelsStore.props.modelSupportsVision(activeModelId)
					: false;

				if (parseMode === 'image' && !hasVisionSupport) {
					toast.warning(
						`PDF "${file.name}" kept as original file: Parse as image needs a vision model.`,
						{ duration: 5000 }
					);
				}

				if (parseMode === 'image' && hasVisionSupport) {
					try {
						const images = await convertPDFToImage(file.file);

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
							parsedAs: 'image',
							processedAsImages: true,
							size: file.size,
							type: AttachmentType.PDF
						});
					} catch (imageError) {
						console.warn(
							`Failed to process PDF ${file.name} as images, keeping original file:`,
							imageError
						);

						toast.warning(`Could not render PDF "${file.name}" as images; keeping original file.`, {
							duration: 4000
						});

						extras.push({
							base64Data: base64Data,
							content: '',
							name: file.name,
							parsedAs: 'none',
							processedAsImages: false,
							size: file.size,
							type: AttachmentType.PDF
						});
					}
				} else if (parseMode === 'text') {
					const content = await convertPDFToText(file.file);

					toast.success(`PDF "${file.name}" processed as text content.`, {
						duration: 3000
					});

					extras.push({
						base64Data: base64Data,
						content: content,
						name: file.name,
						parsedAs: 'text',
						processedAsImages: false,
						size: file.size,
						type: AttachmentType.PDF
					});
				} else {
					extras.push({
						base64Data: base64Data,
						content: '',
						name: file.name,
						parsedAs: 'none',
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
