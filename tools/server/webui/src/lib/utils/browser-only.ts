/**
 * Browser-only utility exports
 *
 * These utilities require browser APIs (DOM, Canvas, MediaRecorder, etc.)
 * and cannot be imported during SSR. Import from '$lib/utils/browser-only'
 * only in client-side code or components that are not server-rendered.
 */

// Audio utilities (MediaRecorder API)
export {
	AudioRecorder,
	convertToWav,
	createAudioFile,
	isAudioRecordingSupported
} from './audio-recording';

// PDF processing utilities (pdfjs-dist with DOMMatrix)
export {
	convertPDFToText,
	convertPDFToImage,
	isPdfFile as isPdfFileFromFile,
	isApplicationMimeType
} from './pdf-processing';

// File conversion utilities (depends on pdf-processing)
export { parseFilesToMessageExtras, type FileProcessingResult } from './convert-files-to-extra';

// File upload processing utilities (depends on pdf-processing, svg-to-png, webp-to-png)
export { processFilesToChatUploaded } from './process-uploaded-files';

// SVG utilities (Canvas/Image API)
export { svgBase64UrlToPngDataURL, isSvgFile, isSvgMimeType } from './svg-to-png';

// WebP utilities (Canvas/Image API)
export { webpBase64UrlToPngDataURL, isWebpFile, isWebpMimeType } from './webp-to-png';
