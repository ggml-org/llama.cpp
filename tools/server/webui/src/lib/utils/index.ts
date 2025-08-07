/**
 * Utility functions for the webui application
 * Exports commonly used helper functions for file processing, UI interactions, and data manipulation
 */

import autoResizeTextarea from './autoresize-textarea';
import { parseFilesToMessageExtras } from './convert-files-to-extra';
import { copyCodeToClipboard, copyToClipboard } from './copy';
import { convertPDFToText, convertPDFToImage, isPdfMimeType } from './pdf-processing';
import { isLikelyTextFile, isTextFileByName, readFileAsText } from './text-files';
import { extractPartialThinking, hasThinkingEnd, hasThinkingStart, parseThinkingContent } from './thinking';
import { processFilesToChatUploaded } from './process-uploaded-files';
import { isSvgFile, isSvgMimeType, svgBase64UrlToPngDataURL } from './svg-to-png';
import { isWebpFile, isWebpMimeType, webpBase64UrlToPngDataURL } from './webp-to-png';

export { 
    autoResizeTextarea,
    copyCodeToClipboard,
    copyToClipboard,
    parseFilesToMessageExtras,
    convertPDFToText,
    convertPDFToImage,
    extractPartialThinking,
    hasThinkingEnd,
    isLikelyTextFile,
    isPdfMimeType,
    isSvgFile,
    isSvgMimeType,
    isTextFileByName,
    isWebpFile,
    isWebpMimeType,
    hasThinkingStart,
    parseThinkingContent,
    processFilesToChatUploaded,
    readFileAsText,
    svgBase64UrlToPngDataURL,
    webpBase64UrlToPngDataURL,
};
