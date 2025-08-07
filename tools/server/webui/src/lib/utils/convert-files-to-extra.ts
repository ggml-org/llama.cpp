import { convertPDFToImage, convertPDFToText, isPdfMimeType } from "./pdf-processing";
import { isSvgMimeType, svgBase64UrlToPngDataURL } from "./svg-to-png";
import { isWebpMimeType, webpBase64UrlToPngDataURL } from "./webp-to-png";
import { config } from '$lib/stores/settings.svelte';
import { isLikelyTextFile, readFileAsText } from "./text-files";

export async function parseFilesToMessageExtras(
    files: ChatUploadedFile[]
): Promise<DatabaseMessageExtra[]> {
    const extras: DatabaseMessageExtra[] = [];

    for (const file of files) {
        if (file.type.startsWith('image/')) {
            if (file.preview) {
                let base64Url = file.preview;

                if (isSvgMimeType(file.type)) {
                    try {
                        base64Url = await svgBase64UrlToPngDataURL(base64Url);
                    } catch (error) {
                        console.error(
                            'Failed to convert SVG to PNG for database storage:',
                            error
                        );
                    }
                } else if (isWebpMimeType(file.type)) {
                    try {
                        base64Url = await webpBase64UrlToPngDataURL(base64Url);
                    } catch (error) {
                        console.error(
                            'Failed to convert WebP to PNG for database storage:',
                            error
                        );
                    }
                }

                extras.push({
                    type: 'imageFile',
                    name: file.name,
                    base64Url
                });
            }
        } else if (isPdfMimeType(file.type)) {
            try {
                const currentConfig = config();
                const shouldProcessAsImages = Boolean(currentConfig.pdfAsImage);
                
                if (shouldProcessAsImages) {
                    // Process PDF as images
                    try {
                        const images = await convertPDFToImage(file.file);
                        extras.push({
                            type: 'pdfFile',
                            name: file.name,
                            content: `PDF file with ${images.length} pages`,
                            images: images,
                            processedAsImages: true
                        });
                    } catch (imageError) {
                        console.warn(`Failed to process PDF ${file.name} as images, falling back to text:`, imageError);
                        // Fallback to text processing
                        const content = await convertPDFToText(file.file);
                        extras.push({
                            type: 'pdfFile',
                            name: file.name,
                            content: content,
                            processedAsImages: false
                        });
                    }
                } else {
                    // Process PDF as text (default)
                    const content = await convertPDFToText(file.file);
                    extras.push({
                        type: 'pdfFile',
                        name: file.name,
                        content: content,
                        processedAsImages: false
                    });
                }
            } catch (error) {
                console.error(`Failed to process PDF file ${file.name}:`, error);
            }
        } else {
            try {
                const content = await readFileAsText(file.file);

                if (isLikelyTextFile(content)) {
                    extras.push({
                        type: 'textFile',
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

    return extras;
}