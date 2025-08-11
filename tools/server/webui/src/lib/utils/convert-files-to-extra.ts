import { convertPDFToImage, convertPDFToText, isPdfMimeType } from "./pdf-processing";
import { isSvgMimeType, svgBase64UrlToPngDataURL } from "./svg-to-png";
import { isWebpMimeType, webpBase64UrlToPngDataURL } from "./webp-to-png";
import { config } from '$lib/stores/settings.svelte';
import { isLikelyTextFile, readFileAsText } from "./text-files";

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

function isAudioMimeType(mimeType: string): boolean {
    return mimeType === 'audio/mpeg' || 
           mimeType === 'audio/wav' || 
           mimeType === 'audio/mp3' || 
           mimeType === 'audio/webm' ||
           mimeType === 'audio/ogg' ||
           mimeType === 'audio/m4a';
}

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
        } else if (isAudioMimeType(file.type)) {
            // Process audio files (MP3 and WAV)
            try {
                const base64Data = await readFileAsBase64(file.file);

                extras.push({
                    type: 'audioFile',
                    name: file.name,
                    base64Data: base64Data,
                    mimeType: file.type
                });
            } catch (error) {
                console.error(`Failed to process audio file ${file.name}:`, error);
            }
        } else if (isPdfMimeType(file.type)) {
            try {
                // Always get base64 data for preview functionality
                const base64Data = await readFileAsBase64(file.file);
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
                            processedAsImages: true,
                            base64Data: base64Data
                        });
                    } catch (imageError) {
                        console.warn(`Failed to process PDF ${file.name} as images, falling back to text:`, imageError);

                        // Fallback to text processing
                        const content = await convertPDFToText(file.file);

                        extras.push({
                            type: 'pdfFile',
                            name: file.name,
                            content: content,
                            processedAsImages: false,
                            base64Data: base64Data
                        });
                    }
                } else {
                    // Process PDF as text (default)
                    const content = await convertPDFToText(file.file);

                    extras.push({
                        type: 'pdfFile',
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