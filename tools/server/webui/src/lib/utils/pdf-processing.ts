/**
 * PDF processing utilities using PDF.js
 * Handles PDF text extraction and image conversion in the browser
 */

import { browser } from '$app/environment';
import { MimeTypeApplication, MimeTypeImage } from '$lib/enums';
import * as pdfjs from 'pdfjs-dist';

type TextContent = {
	items: Array<{ str: string }>;
};

if (browser) {
	// Import worker as text and create blob URL for inline bundling
	import('pdfjs-dist/build/pdf.worker.min.mjs?raw')
		.then((workerModule) => {
			const workerBlob = new Blob([workerModule.default], { type: 'application/javascript' });
			pdfjs.GlobalWorkerOptions.workerSrc = URL.createObjectURL(workerBlob);
		})
		.catch(() => {
			console.warn('Failed to load PDF.js worker, PDF processing may not work');
		});
}

/**
 * Convert a File object to ArrayBuffer for PDF.js processing
 * @param file - The PDF file to convert
 * @returns Promise resolving to the file's ArrayBuffer
 */
async function getFileAsBuffer(file: File): Promise<ArrayBuffer> {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = (event) => {
			if (event.target?.result) {
				resolve(event.target.result as ArrayBuffer);
			} else {
				reject(new Error('Failed to read file.'));
			}
		};
		reader.onerror = () => {
			reject(new Error('Failed to read file.'));
		};
		reader.readAsArrayBuffer(file);
	});
}

/**
 * Extract text content from a PDF file
 * @param file - The PDF file to process
 * @returns Promise resolving to the extracted text content
 */
export async function convertPDFToText(file: File): Promise<string> {
	if (!browser) {
		throw new Error('PDF processing is only available in the browser');
	}

	try {
		const buffer = await getFileAsBuffer(file);
		const pdf = await pdfjs.getDocument(buffer).promise;
		const numPages = pdf.numPages;

		const textContentPromises: Promise<TextContent>[] = [];

		for (let i = 1; i <= numPages; i++) {
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			textContentPromises.push(pdf.getPage(i).then((page: any) => page.getTextContent()));
		}

		const textContents = await Promise.all(textContentPromises);
		const textItems = textContents.flatMap((textContent: TextContent) =>
			textContent.items.map((item) => item.str ?? '')
		);

		return textItems.join('\n');
	} catch (error) {
		console.error('Error converting PDF to text:', error);
		throw new Error(
			`Failed to convert PDF to text: ${error instanceof Error ? error.message : 'Unknown error'}`
		);
	}
}

/**
 * Convert PDF pages to PNG images as data URLs
 * @param file - The PDF file to convert
 * @param scale - Rendering scale factor (default: 1.5)
 * @returns Promise resolving to array of PNG data URLs
 */
export async function convertPDFToImage(file: File, scale: number = 1.5): Promise<string[]> {
	if (!browser) {
		throw new Error('PDF processing is only available in the browser');
	}

	try {
		const buffer = await getFileAsBuffer(file);
		const doc = await pdfjs.getDocument(buffer).promise;
		const pages: Promise<string>[] = [];

		for (let i = 1; i <= doc.numPages; i++) {
			const page = await doc.getPage(i);
			const viewport = page.getViewport({ scale });
			const canvas = document.createElement('canvas');
			const ctx = canvas.getContext('2d');

			canvas.width = viewport.width;
			canvas.height = viewport.height;

			if (!ctx) {
				throw new Error('Failed to get 2D context from canvas');
			}

			const task = page.render({
				canvasContext: ctx,
				viewport: viewport,
				canvas: canvas
			});
			pages.push(
				task.promise.then(() => {
					return canvas.toDataURL(MimeTypeImage.PNG);
				})
			);
		}

		return await Promise.all(pages);
	} catch (error) {
		console.error('Error converting PDF to images:', error);
		throw new Error(
			`Failed to convert PDF to images: ${error instanceof Error ? error.message : 'Unknown error'}`
		);
	}
}

/**
 * Check if a file is a PDF based on its MIME type
 * @param file - The file to check
 * @returns True if the file is a PDF
 */
export function isPdfFile(file: File): boolean {
	return file.type === MimeTypeApplication.PDF;
}

/**
 * Check if a MIME type represents a PDF
 * @param mimeType - The MIME type to check
 * @returns True if the MIME type is application/pdf
 */
export function isApplicationMimeType(mimeType: string): boolean {
	return mimeType === MimeTypeApplication.PDF;
}
