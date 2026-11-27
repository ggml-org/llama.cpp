/**
 * PDF processing utilities using PDF.js
 * Handles PDF text extraction and image conversion in the browser
 */

import { PDF_RENDERING } from '#lib/constants/index.ts';
import { MimeTypeApplication, MimeTypeImage } from '#lib/enums/index.ts';
import { clampPdfDpi, pdfDpiToScale } from '#lib/utils/pdf-dpi.ts';
import { browser } from '$app/env';
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

type TextContent = {
	items: Array<{ str: string }>;
};

let pdfjsModule: Promise<typeof import('pdfjs-dist')> | undefined;

// pdf.js and its worker are files of their own, so the app loads them only when it opens a PDF
function loadPdfjs(): Promise<typeof import('pdfjs-dist')> {
	pdfjsModule ??= import('pdfjs-dist').then((pdfjs) => {
		pdfjs.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

		return pdfjs;
	});

	return pdfjsModule;
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
		const [pdfjs, buffer] = await Promise.all([loadPdfjs(), getFileAsBuffer(file)]);
		const pdf = await pdfjs.getDocument({ data: buffer }).promise;
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
			`Failed to convert PDF to text: ${error instanceof Error ? error.message : 'Unknown error'}`,
			{ cause: error }
		);
	}
}

/**
 * Convert PDF pages to PNG images as data URLs
 * @param file - The PDF file to convert
 * @param dpi - Rendering DPI (default: 108, the scale the app shipped with)
 * @returns Promise resolving to array of PNG data URLs
 */
export async function convertPDFToImage(
	file: File,
	dpi: number = PDF_RENDERING.DEFAULT_DPI
): Promise<string[]> {
	if (!browser) {
		throw new Error('PDF processing is only available in the browser');
	}

	try {
		const [pdfjs, buffer] = await Promise.all([loadPdfjs(), getFileAsBuffer(file)]);
		const doc = await pdfjs.getDocument({ data: buffer }).promise;
		const pages: Promise<string>[] = [];
		const scale = pdfDpiToScale(clampPdfDpi(dpi));

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
				canvas: canvas,
				canvasContext: ctx,
				viewport: viewport
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
			`Failed to convert PDF to images: ${error instanceof Error ? error.message : 'Unknown error'}`,
			{ cause: error }
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
