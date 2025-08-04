import { browser } from '$app/environment';

// Types for PDF.js (imported conditionally)
type TextContent = {
	items: Array<{ str: string }>;
};

type TextItem = {
	str: string;
};

// PDF.js instance (loaded dynamically)
let pdfjs: any = null;

// Initialize PDF.js only on the client side
async function initializePdfJs() {
	if (!browser || pdfjs) return;
	
	try {
		// Dynamic import to prevent SSR issues
		pdfjs = await import('pdfjs-dist');
		
		// Set up PDF.js worker
		pdfjs.GlobalWorkerOptions.workerSrc = new URL(
			'pdfjs-dist/build/pdf.worker.min.mjs',
			import.meta.url
		).toString();
	} catch (error) {
		console.error('Failed to initialize PDF.js:', error);
		throw new Error('PDF.js is not available');
	}
}

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


export async function convertPDFToText(file: File): Promise<string> {
	if (!browser) {
		throw new Error('PDF processing is only available in the browser');
	}
	
	try {
		await initializePdfJs();
		
		const buffer = await getFileAsBuffer(file);
		const pdf = await pdfjs.getDocument(buffer).promise;
		const numPages = pdf.numPages;
		
		const textContentPromises: Promise<TextContent>[] = [];
		for (let i = 1; i <= numPages; i++) {
			textContentPromises.push(
				pdf.getPage(i).then((page: any) => page.getTextContent())
			);
		}
		
		const textContents = await Promise.all(textContentPromises);
		const textItems = textContents.flatMap((textContent: TextContent) =>
			textContent.items.map((item) => item.str ?? '')
		);
		
		return textItems.join('\n');
	} catch (error) {
		console.error('Error converting PDF to text:', error);
		throw new Error(`Failed to convert PDF to text: ${error instanceof Error ? error.message : 'Unknown error'}`);
	}
}

export async function convertPDFToImage(file: File, scale: number = 1.5): Promise<string[]> {
	if (!browser) {
		throw new Error('PDF processing is only available in the browser');
	}
	
	try {
		await initializePdfJs();
		
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
					return canvas.toDataURL('image/png');
				})
			);
		}

		return await Promise.all(pages);
	} catch (error) {
		console.error('Error converting PDF to images:', error);
		throw new Error(`Failed to convert PDF to images: ${error instanceof Error ? error.message : 'Unknown error'}`);
	}
}

export function isPdfFile(file: File): boolean {
	return file.type === 'application/pdf';
}

export function isPdfMimeType(mimeType: string): boolean {
	return mimeType === 'application/pdf';
}
