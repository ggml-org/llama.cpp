// pdf.js loads on first use, with its worker as a file of its own: extracting a PDF's text
// and drawing its pages go through that load.
import { convertPDFToImage, convertPDFToText } from '#lib/utils/pdf-processing.ts';
import { describe, expect, it } from 'vitest';

/** A one-page PDF that shows `text`, with a correct cross-reference table. */
function pdfWithText(text: string): File {
	const stream = `BT /F1 24 Tf 72 720 Td (${text}) Tj ET`;
	const objects = [
		'<< /Type /Catalog /Pages 2 0 R >>',
		'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
		'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>',
		`<< /Length ${stream.length} >>\nstream\n${stream}\nendstream`,
		'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>'
	];

	let body = '%PDF-1.4\n';

	const offsets: number[] = [];

	objects.forEach((object, i) => {
		offsets.push(body.length);
		body += `${i + 1} 0 obj\n${object}\nendobj\n`;
	});

	const xref = body.length;
	const entries = offsets.map((offset) => `${String(offset).padStart(10, '0')} 00000 n \n`);

	body += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n${entries.join('')}`;
	body += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;

	return new File([body], 'hello.pdf', { type: 'application/pdf' });
}

describe('pdf-processing', () => {
	it('extracts the text of a PDF', async () => {
		const text = await convertPDFToText(pdfWithText('Hello from a PDF'));

		expect(text).toContain('Hello from a PDF');
	});

	it('draws each page of a PDF as a PNG', async () => {
		const pages = await convertPDFToImage(pdfWithText('Page one'));

		expect(pages).toHaveLength(1);
		expect(pages[0]).toMatch(/^data:image\/png;base64,/);
	});
});
