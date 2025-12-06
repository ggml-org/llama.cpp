# Helper to manage pdf related requests
# by Humans for All

import urllib.parse
import urlvalidator as uv
import filemagic as mFile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from simpleproxy import ProxyHandler


PDFOUTLINE_MAXDEPTH=4


def extract_pdfoutline(ol: Any, prefix: list[int]):
    """
    Helps extract the pdf outline recursively, along with its numbering.
    """
    if (len(prefix) > PDFOUTLINE_MAXDEPTH):
        return ""
    if type(ol).__name__ != type([]).__name__:
        prefix[-1] += 1
        return f"{".".join(map(str,prefix))}:{ol['/Title']}\n"
    olText = ""
    prefix.append(0)
    for (i,iol) in enumerate(ol):
        olText += extract_pdfoutline(iol, prefix)
    prefix.pop()
    return olText


def process_pdftext(url: str, startPN: int, endPN: int):
    """
    Extract textual content from given pdf.

    * Validate the got url.
    * Get the pdf file.
    * Extract textual contents of the pdf from given start page number to end page number (inclusive).
        * if -1 | 0 is specified wrt startPN, the actual starting page number (rather 1) will be used.
        * if -1 | 0 is specified wrt endPN, the actual ending page number will be used.

    NOTE: Page numbers start from 1, while the underlying list data structure index starts from 0
    """
    import pypdf
    import io
    gotVU = uv.validate_url(url, "HandlePdfText")
    if not gotVU.callOk:
        return { 'status': gotVU.statusCode, 'msg': gotVU.statusMsg }
    gotFile = mFile.get_file(url, "ProcessPdfText", "application/pdf", {})
    if not gotFile.callOk:
        return { 'status': gotFile.statusCode, 'msg': gotFile.statusMsg, 'data': gotFile.contentData}
    tPdf = ""
    oPdf = pypdf.PdfReader(io.BytesIO(gotFile.contentData))
    if (startPN <= 0):
        startPN = 1
    if (endPN <= 0) or (endPN > len(oPdf.pages)):
        endPN = len(oPdf.pages)
    # Add the pdf outline, if available
    outlineGot = extract_pdfoutline(oPdf.outline, [])
    if outlineGot:
        tPdf += f"\n\nOutline Start\n\n{outlineGot}\n\nOutline End\n\n"
    # Add the pdf page contents
    for i in range(startPN, endPN+1):
        pd = oPdf.pages[i-1]
        tPdf = tPdf + pd.extract_text()
    return { 'status': 200, 'msg': "PdfText Response follows", 'data': tPdf }


def handle_pdftext(ph: 'ProxyHandler', pr: urllib.parse.ParseResult):
    """
    Handle requests to pdftext path, which is used to extract plain text
    from the specified pdf file.
    """
    queryParams = urllib.parse.parse_qs(pr.query)
    url = queryParams['url'][0]
    startP = queryParams.get('startPageNumber', -1)
    if isinstance(startP, list):
        startP = int(startP[0])
    endP = queryParams.get('endPageNumber', -1)
    if isinstance(endP, list):
        endP = int(endP[0])
    print(f"INFO:HandlePdfText:Processing:{url}:{startP}:{endP}...")
    gotP2T = process_pdftext(url, startP, endP)
    if (gotP2T['status'] != 200):
        ph.send_error(gotP2T['status'], gotP2T['msg'] )
        return
    ph.send_response(gotP2T['status'], gotP2T['msg'])
    ph.send_header('Content-Type', 'text/text')
    # Add CORS for browser fetch, just in case
    ph.send_header('Access-Control-Allow-Origin', '*')
    ph.end_headers()
    print(f"INFO:HandlePdfText:ExtractedText:{url}...")
    ph.wfile.write(gotP2T['data'].encode('utf-8'))
