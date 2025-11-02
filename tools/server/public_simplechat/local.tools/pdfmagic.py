# Helper to manage pdf related requests
# by Humans for All

import urllib.parse
import urlvalidator as uv
import filemagic as mFile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simpleproxy import ProxyHandler


def process_pdf2text(url: str, startPN: int, endPN: int):
    """
    Extract textual content from given pdf.

    * Validate the got url.
    * Extract textual contents of the pdf from given start page number to end page number (inclusive).
        * if -1 | 0 is specified wrt startPN, the actual starting page number (rather 1) will be used.
        * if -1 | 0 is specified wrt endPN, the actual ending page number will be used.

    NOTE: Page numbers start from 1, while the underlying list data structure index starts from 0
    """
    import pypdf
    import io
    gotVU = uv.validate_url(url, "HandlePdf2Text")
    if not gotVU.callOk:
        return { 'status': gotVU.statusCode, 'msg': gotVU.statusMsg }
    gotFile = mFile.get_file(url, "ProcessPdf2Text", "application/pdf", {})
    if not gotFile.callOk:
        return { 'status': gotFile.statusCode, 'msg': gotFile.statusMsg, 'data': gotFile.contentData}
    tPdf = ""
    oPdf = pypdf.PdfReader(io.BytesIO(gotFile.contentData))
    if (startPN <= 0):
        startPN = 1
    if (endPN <= 0) or (endPN > len(oPdf.pages)):
        endPN = len(oPdf.pages)
    for i in range(startPN, endPN+1):
        pd = oPdf.pages[i-1]
        tPdf = tPdf + pd.extract_text()
    return { 'status': 200, 'msg': "Pdf2Text Response follows", 'data': tPdf }


def handle_pdf2text(ph: 'ProxyHandler', pr: urllib.parse.ParseResult):
    """
    Handle requests to pdf2text path, which is used to extract plain text
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
    print(f"INFO:HandlePdf2Text:Processing:{url}:{startP}:{endP}...")
    gotP2T = process_pdf2text(url, startP, endP)
    if (gotP2T['status'] != 200):
        ph.send_error(gotP2T['status'], gotP2T['msg'] )
        return
    ph.send_response(gotP2T['status'], gotP2T['msg'])
    ph.send_header('Content-Type', 'text/text')
    # Add CORS for browser fetch, just in case
    ph.send_header('Access-Control-Allow-Origin', '*')
    ph.end_headers()
    print(f"INFO:HandlePdf2Text:ExtractedText:{url}...")
    ph.wfile.write(gotP2T['data'].encode('utf-8'))
