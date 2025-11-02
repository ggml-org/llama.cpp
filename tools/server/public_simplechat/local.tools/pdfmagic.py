# Helper to manage pdf related requests
# by Humans for All

import urllib.parse
import urlvalidator as uv
import simpleproxy as root


def process_pdf2text(url: str, startPN: int, endPN: int):
    import pypdf
    import io
    gotVU = uv.validate_url(url, "HandlePdf2Text")
    if not gotVU.callOk:
        return { 'status': gotVU.statusCode, 'msg': gotVU.statusMsg }
    urlParts = urllib.parse.urlparse(url)
    fPdf = open(urlParts.path, 'rb')
    dPdf = fPdf.read()
    tPdf = ""
    oPdf = pypdf.PdfReader(io.BytesIO(dPdf))
    if (startPN < 0):
        startPN = 0
    if (endPN < 0) or (endPN >= len(oPdf.pages)):
        endPN = len(oPdf.pages)-1
    for i in range(startPN, endPN+1):
        pd = oPdf.pages[i]
        tPdf = tPdf + pd.extract_text()
    return { 'status': 200, 'msg': "Pdf2Text Response follows", 'data': tPdf }


def handle_pdf2text(ph: root.ProxyHandler, pr: urllib.parse.ParseResult):
    """
    Handle requests to pdf2text path, which is used to extract plain text
    from the specified pdf file.
    """
    queryParams = urllib.parse.parse_qs(pr.query)
    url = queryParams['url'][0]
    startP = queryParams['startPageNumber'][0]
    if startP:
        startP = int(startP)
    else:
        startP = -1
    endP = queryParams['endPageNumber'][0]
    if endP:
        endP = int(endP)
    else:
        endP = -1
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
