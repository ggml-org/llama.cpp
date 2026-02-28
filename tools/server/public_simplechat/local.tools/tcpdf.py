# Helper to manage pdf related requests
# by Humans for All

import urlvalidator as uv
import filemagic as mFile
import toolcalls as mTC
from typing import Any


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
    gotVU = uv.validate_url(url, "ProcessPdfText")
    if not gotVU.callOk:
        return mTC.TCOutResponse(False, gotVU.statusCode, gotVU.statusMsg)
    gotFile = mFile.get_file(url, "ProcessPdfText", "application/pdf", {})
    if not gotFile.callOk:
        return gotFile
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
    return mTC.TCOutResponse(True, 200, "PdfText Response follows", "text/text", tPdf.encode('utf-8'))


class TCPdfText(mTC.ToolCall):

    def tcf_meta(self) -> mTC.TCFunction:
        return mTC.TCFunction(
            self.name,
            "Fetch pdf from requested local file path / web url through a proxy server and return its text content after converting pdf to text, in few seconds. One is allowed to get a part of the pdf by specifying the starting and ending page numbers",
            mTC.TCInParameters(
                "object",
                {
                    "url": mTC.TCInProperty(
                        "string",
                        "local file path (file://) / web (http/https) based url of the pdf that will be got and inturn converted to text"
                    ),
                    "startPageNumber": mTC.TCInProperty(
                        "integer",
                        "Specify the starting page number within the pdf, this is optional. If not specified set to first page."
                    ),
                    "endPageNumber": mTC.TCInProperty(
                        "integer",
                        "Specify the ending page number within the pdf, this is optional. If not specified set to the last page."
                    )
                },
                [ "url" ]
            )
        )

    def tc_handle(self, args: mTC.TCInArgs, inHeaders: mTC.HttpHeaders) -> mTC.TCOutResponse:
        """
        Handle pdftext request,
        which is used to extract plain text from the specified pdf file.
        """
        try:
            url = args['url']
            startP = int(args.get('startPageNumber', -1))
            endP = int(args.get('endPageNumber', -1))
            print(f"INFO:HandlePdfText:Processing:{url}:{startP}:{endP}...")
            return process_pdftext(url, startP, endP)
        except Exception as exc:
            return mTC.TCOutResponse(False, 502, f"WARN:HandlePdfText:Failed:{exc}")


def ok():
    import importlib
    dep = "pypdf"
    try:
        importlib.import_module(dep)
        return True
    except ImportError as exc:
        print(f"WARN:TCPdf:{dep} missing or has issues, so not enabling myself")
        return False
