# A simple mcp server with a bunch of bundled tool calls
# by Humans for All
#
# Listens on the specified port (defaults to squids 3128)
# * return the supported tool calls meta data when requested
# * execute the requested tool call and return the results
# * any request to aum path is used to respond with a predefined text response
#   which can help identify this server, in a simple way.
#
# Expects a Bearer authorization line in the http header of the requests got.
#


import sys
import http.server
import urllib.parse
import time
import ssl
import traceback
import json
from typing import Any
from dataclasses import asdict
import tcpdf as mTCPdf
import tcweb as mTCWeb
import toolcalls as mTC
import config as mConfig



gMe = mConfig.Config()


def bearer_transform():
    """
    Transform the raw bearer token to the network handshaked token,
    if and when needed.
    """
    global gMe
    year = str(time.gmtime().tm_year)
    if gMe.op.bearerTransformedYear == year:
        return
    import hashlib
    s256 = hashlib.sha256(year.encode('utf-8'))
    s256.update(gMe.sec.bearerAuth.encode('utf-8'))
    gMe.op.bearerTransformed = s256.hexdigest()
    gMe.op.bearerTransformedYear = year


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    """
    Implements the logic for handling requests sent to this server.
    """

    def send_headers_common(self):
        """
        Common headers to include in responses from this server
        """
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        """
        Overrides the SendError helper
        so that the common headers mentioned above can get added to them
        else CORS failure will be triggered by the browser on fetch from browser.
        """
        if not message:
            message = ""
        print(f"WARN:PH:SendError:{code}:{message}")
        self.send_response(code, message)
        self.send_headers_common()

    def auth_check(self):
        """
        Simple Bearer authorization
        ALERT: For multiple reasons, this is a very insecure implementation.
        """
        bearer_transform()
        authline = self.headers['Authorization']
        if authline == None:
            return mTC.TCOutResponse(False, 400, "WARN:No auth line")
        authlineA = authline.strip().split(' ')
        if len(authlineA) != 2:
            return mTC.TCOutResponse(False, 400, "WARN:Invalid auth line")
        if authlineA[0] != 'Bearer':
            return mTC.TCOutResponse(False, 400, "WARN:Invalid auth type")
        if authlineA[1] != gMe.op.bearerTransformed:
            return mTC.TCOutResponse(False, 400, "WARN:Invalid auth")
        return mTC.TCOutResponse(True, 200, "Auth Ok")

    def send_mcp(self, statusCode: int, statusMessage: str, body: Any):
        self.send_response(statusCode, statusMessage)
        self.send_header('Content-Type', "application/json")
        # Add CORS for browser fetch, just in case
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        data = asdict(body)
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def mcp_toolscall(self, oRPC: Any):
        """
        If authorisation is ok for the request, run the specified handler.
        """
        try:
            if not gMe.op.toolManager:
                raise RuntimeError("DBUG:PH:MCPToolsCall:ToolManager uninitialised")
            inHeaders: Any = self.headers
            resp = gMe.op.toolManager.tc_handle(oRPC["id"], oRPC["params"]["name"], oRPC["params"]["arguments"], inHeaders)
            if not resp.response.callOk:
                self.send_error(resp.response.statusCode, resp.response.statusMsg)
                return
            tcresp = mTC.MCPToolCallResponse(
                resp.tcid,
                resp.name,
                mTC.MCPTCRResult([
                    mTC.MCPTCRContentText(resp.response.contentData.decode('utf-8'))
                ])
            )
            self.send_mcp(resp.response.statusCode, resp.response.statusMsg, tcresp)
        except Exception as e:
            self.send_error(400, f"ERRR:PH:{e}")

    def mcp_toolslist(self, oRPC: Any):
        if not gMe.op.toolManager:
            raise RuntimeError("DBUG:PH:MCPToolsList:ToolManager uninitialised")
        tcl = mTC.MCPToolsList(oRPC["id"], mTC.MCPTLResult(gMe.op.toolManager.meta()))
        self.send_mcp(200, "tools/list follows", tcl)

    def mcp_run(self, body: bytes):
        oRPC = json.loads(body)
        if oRPC["method"] == "tools/call":
            self.mcp_toolscall(oRPC)
        elif oRPC["method"] == "tools/list":
            self.mcp_toolslist(oRPC)
        else:
            self.send_error(400, f"ERRR:PH:MCP:Unknown")

    def _do_POST(self):
        """
        Handle POST requests
        """
        print(f"DBUG:PH:Post:{self.address_string()}:{self.path}")
        print(f"DBUG:PH:Post:Headers:{self.headers}")
        if gMe.op.sslContext or gMe.sec.bAuthAlways:
            acGot = self.auth_check()
            if not acGot.callOk:
                self.send_error(acGot.statusCode, acGot.statusMsg)
                return
        pr = urllib.parse.urlparse(self.path)
        print(f"DBUG:PH:Post:{pr}")
        if pr.path != '/mcp':
            self.send_error(400, f"WARN:UnknownPath:{pr.path}")
            return
        bytesToRead = min(int(self.headers.get('Content-Length', -1)), gMe.nw.maxReadBytes)
        if bytesToRead <= -1:
            self.send_error(400, f"WARN:ContentLength missing:{pr.path}")
            return
        if bytesToRead == gMe.nw.maxReadBytes:
            self.send_error(400, f"WARN:RequestOverflow:{pr.path}")
            return
        body = self.rfile.read(bytesToRead)
        if len(body) != bytesToRead:
            self.send_error(400, f"WARN:ContentLength mismatch:{pr.path}")
            return
        self.mcp_run(body)

    def do_POST(self):
        """
        Catch all / trap any exceptions wrt actual post based request handling.
        """
        try:
            self._do_POST()
        except:
            print(f"ERRR:PH:ThePOST:{traceback.format_exception_only(sys.exception())}")
            self.send_error(500, f"ERRR: handling request")

    def do_GET(self):
        self.send_error(400, "Bad request")

    def do_OPTIONS(self):
        """
        Handle OPTIONS for CORS preflights (just in case from browser)
        """
        print(f"DBUG:ProxyHandler:OPTIONS:{self.path}")
        self.send_response(200)
        self.send_headers_common()

    def handle(self) -> None:
        """
        Helps handle ssl setup in the client specific thread, if in https mode
        """
        print(f"\n\n\nDBUG:ProxyHandler:Handle:RequestFrom:{self.client_address}")
        try:
            if (gMe.op.sslContext):
                self.request = gMe.op.sslContext.wrap_socket(self.request, server_side=True)
                self.setup()
        except:
            print(f"ERRR:ProxyHandler:SSLHS:{traceback.format_exception_only(sys.exception())}")
            return
        return super().handle()


def setup_toolmanager():
    """
    Setup the ToolCall helpers.
    Ensure the toolcall module is ok before setting up its tool calls.
    """
    gMe.op.toolManager = mTC.ToolManager()
    if mTCWeb.ok():
        gMe.op.toolManager.tc_add("fetch_url_raw", mTCWeb.TCUrlRaw("fetch_url_raw"))
        gMe.op.toolManager.tc_add("fetch_html_text", mTCWeb.TCHtmlText("fetch_html_text"))
        gMe.op.toolManager.tc_add("fetch_xml_filtered", mTCWeb.TCXmlFiltered("fetch_xml_filtered"))
        gMe.op.toolManager.tc_add("search_web_text", mTCWeb.TCSearchWeb("search_web_text"))
    if mTCPdf.ok():
        gMe.op.toolManager.tc_add("fetch_pdf_text", mTCPdf.TCPdfText("fetch_pdf_text"))


def setup_server():
    """
    Helps setup a http/https server
    """
    try:
        gMe.op.server = http.server.ThreadingHTTPServer(gMe.nw.server_address(), ProxyHandler)
        if gMe.sec.get('keyFile') and gMe.sec.get('certFile'):
            sslCtxt = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            sslCtxt.load_cert_chain(certfile=gMe.sec.certFile, keyfile=gMe.sec.keyFile)
            sslCtxt.minimum_version = ssl.TLSVersion.MAXIMUM_SUPPORTED
            sslCtxt.maximum_version = ssl.TLSVersion.MAXIMUM_SUPPORTED
            gMe.op.sslContext = sslCtxt
            print(f"INFO:SetupServer:Starting on {gMe.nw.server_address()}:Https mode")
        else:
            print(f"INFO:SetupServer:Starting on {gMe.nw.server_address()}:Http mode")
    except Exception as exc:
        print(f"ERRR:SetupServer:{traceback.format_exc()}")
        raise RuntimeError(f"SetupServer:{exc}") from exc


def run():
    try:
        setup_server()
        if not gMe.op.server:
            raise RuntimeError("Server missing!!!")
        gMe.op.server.serve_forever()
    except KeyboardInterrupt:
        print("INFO:Run:Shuting down...")
        if gMe.op.server:
            gMe.op.server.server_close()
        sys.exit(0)
    except Exception as exc:
        print(f"ERRR:Run:Exiting:Exception:{exc}")
        if gMe.op.server:
            gMe.op.server.server_close()
        sys.exit(1)


if __name__ == "__main__":
    gMe.process_args(sys.argv)
    setup_toolmanager()
    run()
