# ToolCalls and MCP related types and bases
# by Humans for All

from typing import Any, TypeAlias
from dataclasses import dataclass, field



#
# Dataclasses to help with Tool Calls
#

TCInArgs: TypeAlias = dict[str, Any]

@dataclass
class TCInProperty():
    type: str
    description: str

TCInProperties: TypeAlias = dict[str, TCInProperty]

@dataclass
class TCInParameters():
    type: str = "object"
    properties: TCInProperties = field(default_factory=dict)
    required: list[str] = field(default_factory=list)

@dataclass
class TCFunction():
    name: str
    description: str
    parameters: TCInParameters ### Delta wrt naming btw OpenAi Tools HS (parameters) and MCP(inputSchema)

@dataclass
class ToolCallMeta(): ### Delta wrt tree btw OpenAi Tools HS (Needs this wrapper) and MCP (directly use TCFunction)
    type: str = "function"
    function: TCFunction|None = None

@dataclass(frozen=True)
class TCOutResponse:
    """
    Used to return result from tool call.
    """
    callOk: bool
    statusCode: int
    statusMsg: str = ""
    contentType: str = ""
    contentData: bytes = b""

@dataclass
class ToolCallResponseEx():
    tcid: str
    name: str
    response: TCOutResponse

@dataclass(frozen=True)
class MCPTCRContentText:
    text: str
    type: str = "text"

@dataclass
class MCPTCRResult:
    content: list[MCPTCRContentText]

@dataclass
class MCPToolCallResponse:
    id: str
    name: str
    result: MCPTCRResult
    jsonrpc: str = "2.0"

#HttpHeaders: TypeAlias = dict[str, str] | email.message.Message[str, str]
HttpHeaders: TypeAlias = dict[str, str]


@dataclass
class ToolCall():
    name: str

    def tcf_meta(self) -> TCFunction|None:
        return None

    def tc_handle(self, args: TCInArgs, inHeaders: HttpHeaders) -> TCOutResponse:
        return TCOutResponse(False, 500)


MCPTLTools: TypeAlias = list[ToolCallMeta]

@dataclass
class MCPTLResult:
    tools: MCPTLTools

@dataclass
class MCPToolsList:
    id: str
    result: MCPTLResult
    jsonrpc: str = "2.0"


class ToolManager():

    def __init__(self) -> None:
        self.toolcalls: dict[str, ToolCall] = {}

    def tc_add(self, fName: str, tc: ToolCall):
        self.toolcalls[fName] = tc

    def meta(self):
        lMeta: MCPTLTools = []
        for tcName in self.toolcalls.keys():
            tcfMeta = self.toolcalls[tcName].tcf_meta()
            lMeta.append(ToolCallMeta("function", tcfMeta))
        return lMeta

    def tc_handle(self, callId: str, tcName: str, tcArgs: TCInArgs, inHeaders: HttpHeaders) -> ToolCallResponseEx:
        try:
            response = self.toolcalls[tcName].tc_handle(tcArgs, inHeaders)
            return ToolCallResponseEx(callId, tcName, response)
        except KeyError:
            return ToolCallResponseEx(callId, tcName, TCOutResponse(False, 400, "Unknown tool call"))
