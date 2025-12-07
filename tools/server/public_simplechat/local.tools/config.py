# Config entries
# by Humans for All
#

from dataclasses import dataclass, field, fields
from typing import Any, Optional
import http.server
import ssl
import sys
import urlvalidator as mUV
import debug as mDebug
import toolcalls as mTC


gConfigNeeded = [ 'acl.schemes', 'acl.domains', 'sec.bearerAuth' ]


@dataclass
class DictyDataclassMixin():
    """
    Mixin to ensure dataclass attributes are also accessible through
    dict's [] style syntax and get helper.
    """

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key, default=None):
        try:
            return self[key]
        except:
            return default


@dataclass
class Sec(DictyDataclassMixin):
    """
    Used to store security related config entries
    """
    certFile: str = ""
    keyFile: str = ""
    bearerAuth: str = ""
    bAuthAlways: bool = True


@dataclass
class ACL(DictyDataclassMixin):
    """
    Used to store access control related config entries
    """
    schemes: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)


@dataclass
class Network(DictyDataclassMixin):
    """
    Used to store network related config entries
    """
    port: int = 3128
    addr: str = ''
    maxReadBytes: int = 1*1024*1024

    def server_address(self):
        return (self.addr, self.port)


@dataclass
class Op(DictyDataclassMixin):
    """
    Used to store runtime operation related config entries and states
    """
    configFile: str = "/dev/null"
    debug: bool = False
    server: http.server.ThreadingHTTPServer|None = None
    sslContext: ssl.SSLContext|None = None
    toolManager: mTC.ToolManager|None = None
    bearerTransformed: str = ""
    bearerTransformedYear: str = ""


@dataclass
class Config(DictyDataclassMixin):
    op: Op = field(default_factory=Op)
    sec: Sec = field(default_factory=Sec)
    acl: ACL = field(default_factory=ACL)
    nw: Network = field(default_factory=Network)

    def get_type(self, keyTree: str):
        cKeyList = keyTree.split('.')
        cur = self
        for k in cKeyList[:-1]:
            cur = self[k]
        return type(cur[cKeyList[-1]])

    def get_value(self, keyTree: str):
        cKeyList = keyTree.split('.')
        cur = self
        for k in cKeyList[:-1]:
            cur = self[k]
        return cur[cKeyList[-1]]

    def set_value(self, keyTree: str, value: Any):
        cKeyList = keyTree.split('.')
        cur = self
        for k in cKeyList[:-1]:
            cur = self[k]
        cur[cKeyList[-1]] = value

    def validate(self):
        for k in gConfigNeeded:
            if self.get_value(k) == None:
                print(f"ERRR:ProcessArgs:Missing:{k}:did you forget to pass the config file...")
                exit(104)
        mDebug.setup(self.op.debug)
        if (self.acl.schemes and self.acl.domains):
            mUV.validator_setup(self.acl.schemes, self.acl.domains)

    def load_config(self, configFile: str):
        """
        Allow loading of a json based config file

        The config entries should be named same as their equivalent cmdline argument
        entries but without the -- prefix.

        As far as the logic is concerned the entries could either come from cmdline
        or from a json based config file.
        """
        import json
        self.op.configFile = configFile
        with open(self.op.configFile) as f:
            cfgs: dict[str, Any] = json.load(f)
            for cfg in cfgs:
                print(f"DBUG:LoadConfig:{cfg}")
                try:
                    neededType = self.get_type(cfg)
                    gotValue = cfgs[cfg]
                    gotType = type(gotValue)
                    if gotType.__name__ != neededType.__name__:
                        print(f"ERRR:LoadConfig:{cfg}:expected type [{neededType}] got type [{gotType}]")
                        exit(112)
                    self.set_value(cfg, gotValue)
                except KeyError:
                    print(f"ERRR:LoadConfig:{cfg}:UnknownConfig!")
                    exit(113)

    def process_args(self, args: list[str]):
        """
        Helper to process command line arguments.

        Flow setup below such that
        * location of --config in commandline will decide whether command line or config file will get
        priority wrt setting program parameters.
        * str type values in cmdline are picked up directly, without running them through ast.literal_eval,
        bcas otherwise one will have to ensure throught the cmdline arg mechanism that string quote is
        retained for literal_eval
        """
        import ast
        print(self)
        iArg = 1
        while iArg < len(args):
            cArg = args[iArg]
            if (not cArg.startswith("--")):
                print(f"ERRR:ProcessArgs:{iArg}:{cArg}:MalformedCommandOr???")
                exit(101)
            cArg = cArg[2:]
            print(f"DBUG:ProcessArgs:{iArg}:{cArg}")
            try:
                aTypeCheck = self.get_type(cArg)
                aValue = args[iArg+1]
                if aTypeCheck.__name__ != 'str':
                    aValue = ast.literal_eval(aValue)
                    aType = type(aValue)
                    if aType.__name__ != aTypeCheck.__name__:
                        print(f"ERRR:ProcessArgs:{iArg}:{cArg}:expected type [{aTypeCheck}] got type [{aType}]")
                        exit(102)
                self.set_value(cArg, aValue)
                iArg += 2
                if cArg == 'op.configFile':
                    self.load_config(aValue)
            except KeyError:
                print(f"ERRR:ProcessArgs:{iArg}:{cArg}:UnknownArgCommand!:{sys.exception()}")
                exit(103)
        print(self)
        self.validate()
