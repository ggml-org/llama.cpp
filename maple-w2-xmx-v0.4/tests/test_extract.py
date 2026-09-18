#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('extract_tq2', Path(__file__).parents[1] / 'tools' / 'extract_tq2.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
def string(x):
    b=x.encode();return struct.pack('<Q',len(b))+b

def gguf_bytes(ty=35, alignment=32, truncate=0):
    payload = (bytes([0x55])*64+struct.pack('<H',0x3400))*8
    h=b'GGUF'+struct.pack('<IQQ',3,1,2)
    h+=string('general.alignment')+struct.pack('<II',4,alignment)
    h+=string('tokenizer.test')+struct.pack('<IIQ',9,8,2)+string('a')+string('나')
    h+=string('blk.0.ffn_gate_exps.weight')+struct.pack('<IQQQIQ',3,256,8,1,ty,0)
    if alignment > 0:
        h+=b'\x00'*((-len(h))%alignment)
    data=h+payload
    return data[:-truncate] if truncate else data

class TestExtract(unittest.TestCase):
    def make(self,root,**kw):
        p=root/'m.gguf';p.write_bytes(gguf_bytes(**kw));return p
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.make(root)
            with p.open('rb') as f:
                g=mod.GGUF(f);info=g.extract('blk.0.ffn_gate_exps.weight',root/'x.mw2')
            raw=(root/'x.mw2').read_bytes()
            self.assertEqual(len(raw),32+528)
            self.assertEqual(struct.unpack('<8sIIIIQ',raw[:32]),(mod.MAGIC,1,256,8,1,528))
            self.assertEqual(info['payload_bytes'],528)
            self.assertEqual(raw[32:],gguf_bytes()[-528:])
    def test_custom_q2_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.make(root,ty=42)
            with p.open('rb') as f:
                g=mod.GGUF(f)
                with self.assertRaises(ValueError):g.extract('blk.0.ffn_gate_exps.weight',root/'x.mw2')
    def test_truncated(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.make(root,truncate=1)
            with p.open('rb') as f:
                g=mod.GGUF(f)
                with self.assertRaises(ValueError):g.extract('blk.0.ffn_gate_exps.weight',root/'x.mw2')
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.make(root);out=root/'x.mw2';out.write_bytes(b'KEEP')
            with p.open('rb') as f:
                g=mod.GGUF(f)
                with self.assertRaises(FileExistsError):g.extract('blk.0.ffn_gate_exps.weight',out)
            self.assertEqual(out.read_bytes(),b'KEEP')
    def test_alignment(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.make(root,alignment=64)
            with p.open('rb') as f:
                g=mod.GGUF(f);self.assertEqual(g.data_offset%64,0)
    def test_malformed_header(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'m.gguf';p.write_bytes(b'GGUF'+b'\xff'*20)
            with p.open('rb') as f:
                with self.assertRaises(ValueError):mod.GGUF(f)
if __name__=='__main__':unittest.main()
