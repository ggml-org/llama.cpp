# SPDX-License-Identifier: MIT
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import apply_architecture_port as port
class PortApplyTests(unittest.TestCase):
    def fixture(self,root):
        p=root/'payload';t=root/'module';p.mkdir();t.mkdir()
        (p/'a.txt').write_text('new');(t/'a.txt').write_text('old')
        (p/'added.txt').write_text('added')
        (p/'PORT_PATCH_FILES.json').write_text(json.dumps({'files':[
            {'path':'a.txt','old_sha256':port.digest(t/'a.txt'),'new_sha256':port.digest(p/'a.txt')},
            {'path':'added.txt','old_sha256':None,'new_sha256':port.digest(p/'added.txt')}]}))
        return p,t
    def test_preflight_apply_idempotence(self):
        with tempfile.TemporaryDirectory() as td:
            p,t=self.fixture(Path(td));r=port.apply(t,p);self.assertFalse(r['applied']);self.assertEqual((t/'a.txt').read_text(),'old')
            r=port.apply(t,p,True);self.assertTrue(r['applied']);self.assertEqual((t/'a.txt').read_text(),'new')
            self.assertEqual((Path(r['backup'])/'a.txt').read_text(),'old');self.assertEqual(port.apply(t,p)['already_applied'],2)
    def test_conflict_fails_without_write(self):
        with tempfile.TemporaryDirectory() as td:
            p,t=self.fixture(Path(td));(t/'added.txt').write_text('user data')
            with self.assertRaises(ValueError):port.apply(t,p,True)
            self.assertEqual((t/'a.txt').read_text(),'old')
    def test_payload_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            p,t=self.fixture(Path(td));(p/'a.txt').write_text('wrong')
            with self.assertRaises(ValueError):port.apply(t,p,True)
    def test_traversal(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):port.safe_path(Path(td),'../x')
            with self.assertRaises(ValueError):port.safe_path(Path(td),'.git/config')
if __name__=='__main__':unittest.main()
