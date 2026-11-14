import argparse
from pathlib import Path
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as t
import numpy as np
parser = argparse.ArgumentParser(description="Build an ORT baseline for the captured 8-frame codec block")
parser.add_argument("capture_directory", type=Path)
p = parser.parse_args().capture_directory
w=np.fromfile(p/'weight.f32','<f4').reshape(384,384,7)
b=np.fromfile(p/'bias.f32','<f4')
g=h.make_graph([h.make_node('Conv',['x','w','b'],['y'])],'codec-conv',
 [h.make_tensor_value_info('x',t.FLOAT,[1,384,1286])],
 [h.make_tensor_value_info('y',t.FLOAT,[1,384,1280])],
 [nh.from_array(w,'w'),nh.from_array(b,'b')])
m=h.make_model(g,opset_imports=[h.make_opsetid('',17)],ir_version=9)
onnx.checker.check_model(m)
onnx.save(m,p/'conv.onnx')
