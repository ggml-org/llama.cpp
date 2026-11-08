{
  lib,
  llamaVersion,
  numpy,
  tqdm,
  requests,
  sentencepiece,
  pyyaml,
  poetry-core,
  buildPythonPackage,
  pytestCheckHook,
}:

buildPythonPackage {
  pname = "gguf";
  version = llamaVersion;
  pyproject = true;
  nativeBuildInputs = [ poetry-core ];
  propagatedBuildInputs = [
    numpy
    tqdm
    sentencepiece
    pyyaml
    requests
  ];
  src = lib.cleanSource ../../gguf-py;
  pythonImportsCheck = [
    "numpy"
    "gguf"
  ];
  nativeCheckInputs = [ pytestCheckHook ];
  doCheck = true;
  meta = with lib; {
    description = "Python package for writing binary files in the GGUF format";
    license = licenses.mit;
    maintainers = [ maintainers.ditsuke ];
  };
}
