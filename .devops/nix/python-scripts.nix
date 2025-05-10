{
  lib,
  stdenv,
  buildPythonPackage,
  poetry-core,
  mkShell,
  python3Packages,
  gguf-py,
}@inputs:

let
  llama-python-deps = with python3Packages; [
    numpy
    sentencepiece
    transformers
    protobuf
    torchWithoutCuda
    gguf-py
    tqdm

    # for scripts/compare-llama-bench.py
    gitpython
    tabulate

    # for examples/pydantic-models-to-grammar-examples.py
    docstring-parser
    pydantic

  ];

  llama-python-test-deps = with python3Packages; [
    # Server bench
    matplotlib

    # server tests
    openai
    pytest
    prometheus-client
  ];
in

buildPythonPackage ({
  pname = "llama-scripts";
  version = "0.0.0";
  pyproject = true;

  # NOTE: The files filtered out here are not visible in the build sandbox, neither
  # do they affect the output hash. They can be modified without triggering a rebuild.
  src = lib.cleanSourceWith {
    filter =
      name: type:
      let
        any = builtins.any (x: x);
        baseName = builtins.baseNameOf name;
      in
      any [
        (lib.hasSuffix ".py" name)
        (baseName == "README.md")
        (baseName == "pyproject.toml")
      ];
    src = lib.cleanSource ../../.;
  };
  nativeBuildInputs = [ poetry-core ];
  nativeCheckInputs = llama-python-test-deps;
  dependencies = llama-python-deps;
})
