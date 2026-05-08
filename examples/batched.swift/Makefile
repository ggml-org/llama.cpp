.PHONY: build

build:
	xcodebuild -scheme llama-batched-swift -destination "generic/platform=macOS" -derivedDataPath build
	rm -f ./llama-batched-swift
	ln -s ./build/Build/Products/Debug/llama-batched-swift ./llama-batched-swift
