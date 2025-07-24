VERSION ?= 25.07.0

.PHONY: build-cpu build-cuda build-rocm build-all clean

build-cpu:
	@echo "🔧 Building lm-cpu:${VERSION}"
	docker build --no-cache -t lm-cpu:${VERSION} -f .devops/cpu.Dockerfile .

build-cuda:
	@echo "🔧 Building lm-cuda:${VERSION}"
	docker build --no-cache -t lm-cuda:${VERSION} -f .devops/cuda.Dockerfile .

build-rocm:
	@echo "🔧 Building lm-rocm:${VERSION}"
	docker build --no-cache -t lm-rocm:${VERSION} -f .devops/rocm.Dockerfile .

build-all: build-cpu build-cuda build-rocm

clean:
	@echo "🧹 Cleaning..."
	@docker builder prune -f
	@docker rmi lm-cpu:${VERSION} lm-cuda:${VERSION} lm-rocm:${VERSION} || true
