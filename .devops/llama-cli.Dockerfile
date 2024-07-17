ARG UBUNTU_VERSION=22.04

FROM ubuntu:$UBUNTU_VERSION AS build

RUN apt-get update && \
    apt-get install -y build-essential git

WORKDIR /app

COPY . .

RUN make -j$(nproc) llama-cli

FROM ubuntu:$UBUNTU_VERSION AS runtime

RUN apt-get update && \
    apt-get install -y libgomp1

COPY --from=build /app/llama-cli /llama-cli

ENV LC_ALL=C.utf8

ENTRYPOINT [ "/llama-cli" ]
