# Minimal image providing XRT's xclbinutil (Ubuntu packages a working xclbinutil from 26.04 on).
# Used only to pack IRON-generated NPU artifacts into an .xclbin; no NPU access needed.
FROM ubuntu:26.04
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libxrt-utils \
 && rm -rf /var/lib/apt/lists/*
ENTRYPOINT ["xclbinutil"]
