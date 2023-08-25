# hash:sha256:eee14cd5c11b13604efe5b91a179a1cbfe649ff26b130163ce2d52d19ee3ac3e
FROM registry.codeocean.allenneuraldynamics.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl=7.68.0-1ubuntu2.19 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y \
        umap \
    && conda clean -ya

COPY postInstall /
RUN /postInstall
