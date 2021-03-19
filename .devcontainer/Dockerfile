# See here for image contents: https://github.com/microsoft/vscode-dev-containers/blob/master/containers/python-3-anaconda/.devcontainer/base.Dockerfile
ARG VARIANT="3"
FROM mcr.microsoft.com/vscode/devcontainers/anaconda:0-${VARIANT}

# Additional packages
RUN sudo apt-get update
RUN sudo apt-get install --reinstall build-essential -y

# Get local user
ARG USERNAME=vscode

# Change conda to be owned by the local user
RUN chown -R $USERNAME:$USERNAME /opt/conda

# Activate local user
USER $USERNAME

# Conda init
RUN conda init bash

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY requirements.txt /tmp/pip-tmp/
RUN pip --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
    && sudo rm -rf /tmp/pip-tmp