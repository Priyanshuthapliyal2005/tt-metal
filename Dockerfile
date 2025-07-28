FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y wget sudo git python3 python3-pip python3-venv build-essential && \
    apt-get clean

# Download and run Tenstorrent's dependency and SFPI install scripts
RUN wget https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/install_dependencies.sh && \
    wget https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/tt_metal/sfpi-version.sh && \
    chmod a+x install_dependencies.sh && \
    sudo ./install_dependencies.sh && \
    rm install_dependencies.sh sfpi-version.sh

# Install Poetry for pyproject.toml support
RUN pip3 install poetry

# Copy your project files into the image
WORKDIR /workspace
COPY . .

# Install Python dependencies from pyproject.toml
RUN poetry install

# Install dev requirements
RUN python3 -m pip install -r tt_metal/python_env/requirements-dev.txt

# (Optional) Add your docs step here if needed
# COPY docs/ /workspace/docs/

# Set environment variables
ENV PYTHONPATH=/workspace

# Run your demo script
CMD ["python3", "models/tt_transformers/demo/simple_text_demo.py"]
