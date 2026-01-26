FROM runpod/base:0.6.0-cuda12.1.1

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN uv pip install --system --no-cache -r /workspace/requirements.txt

# copy handler from src/
COPY src/handler.py /workspace/handler.py

CMD ["python3", "-u", "/workspace/handler.py"]
