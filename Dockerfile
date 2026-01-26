FROM runpod/base:0.6.2-cuda12.2.0

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN uv pip install --system --no-cache -r /workspace/requirements.txt

# your handler is in src/
COPY src/handler.py /workspace/handler.py

CMD ["python3", "-u", "/workspace/handler.py"]
