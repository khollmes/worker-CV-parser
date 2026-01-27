FROM runpod/base:0.6.2-cuda12.2.0

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

RUN python3 -m pip install --no-cache-dir -r /workspace/requirements.txt

COPY src/handler.py /workspace/handler.py

CMD ["python3", "-u", "/workspace/handler.py"]
