FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# The real Context Runtime geospatial capability (dependency-free engine + contracts) + the harvested data.
COPY context_runtime context_runtime
COPY data data
COPY app.py .
COPY static/ static/
EXPOSE 8098
# --loop asyncio / --http h11: the host's AppArmor 4.1 blocks socketpair() in uvloop/httptools (same as
# grc/chat/gov).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8098", "--loop", "asyncio", "--http", "h11"]
