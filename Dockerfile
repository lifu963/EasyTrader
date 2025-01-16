FROM python:3.11-slim

WORKDIR /usr/src/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY automated_trading/  /usr/src/app/automated_trading/
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV PYTHONPATH="/usr/src/app"

CMD ["python", "-m", "automated_trading.main"]