FROM python:3.9.17-slim-bullseye
WORKDIR /app
COPY . .
RUN pip install --upgrade pi
RUN pip install -r requirements.txt
EXPOSE 5000/tcp
CMD ["python", "server.py"]