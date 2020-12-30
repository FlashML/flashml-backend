FROM python:3.8

WORKDIR usr/src/flash_ml
COPY requirements.txt .

RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx \ 
    && pip3 install --upgrade pip \
    && pip3 install --no-cache-dir -r requirements.txt

COPY wsgi.py ./
COPY app ./app/
COPY data ./data/

EXPOSE 5000
CMD ["gunicorn",  "-w", "1", "-b", "0.0.0.0:5000", "wsgi:app"]
