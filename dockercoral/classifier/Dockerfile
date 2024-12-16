FROM coral-python:1

RUN apt-get install python3-opencv -y

RUN mkdir /coral
WORKDIR /coral
RUN git clone https://github.com/google-coral/tflite.git
WORKDIR /coral/tflite/python/examples/classification

RUN bash install_requirements.sh
WORKDIR /app/data

EXPOSE 5005
