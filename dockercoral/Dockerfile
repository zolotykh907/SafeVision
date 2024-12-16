
FROM debian:buster

RUN apt-get update --allow-releaseinfo-change && \
apt-get install curl gnupg ca-certificates zlib1g-dev libjpeg-dev git apt-utils -y


RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

RUN apt-get update --allow-releaseinfo-change && \
    apt-get install python3 python3-pip -y
RUN apt-get install libedgetpu1-legacy-std python3-edgetpu --reinstall -y
RUN apt-get install nano -y
RUN apt-get install python3-pycoral python3-tflite-runtime  -y
RUN apt-get install libblas-dev liblapack-dev -y
RUN apt install python3-sklearn python3-skimage python3-pandas -y
RUN apt install libffi-dev

RUN apt install llvm-11X** -y

RUN apt-get install python3-opencv -y
RUN apt-get install libgl1-mesa-glx -y
RUN apt-get install libxml2-dev libxslt-dev python-dev zlib1g-dev -y
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    python3-dev \
    gcc
RUN pip3 install --upgrade pip
RUN pip3 install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp37-cp37m-linux_x86_64.whl
RUN pip3 install pillow
RUN apt-get install libedgetpu1-std -y
RUN apt-get install libopencv-dev -y
RUN apt-get install libsm6 libxrender1 libfontconfig1
RUN pip3 install opencv-python==4.1.2.30
# RUN pip3 install scipy
# RUN pip3 install scikit-learn
RUN pip3 install librosa --ignore-installed joblib
RUN pip3 install tensorflow
RUN pip3 install IPython
RUN pip3 install matplotlib
RUN pip3 install numpy
RUN pip3 install tflite-runtime==2.11.0
WORKDIR /app/data
RUN export QT_X11_NO_MITSHM=1
