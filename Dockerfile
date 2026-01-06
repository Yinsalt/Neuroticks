FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV INSTALL_DIR=/opt/neuroticks
ENV VENV_DIR=${INSTALL_DIR}/venv
ENV NEST_SRC=${INSTALL_DIR}/nest-src
ENV NEST_BUILD=${INSTALL_DIR}/nest-build
ENV NEUROTICKS_DIR=${INSTALL_DIR}/NeuroTicks

RUN apt-get update && apt-get install -y \
    build-essential cmake git wget curl autoconf automake libtool pkg-config \
    libgsl-dev libltdl-dev libncurses-dev libreadline-dev \
    libboost-dev libboost-filesystem-dev libboost-system-dev cython3 \
    python3.12 python3.12-dev python3.12-venv \
    qt6-base-dev libqt6opengl6-dev libgl1-mesa-dev libglu1-mesa-dev libegl1 \
    libxrender1 libxcb-xinerama0 libxcb-cursor0 libxkbcommon-x11-0 \
    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
    libxcb-render-util0 libxcb-shape0 libdbus-1-3 libfontconfig1 libgtk-3-0 \
    libxcb1 libx11-xcb1 libxi6 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p ${INSTALL_DIR} ${NEST_BUILD}

RUN python3.12 -m venv ${VENV_DIR}

ENV PATH="${VENV_DIR}/bin:$PATH"
ENV VIRTUAL_ENV="${VENV_DIR}"

RUN pip install --upgrade pip wheel setuptools && \
    pip install numpy 'cython<3' && \
    pip install pandas scipy networkx matplotlib && \
    pip install PyQt6 PyQt6-WebEngine pyqtgraph && \
    pip install vtk pyvista pyvistaqt && \
    pip install PyOpenGL==3.1.10 && \
    pip install ipython

RUN git clone --branch v3.8 --depth 1 https://github.com/nest/nest-simulator.git ${NEST_SRC}

WORKDIR ${NEST_BUILD}
RUN cmake ${NEST_SRC} \
    -DCMAKE_INSTALL_PREFIX=${VENV_DIR} \
    -DPYTHON_EXECUTABLE=${VENV_DIR}/bin/python \
    -Dwith-python=ON \
    -Dwith-gsl=ON \
    -Dwith-readline=ON \
    -Dwith-ltdl=ON \
    -Dwith-openmp=ON \
    -Dwith-boost=ON \
    -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install

RUN git clone https://github.com/Yinsalt/NeuroTicks.git ${NEUROTICKS_DIR}

RUN rm -rf ${NEST_SRC} ${NEST_BUILD}

WORKDIR ${NEUROTICKS_DIR}

ENV QT_QPA_PLATFORM=xcb
ENV DISPLAY=:0

CMD ["python", "Main.py"]
