#!/bin/bash
set -e

INSTALL_DIR="${HOME}/neuroticks"
VENV_DIR="${INSTALL_DIR}/venv"
NEST_SRC="${INSTALL_DIR}/nest-src"
NEST_BUILD="${INSTALL_DIR}/nest-build"
NEUROTICKS_DIR="${INSTALL_DIR}/NeuroTicks"
PYTHON_VERSION="3.12"
NUM_CORES=$(nproc)

sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git wget curl autoconf automake libtool pkg-config \
    libgsl-dev libltdl-dev libncurses-dev libreadline-dev \
    libboost-dev libboost-filesystem-dev libboost-system-dev cython3 \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    qt6-base-dev libqt6opengl6-dev libgl1-mesa-dev libglu1-mesa-dev libegl1 \
    libxrender1 libxcb-xinerama0 libxcb-cursor0 libxkbcommon-x11-0 \
    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
    libxcb-render-util0 libxcb-shape0 libdbus-1-3 libfontconfig1 libgtk-3-0

mkdir -p "${INSTALL_DIR}" "${NEST_BUILD}"

python${PYTHON_VERSION} -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip wheel setuptools
pip install numpy 'cython<3'
pip install pandas scipy networkx matplotlib
pip install PyQt6 PyQt6-WebEngine pyqtgraph
pip install vtk pyvista pyvistaqt
pip install PyOpenGL==3.1.10
pip install ipython

git clone --branch v3.8 --depth 1 https://github.com/nest/nest-simulator.git "${NEST_SRC}"

cd "${NEST_BUILD}"
cmake "${NEST_SRC}" \
    -DCMAKE_INSTALL_PREFIX="${VENV_DIR}" \
    -DPYTHON_EXECUTABLE="${VENV_DIR}/bin/python" \
    -Dwith-python=ON \
    -Dwith-gsl=ON \
    -Dwith-readline=ON \
    -Dwith-ltdl=ON \
    -Dwith-openmp=ON \
    -Dwith-boost=ON \
    -DCMAKE_BUILD_TYPE=Release

make -j${NUM_CORES}
make install

git clone https://github.com/Yinsalt/NeuroTicks.git "${NEUROTICKS_DIR}"

cat > "${INSTALL_DIR}/activate.sh" << EOF
#!/bin/bash
source "${VENV_DIR}/bin/activate"
export NEUROTICKS_DIR="${NEUROTICKS_DIR}"
cd "${NEUROTICKS_DIR}"
EOF
chmod +x "${INSTALL_DIR}/activate.sh"

echo ""
echo "============================================"
echo "  Installation complete!"
echo "============================================"
echo ""
echo "  Virtual environment: ${VENV_DIR}"
echo "  NeuroTicks location: ${NEUROTICKS_DIR}"
echo ""
echo "  To run NeuroTicks:"
echo "    source ${INSTALL_DIR}/activate.sh"
echo "    python Main.py"
echo ""
