# Cross-compilation autoframe for NXP

Sometimes you need to cross-compile MediaPipe source code, e.g. get `ARM32`
or `ARM64` binaries on `x86` system. Install cross-compilation toolchain on
your system or use our preconfigured Docker environment for that:

### First time
```bash
# For ARM32 (e.g. Raspberry Pi)
make -C mediapipe/examples/coral PLATFORM=armhf docker

# For ARM64 (e.g. Coral Dev Board)
make -C mediapipe/examples/coral PLATFORM=arm64 docker
```
### Second time onwards
You only have to run the above command first time, the environment is saved on disk so that you can compile the application again. Follow the following steps to compile the autoframe binary second time onwards

```bash
docker run -i BuildContainer
```

### Installing Opencv 4.4
The docker tries to use Opencv 3.4 but NXP EVK uses Opencv 4.4.0, the repository supplies the precompiled binaries in opencv32_arm64_libs_nxp , but we still have to install the header files.

```bash
cd ~
git clone https://github.com/opencv/opencv.git 
cd opencv
git checkout tags/4.4.0 -b 4.4.0
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j4 
sudo make install
```

After running this command you'll get a shell to the Docker environment which
has everything ready to start compilation:

```bash
# For ARM32 (e.g. Raspberry Pi)
bazel build \
    --crosstool_top=@crosstool//:toolchains \
    --compiler=gcc \
    --cpu=armv7a \
    --define MEDIAPIPE_DISABLE_GPU=1 \
    mediapipe/examples/desktop/auto_frame:autoframe_cpu

# For ARM64 (e.g. NXP Dev Board)
bazel build \
    --crosstool_top=@crosstool//:toolchains \
    --compiler=gcc \
    --cpu=aarch64 \
    --define MEDIAPIPE_DISABLE_GPU=1 \
    mediapipe/examples/desktop/auto_frame:autoframe_cpu
```


To summarize everything:

| Arch  | PLATFORM       | Output      | Board                                                    |
| ----- | -------------- | ----------- | -------------------------------------------------------- |
| ARM32 | PLATFORM=armhf | out/armv7a  | Raspberry Pi  |
| ARM64 | PLATFORM=arm64 | out/aarch64 | NXP Dev Board |

## Output directory

Output will be generated in bazel-bin directory , you can copy the binary file along with supporting graphs and model files to the device to test .
