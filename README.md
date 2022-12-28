# YOLO inference using ONNX model

Run YOLO inference in C++ or Python using ONNX model

### Prerequisites/Tested On

- OpenCV 4.6.0
- Python 3.8
- CMake 3.5.1
- C++ 17

# Config

- Change the input image, class names, and model path in `src/yolo_inference.cpp` or `src/yolo_inference.py`

### Use below drop-down to see the steps

<details><summary>C++</summary><br/>

### Build

- Clone the repository
- Create a build directory
- Run cmake
- Run make

#### Steps

```bash
git clone https://github.com/kvnptl/yolo-inference-onnx.git
cd yolo-inference-onnx
mkdir build
cd build
cmake ..
make
```

### Run

- Go to the build directory
- Run the executable

```bash
./yolo_inference
```
</details>

<details><summary>Python</summary><br/>

```bash
python3 src/yolo_inference.py
```

</details>
<!-- ### Python -->
