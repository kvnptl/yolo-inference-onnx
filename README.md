# YOLO inference using ONNX model

Run YOLO inference in C++ or Python using ONNX model

![output](https://user-images.githubusercontent.com/47410011/210271002-06f079c0-7c30-4401-b0df-c0c92cb6043a.jpg)*Pic credit: Matthias Hangst/Getty Images*

### Prerequisites/Tested On

- OpenCV 4.6.0
- Python 3.8
- CMake 3.5.1
- C++ 17
- Yolov5 

# Config

- Change the input image, class names, and model path in `src/yolo_inference.cpp` or `src/yolo_inference.py`

### Use below drop-down to see the steps

<!-- C++ -->
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

(OPTIONAL) Note: there is also a header file `include/yolo_inference.hpp` which contains the inference function. You can use that function in your own code if you want.
</details>

<!-- ### Python -->

<details><summary>Python</summary><br/>

```bash
python3 src/yolo_inference.py
```

</details>
