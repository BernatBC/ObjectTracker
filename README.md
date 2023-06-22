# Object Tracker
Object Tracker is a recopilation of algorithms made specifically for [Long-Term Visual Object Tracking Benchmark](https://amoudgl.github.io/tlp/).

## Features
- Tracking by histogram comparison
- Tracking by SIFT and RANSAC
- Recognition using [Yolov8](https://docs.ultralytics.com/)
- Recognition using [Yolov8](https://docs.ultralytics.com/) trained by ourselves

## Getting Started

### Tracking by histogram comparison
#### 1. Install dependencies
The following dependencies are required:
- Matlab (latest version not necessary but recommended)

#### 2. Get the source code
To get the source code, you can simply download the zip file, or you can clone this repository by typing:

```bash
git clone https://github.com/BernatBC/ObjectTracker.git
```

#### 3. Launch Matlab and run code
You'll need to execute Matlab. Then you just open `tracking_nostre.mlx`, and press the Button Run. 

Make sure you have `TinyTLP` folder in the same directory as the `tracking_nostre.mlx`.

### Tracking by SIFT and RANSAC
You need to follow the same steps as `Tracking by histogram comparison`, but with the source code file `tracking_internet.mlx`.

### Recognition using [Yolov8](https://docs.ultralytics.com/)
#### 1. Install dependencies
The following dependencies are required:
- python3
- pip

After the installation of those packages, we'll install the following python packages:
```bash
pip install ultralytics
```

#### 2. Get the source code
To get the source code, you can simply download the zip file, or you can clone this repository by typing:

```bash
git clone https://github.com/BernatBC/ObjectTracker.git
```

#### 3. Run python script
From the repository directory, run the following command:

```bash
python3 reconeixament_internet.py
```
Make sure you have `TinyTLP` folder in the same directory as the `reconeixament_internet.py`.

### Recognition using [Yolov8](https://docs.ultralytics.com/) trained by ourselves
You need to follow the steps 1 and 2 from `Recognition using Yolov8`.

#### 3. Run python script
From the repository directory, run the following command:

```bash
python3 reconeixament_nostre.py
```
Make sure you have `TinyTLP` folder in the same directory as the `reconeixament_nostre.py`.