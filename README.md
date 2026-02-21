# Ambiguity-Aware Driver State Recognition Using Multimodal Sensing, Vision–Language Model, and LLM-Based Reasoning

A late-fusion driver monitoring system built on ROS2 combining IR camera-based facial analysis, vehicle dynamics from CARLA, event-triggered VLM scene interpretation, and LLM-mediated reasoning. The system is **ambiguity-aware**: rather than forcing a hard classification when signals conflict or degrade, an LLM reasons over uncertainty across modalities and produces interpretable, conservative driver state decisions.

> For full methodology, dataset details, experiments, and results see [`report/report.pdf`](report/report.pdf).  
> Authors: Adit Raj Venkataraj, Harshitha Nidaghatta Siddaraju, Nikhil Prakash Katti, Lloyd Alston Dsouza  

---

## How it works

Per-minute features (PERCLOS, blink rate, blink duration, steering entropy, steering reversal rate, SDLP) are computed from IR camera and CARLA steering data over fixed 60-second windows and fed into two independent Random Forest classifiers. When heuristic ambiguity flags are raised, **Qwen2.5-VL (3B)** is triggered and receives 8 IR frames over a 2-second window, returning a structured scene description. **LLaMA 3.1 (8B)** then fuses the ML outputs, confidence scores, and VLM context to produce a final driver state (Alert / Drowsy / Very Drowsy / Unknown), an intervention action, and a natural-language reasoning trace. Both models run fully locally via Ollama — no cloud API required.

---

## ROS2 nodes overview

**`spinnaker_camera_node`** (C++) — Publishes raw IR image stream from FLIR Firefly S USB3 at 30 Hz.

**`camera_mediapipe_node`** — Runs MediaPipe (468 landmarks) on IR frames, computes EAR and MAR, publishes `/ear_mar`.

**`carla_manual_control`** — Connects to CARLA 0.9.16, spawns ego vehicle, handles manual driving via Logitech G29.

**`carla_steering_metrics_node`** — Computes steering entropy, reversal rate, and SDLP over 60-second windows at 100 Hz.

**`camera_ml_node`** — Buffers 60 s of EAR/MAR, computes PERCLOS/blink features, runs 3-class Random Forest, publishes `/camera_predictions` with class probabilities.

**`carla_ml_node`** — Buffers 60 s of steering metrics, runs 3-class Random Forest, publishes `/carla_predictions`.

**`hybrid_vlm_node`** — On ambiguity flag, sends 8 IR frames (2 s at 4 Hz) to Qwen2.5-VL via Ollama. Returns structured JSON (occlusion, eye/mouth state, behaviours). Runs asynchronously — does not block the main pipeline.

**`integrated_llm_node`** — Every 60 s, fuses ML outputs + VLM context via LLaMA 3.1 (Ollama). Outputs final state, intervention action, and reasoning trace. Includes partial-trigger and watchdog fallback (85 s timeout).

**`drowsiness_alert_dispatcher`** — Routes LLM decisions to intervention nodes.

**`speaker_node`**, **`fan_node`**, **`steering_node`** — Intended hardware actuators (implemented but not validated on physical hardware).

**`data_logger_node`** — Logs all metrics, predictions, VLM/LLM outputs, and reasoning traces to disk.

**`labelling_tool`** *(optional)* — Flask UI for manual labelling. Not used in the main evaluation.

---

## Hardware and software

| Component | Spec |
|---|---|
| IR Camera | FLIR Firefly S USB3 (IMX273), 30 Hz |
| Steering Wheel | Logitech G29, 100 Hz |
| Simulator | CARLA 0.9.16 |
| Landmark Detection | MediaPipe 0.10.9 |
| Middleware | ROS2 Humble |
| GPU | NVIDIA RTX 5090 (24 GB VRAM) |
| OS | Ubuntu 24.04 |
| VLM | Qwen2.5-VL 3B (8-bit) via Ollama |
| LLM | LLaMA 3.1 8B via Ollama |
| ML classifiers | scikit-learn 1.3.2, RandomForest (100 estimators) |

---

## Local inference setup

### Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Pull models

```bash
ollama pull llama3.1:8b
ollama pull qwen2.5vl:3b
```

### Start the server

```bash
ollama serve
# Listens on http://localhost:11434 by default
```

### Verify

```bash
ollama list
# Both llama3.1:8b and qwen2.5vl:3b should appear before launching
```

---

## Prerequisites

### CARLA Simulator

1. Install CARLA 0.9.16 from the [official website](https://carla.org) — follow the [installation guide](https://carla.readthedocs.io/en/0.9.16/start_quickstart/)
2. Launch CARLA with ROS2 support:

```bash
./CarlaUE4.sh --ros2
```

### Logitech G29 steering wheel (udev rules)

```bash
sudo nano /etc/udev/rules.d/99-logitech-g29.rules
```

Paste the following:

```
KERNEL=="hidraw*", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="c24f", MODE="0666", SYMLINK+="logitech_g29"
```

Reload and trigger:

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### Custom ROS2 messages

**Drowsiness Detection Messages** (`src/drowsiness_detection_msg/msg/`):

| Message | Description | Key fields |
|---|---|---|
| `DrowsinessMetricsData.msg` | Complete drowsiness metrics | PERCLOS, blink rate, yawn frequency |
| `EarMarValue.msg` | Facial measurements | EAR, MAR |
| `LanePosition.msg` | Vehicle positioning | Lane deviation, heading angle |
| `Vibration.msg` | Haptic feedback control | Intensity, duration, pattern |

**CARLA Interface Messages** (`src/ros_carla_msgs/msg/`):

| Message | Description |
|---|---|
| `CarlaEgoVehicleControl.msg` | Vehicle control interface for CARLA simulator |

---

## Build and run

```bash
cd Ambiguity-Aware-Driver-State-Recognition
colcon build --symlink-install
source install/setup.bash
```

Start CARLA:

```bash
./CarlaUE4.sh --ros2
```

Run the full system:

```bash
ros2 launch drowsiness_detection_pkg complete_system_launch.py driver_id:=session_01
```

Optional labelling UI:

```bash
ros2 launch drowsiness_detection_pkg labelling_tool_launch.py
# Access via http://<linux-ip>:5000
```

---

## Credits

`camera_mediapipe_node` and parts of `carla_manual_control` are based on the course lab repository shared at the start of the project. All other components were designed and implemented by the project team.
