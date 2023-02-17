# Get started with Triton Nvidia

## Run Triton Nvidia Server

```bash
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:23.01-py3 bash

tritonserver --model-repository=/models
```

## Run Client




