# Docker setup

This docker setup is tested on Windows 10.

make sure you are under this directory yourworkspace/Hunyuan3D-2.1/docker

Build docker image:

```
docker build -t hunyuan3d21:latest .
```

Run docker image at the first time:

```
docker run -it --name hy3d21 -p 7860:7860 --gpus all hunyuan3d21 python gradio_app.py --port 7860
```

After first time:
```
docker start hy3d21
docker exec -it hy3d21 python gradio_app.py --port 7860
```

Stop the container:
```
docker stop hy3d21
```

You can find the demo link showing in terminal, such as `http://0.0.0.0:7860`, then you cuold access `http://127.0.0.1:7860` from your host machine.

Some notes:
1. the total built time might take more than one hour.
2. the total size of the built image will be more than 70GB.