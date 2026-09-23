docker run --rm -it --gpus all --shm-size=8g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e USER="$(id -un)" \
  -e LOGNAME="$(id -un)" \
  -v "$(pwd):/workspace" \
  credit
  