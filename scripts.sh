#!/bin/bash

# local running - CPU machine
python3 -m projects.style_recognition.research.local_driver --no-cuda
# local running - GPU machine
python3 -m projects.style_recognition.research.local_driver
# running style recognition in floydhub - CPU machine
floyd run --env pytorch-1.0 --follow "python3 -m projects.style_recognition.research.local_driver --no-cuda"
# running style recognition in floydhub - GPU machine
floyd run --gpu --env pytorch-1.0 --follow "python3 -m projects.style_recognition.research.local_driver"

# running style recognition in floydhub - GPU machine - supported data
floyd run --gpu --env pytorch-1.0 --data ahani/datasets/style_recognition_dataset/1:data --follow "python3 -m projects.style_recognition.research.spooky_authors"