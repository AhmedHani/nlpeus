#!/bin/bash

# running style recognition in floydhub - CPU machine
floyd run --env pytorch-1.0 --follow "python3 -m style_recognition.research.local_driver"