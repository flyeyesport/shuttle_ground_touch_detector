# shuttle_ground_touch_detector

to train run:
```
python train_3_3.py -t <path_to_training_dataset> -v <path_to_validation_dataset>> -r <path_to_results_directory> -f 8 -mf 2 -b 50 -s <path_to_initial_weights> -sp -ma 127.0.0.1 -mp 8888
```

to test speed run:
```
./generate.sh
```
from speed_test folder.
