1. Evaluation
・Usage:
  - In the command line, execute
    $ python IOU.py -g path/to/ground/truth -p path/to/prediction
    and you can get the IOU for the ground truth and the prediction.
・Environment:
  - python>=3.6
・Notes
  - The format of the ground truth file and the prediction file:
    ・File Name: ***.json (*** = whatever name you like(e.g. ground_truth))
    ・Description:
      - image_file_0:
          - category_1:
              - y_coordinate_0: [[x1, x2],...]
              ...
          - category_2:
              - y_coordinate_0: [[x1, x2],...]
              ...
          ...
      - image_file_1:
          - category_1:
              - y_coordinate_0: [[x1, x2],...]
              ...
          - category_2:
              - y_coordinate_0: [[x1, x2],...]
              ...
          ...
      ...
    ・For each image file, the number of categories may differ.
      - e.g.) image_file_0: ["car", "pedestrian"], image_file_1: ["truck"]
    ・Please refer to "figure.pdf" for more information about "y_coordinate_0", "x1", "x2", etc.
    ・Please also refer to "sample_submit.json".

2. Making the Submission File for the Predictions in png
・Usage
  - In the command line, execute
    $ python make_submit.py -p path/to/annotations
    and you can convert all png files in path/to/annotations(predictions) to one json file for the submission.
・Environment:
  - python>=3.6
  - Pillow>=5.2.0
  - numpy>=1.15.4
