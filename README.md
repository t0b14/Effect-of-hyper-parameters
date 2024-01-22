## Introduction

Project code for the RNN.

## Installation

To install all the dependencies run:

```
pip install -r requirements.txt
```

## Usage

To train and evaluate the models, run the following commands in the terminal:

1. To train the desired model update the config file rnn.yaml. For additional changes, the model itself is in "/src/network/rnn.py". It will run 3 total runs which can be modified in runner.py file.
```
python main.py --config rnn.yaml
```
2. After training the model please make sure the saved model exists in "/io/output/rnn1/trained_models/".
The plotting is segmented into 4 parts (as in https://gitlab.com/neuroinf/operativeDimensions), one can comment out the unwanted parts by commenting the call to the respective function in the get_op_dim.py file. Note part 2 is required to generate the operative dimensions.
To get the operative dimensions run:
```
cd .\src\operative_dimensions\
python get_op_dim.py --config rnn.yaml
```
3. To visualize the operative dimensions of the model run
```
python plot_global_operative_dimensions.py --config rnn.yaml
```
4. For completeness sake, to visualize the mean and standard deviation of different runs run:
```
python plot_combined_global_oper_dim.py --config rnn.yaml
```

