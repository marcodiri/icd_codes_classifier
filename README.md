# ICD Codes Classifier
String classifier based on [marcodiri/voted_perceptron][https://github.com/marcodiri/voted_perceptron].

This Python3 project has been built to classify strings into ICD-10 Codes using a SVM with a string kernel
[marcodiri/mismatch_string_kernel][https://github.com/marcodiri/mismatch_string_kernel].
The project is currently tied to that kernel but can easily be refactored to support different
kernels.

[https://github.com/marcodiri/voted_perceptron]: https://github.com/marcodiri/voted_perceptron

[https://github.com/marcodiri/mismatch_string_kernel]: https://github.com/marcodiri/mismatch_string_kernel

## Usage
Install the required dependencies with

    pip install -r requirements.txt

Modify the `configs.py` file to match your dataset format and change other self explanatory settings.

### Command line parameters:
You can train a MulticlassClassifier with the following parameters:

    -p, --process_count [int]: number of concurrent processes to use to 
        train/test on a dataset
        default to os.cpu_count()
    train
    -e, --epochs [1 to 10]: number of times the training set will be repeated.
        If greater than 1, the dataset will be repeated but with
        shuffled examples, therefore subsequent training with the same parameters
        will result in different quality classifiers
        default to 1
    -s, --seed: if epochs>1, the seeds with which the examples will be shuffled.
        After every training you can find these seeds in the logs/events.log file
        if you want to save them so that training will be repeatable
    -k [1 to 10]: the lenght of the k-mers for the mismatch string kernel
    -m [1 to 10 and less than k]: maximum number of mismatches for the mismatch string kernel

Three files will be generated in the folder specified in config.py (default to ./save):

`mismatch_vectors_k_m.pk`, `kernel_matrix_k_m.pk` and `MismatchKernel_k_m_epochs_errors.pk`.
For every subsequent training with same `k` and `m` the corresponding mismatch vectors and 
kernel matrix file will be reused saving a lot of computational time.

Example to train for 3 epochs with seeds 42 and 46, with 4 running processes:

    python runner.py -p 4 train -e 3 -s 42 46 -k 3 -m 1
    
The dataset will be first shuffled with seed 42 and then with seed 46, and these two shuffles
will be attached to the original dataset to form the 3 epochs.

You can classify a string with a saved training file with the following parameters:

    predict
    -i, --input: the input string
    -f: the training file path or the keyword 'all' to predict with every classifier
        in the save folder

Example to classify the string "fever" with every saved classifier:

    python runner.py predict -i "fever" -f all