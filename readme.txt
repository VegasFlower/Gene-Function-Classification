The code can be run in following environment:

operating system:       macOS BigSur 11.6
Programming language:   Python 3.7
Recommended IDE:        Pycharm 2019.1.3 Community Edition (package auto-installation)

packages required:
    'pandas'
    'csv'
    'scikit-learn'

Automatic package installation:
Downloading Pycharm from its official website, put the 'classifier.py' into
the same folder where '.csv' ('Ecoli.csv', 'Ecoli_test.csv') located.
Open the classifier.py in Pycharm, right click the warning on uninstalled
packages and let Pycharm install them automatically.

Produce report:
To generate the result report, directly run the classifier.py file would
be enough. The main function has aggregate all tasks required including
cross validation on training data and predicting on testing data. After
finishing both tasks, the program will produce the final report within
the same folder.

cv evaluation on training data:
    accuracy_score   is around 96
    f1_score         is around 82

