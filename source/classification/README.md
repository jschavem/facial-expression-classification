# Requirements

* Python 2.7+
* scikit-learn
* ruby (runner.rb only)

# Usage

The following command will run the python classification configured on multiple datasets:

    ruby runner.rb <output file> <perform_rfe?>

The output will be a CSV file with the chosen filename. Note that this is highly specific for the particular datasets
and their names in this thesis.

In the more general case, the following two scripts may be used and both provide help with passed the `-h` parameter:

Specific subject cross-validation:

    python cv_per_subject.py

Multiple subject cross-validation:

    python cv_subjects.py

