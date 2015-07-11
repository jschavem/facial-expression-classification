# README

This project contains the processing pipeline to go from Noldus FaceReader 5 log exports, to binary emotion classifications.
For this an aggregation step is performed, turning the log files for one watched video into a row in a CSV file.

For this the project consists of two major parts:

1. `aggregation`
2. `classification`

# General method

1. Load video recording of user into the Noldus Facereader software
2. Obtain action unit logs from FaceReader
3. Rename log files to match some identifier for that particular video to allow for matching the video and ground truth.
4. Load logs per session
5. Perform some aggregation method
6. Possibly normalize the data
7. Determine the optimal number of features using RFECV
8. Perform RFE
9. Cross validate SVM trained on the features selected by RFE.
10. Output statistics

## Aggregation

### Requirements

* Ruby 1.9+
* `rubygems`
* Various dependencies; `cd aggregation; gem install bundler && bundle install`


## Classification

### Requirements

* Python 2.7+
* `scikit-learn`