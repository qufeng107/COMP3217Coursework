# COMP3217Coursework

Detection of Manipulated Pricing in Smart Energy CPS Scheduling

The purpose of this coursework is to understand the linear programming based energy scheduling for smart home cyber-physical system, understand the interdependence between the pricing information and the energy load scheduling, develop detection techniques for pricing attacks, and get familiar with some cyber-physical system security programming skills.



## Install

Download the files from https://github.com/qufeng107/COMP3217Coursework.git. 

`setup.py` can be found in the root directory of the project. Run (python setup.py install/) from the directory where (setup.py/) is located.



## Run detection

(model.py/) can be found in the root directory of the project.
First, put (TrainingData.txt/) and (TestingData.txt/) in the directory where (model.py/) is located.
Then, Run (python model.py/). 
The script will take training data from (TrainingData.txt/) for fitting model, calculate labels for testing data in (TestingData.txt/) and store the results into (TestingResults.txt/).



## Run scheduling

(schedule.py/) can be found in the root directory of the project.
First, put (COMP3217CW2Input.xlsx/) and (TestingResults.txt/) in the directory where (schedule.py/) is located.
Then, Run (python schedule.py/).
The script will compute the linear programming based energy scheduling solution for users in (COMP3217CW2Input.xlsx/), according to abnormal predictive guideline price curve in (TestingResults.txt/). And energy scheduling solution will be plotted on bar charts and stored in the dictionary (./charts_for_abnormal_prices).
