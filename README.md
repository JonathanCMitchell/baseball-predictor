# Baseball predictor

#### Hello!  This is a repo that uses tensorflow's learn API.  We will use the tensorflow learn library to predict the following...
 * Predict wether someone will hit more or less than 55 RBIs
 * Predict if someone has more or less than 17 Stolen Bases(SB)
 * Predict a players position

### Build
* Fork the project, then clone it
* To run this project you are going to need [python](http://docs.python-guide.org/en/latest/starting/installation/)
* If you have python you probably already have pip, which is the package management system for python
* Next you will need to install Tensorflow. [Tensorflow](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html)
* Make sure you install the right binary file for your version of python, and your operating system
* To get the required packages for this project run `pip install -r requirements.txt`, if you have python3 use `pip3` instead

### Notes about the CSV Data
* Only included batters with 100 At Bats or more, except pitcher who need a minimum of 15 At Bats
* Using 2015 and 2016 data from national league only
* Position `1: Pitcher, 2: Catcher, 3: First Base, 4: Second Base, 5: Third Base, 6: Shortstop, 7: Outfield`
