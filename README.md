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


### What do the header values for the stats mean?
* AB stands for the number of official at bats
* R stands for the amount of runs scored by that individual
* H stands for a hit, where a batter at least reaches first base safely
* 2B stands for a double, the hitter reaches second base safely
* 3B stands for a triple, the hitter reached third base safely
* HR stands for home run, where a hitter reaches home safely
* RBI stands for run batted in, when a teammate on base reaches home safely caused by the hitter.
* SB stands for a stolen base, when a runner advances a base without assistance of the hitter
* BB stands for base on balls or a walk.  The hitter is given safe pass to first base for a walk.
* SO stands for strike out
* BA stands for batting average and is calculated by H/AB and is represented as a decimal number
* OBP stands for on base percentage and is represented by a decimal.  It is calculated by the number of times a hitter reaches at least first base safely divided by the number of plate appearances.
* Pos stands for position
