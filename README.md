# ![Rain Prediction Model](/rmpic/rainpic.jpg)
<table>
<tr>
<td>
This is a predictive model that can predict, based on 9 years of collected data, if it is going to rain or not and roughly how much rain to expect. This model was built as part of a my final assignment for the IBM machine learning certificate. 
</td>
</tr>
</table>


## Data
## [Data](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/Weather_Data.csv)
The data in this model was sourced from the skills network cloud provided by IBM.


## Main Concept
Based on independent variables from historical weather data such as:
- Date
- Location
- Minimum Temperature
- Maximum Temperature
- Amount of Rainfall
- Amount of Evaporation
- Amount of Sunshine
- Wind Gust Direction at 9am&3pm 
- Wind Gust Speed at 9am&3pm
- Humidity at 9am&3pm
- Atmospheric Pressure at 9am&3pm
- Cloud Cover at 9am&3pm
- Temperature Measured at 9am&3pm 
- If There Was Rain


Can we predict if there will be rain tomorrow?

## Use cases
This code can be used to apply these same methods on another similar data set in an aim to predict what a specific value might be based on a number of different independent variables.
I think its also important to note that in general it is pretty useful simply to be able to predict if it is going to rain tomorrow or not.
There are many online sources of free weather data that can be sourced and applied to this model for a more up to date prediction.


## [Usage](https://colab.research.google.com/) 
- Anyone could use google Colaboratory to use this code and modify it to their data needs.
### Development
Want to contribute? Great!

To fix a bug or enhance an existing module, follow these steps:

- Fork the repo
- Create a new branch (`git checkout -b improve-feature`)
- Make the appropriate changes in the files
- Add changes to reflect the changes made
- Commit your changes (`git commit -am 'Improve feature'`)
- Push to the branch (`git push origin improve-feature`)
- Create a Pull Request 

### Bug / Feature Request

If you find a bug in this code please let me know.

## Built with 

- [Scikit-Learn](https://scikit-learn.org/) - For their models and tools
- [Python](https://www.python.org/) - For its flexibility and abundant amount of resources
- [Numpy](https://numpy.org/) - For computation
- [Pandas](https://pandas.pydata.org/) - For organization, processing and data visualization
- [Matplotlib](https://matplotlib.org/) - For data visualization

## To-do
- Build an accompanying gui or webapp as to make this an actual application anyone can use.
- Create more data visualizations



