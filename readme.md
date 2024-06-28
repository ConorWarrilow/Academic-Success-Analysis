# $${\color{#00D8DB}\text{Machine Learning For Early Detection of} \space \color{#00D8DB}\text{Students at} \space \color{#00D8DB}\text{Risk}}$$


The transition to higher education can be challenging for many students, and many factors can influence academic performance. As a result, some students may struggle to keep up with the demands of their coursework, leading to increased dropout rates and underperformance. The use of machine learning for Early detection of students on the path to dropping out can help catch problems early and allow universities to implement intervention strategies on a student-to-student basis.






# $${\color{#00D8DB}\text{Summary}}$$
This dataset was part of a kaggle competition running through 1/6/2024 to 1/7/2024.

<details>
  <summary>$${\color{#F2EDD8}\text{What is kaggle?}}$$</summary>
	<br/>
Kaggle is a global online platform designed for data scientists and machine learning practitioners, where individuals or teams compete to solve complex data problems. Kaggle competitions involve leaderboards, with large prices (often over $10,000) rewarded to the top participant/team.
</details>

My final model was able to obtain an accuracy score of 83.82%, just 0.31% shy of the (current) leaderboard highscore, putting me in the top 8% of the leaderboard out of over 2500 participants. My main focus however was on EDA and optimizing the recall score for the dropout class, which can be seen below.




# $${\color{#00D8DB}\text{Goals}}$$

1. To explore and assess the feature sthat may indicate whether a student is on the path to dropping out, gradiating, or remaining enrolled by the end of the normal duration of their course.
2. To construct a machine learning model aimed towards the early detection of students displaying the signs associated with dropping their course.
3. Analyze the pitfalls of a purely machine learning based approach and determine what improvements can be made to increase our rate of early detection

   



# $${\color{#00D8DB}\text{Table of Contents}}$$


The data set provided contains a total of 36 features relating to academic performance age gender attendant style Marshall status course type parent qualifications or occupations and various social economic metrics.

A full list of [features and their definitons can be found here.](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

The situation is a multiclass classification, with 3 classes to predict.

**Target value Labels:**

0 - Dropout
1 - Graduate
2 - Enrolled


# $${\color{#00D8DB}\text{1. Exploratory Data Analysis}}$$

## $${\color{#00A5A8}\text{1.1 The Basics}}$$

### Initial EDA
- No missing/null values
- 8 binary/boolean columns
- 7 float columns
- 21 integer columns
- 0 categorical columns

### Data Types
Several Integer columns appear appropriate for one-hot-encoding.

###Target Class Imbalance


<details>
  <summary>$${\color{#F2EDD8}\text{View Class Imbalance}}$$</summary>
<div align="center">
	<img width = "600" src="https://github.com/ConorWarrilow/Academic-Success-Analysis/assets/152389538/ca33cee2-824c-412c-bd22-0b3270f4d39a">
</div>
</details>





## $${\color{#00A5A8}\text{1.2 Correlations}}$$

### Correlation Matrix

<details>
  <summary>$${\color{#F2EDD8}\text{View Correlation Matrix}}$$</summary>
<div align="center">
	<img width = "600" src="">
</div>
</details>


### Dendogram
We can plot a dendogram to better visualize the correlations (again, pearson in this case) between features:
<details>
  <summary>$${\color{#F2EDD8}\text{View Dendogram}}$$</summary>
<div align="center">
	<img width = "600" src="">
</div>
</details>

While the dendogram doesn't tell us anything new, its a useful way to visualize different 'batches' or 'groups' of features. For example we can see how the four features martial status, daytime/evening attendance, application mode, and age at enrollment have been grouped together, which isn't as visually obvious in the correlation Matrix.



### High Correlations
Finally, we can plot a table containing any features with correlation values over a certain threshold (0.8 in this case). We'll keep an eye on these later when we assess feature importance.
<details>
  <summary>$${\color{#F2EDD8}\text{View High Correlations Table}}$$</summary>
<div align="center">
	<img width = "600" src="">
</div>
</details>




## $${\color{#00A5A8}\text{1.3 Kde and Cumulative Kde plots and Analysis}}$$
### Kde Plots
Next We'll look at the kde plots for our numeric features.

<details>
  <summary>$${\color{#F2EDD8}\text{View Kde Plots}}$$</summary>
<div align="center">
	<img width = "600" src="">
</div>
</details>

Looking at the kde plots, two features stand out the most: Curricular units 1st sem (approved), and curricular units 2nd sem (approved). These features show distinct distributions across the target classes with very little overlap amongst the dropout and graduate classes. Looking at the dropout class, the majority of values are zero in both features. In contrast, most values range between 5 and 7 for the graduate class, while the enrolled class spans a somewhat gaussian distribution between the other classes, with its meaning falling much closer to the graduate class.

Curricular units 1st/2nd sem (grade) features also sow string separation between class distributions, particuarly between the dropout and graudate/enrolled classes.

Age at enrollment shows older students as being much more likely to dropout, with the dropout class showing a much more prominant right skew compared to the other classes.



<details>
  <summary>$${\color{#F2EDD8}\text{View Cumulative Kde Plots}}$$</summary>
<div align="center">
	<img width = "600" src="">
</div>
</details>





# $${\color{#00D8DB}\text{2. Feature Transformations}}$$








## $${\color{#00A5A8}\text{2.1 Transformation Types}}$$

## $${\color{#00A5A8}\text{2.2 Handling Categorical Data}}$$

## $${\color{#00A5A8}\text{2.3 Normalization and Standardization}}$$




# $${\color{#00D8DB}\text{3. Feature Importance Analysis}}$$





## $${\color{#00A5A8}\text{3.1 Impurity-Based Feature Importance}}$$


<details>
  <summary>$${\color{#F2EDD8}\text{View Impurity-Based Results}}$$</summary>
<div align="center">
	<img width = "600" src="">
</div>
</details>


## $${\color{#00A5A8}\text{3.2 Permutation-Based Feature Importance}}$$


<details>
  <summary>$${\color{#F2EDD8}\text{View Permutation-Based Results}}$$</summary>
<div align="center">
	<img width = "600" src="">
</div>
</details>





<details>
  <summary>$${\color{#F2EDD8}\text{View Feature Importance Summary Table}}$$</summary>
<div align="center">
	<img width = "600" src="">
</div>
</details>



## $${\color{#00A5A8}\text{3.3 Altermative Methods of Feature Importance}}$$



# $${\color{#00D8DB}\text{4. Feature Engineering}}$$



# $${\color{#00D8DB}\text{5. Machine Learning}}$$









## $${\color{#00A5A8}\text{5.1 Building a Pipeline}}$$







## $${\color{#00A5A8}\text{5.2 Choosing the Right Model}}$$

## $${\color{#00A5A8}\text{5.3 Hyperparameter Tuning}}$$

## $${\color{#00A5A8}\text{5.4 Stacked Ensembles}}$$



# $${\color{#00D8DB}\text{6. Results}}$$



# $${\color{#00D8DB}\text{7. Further Analysis}}$$










