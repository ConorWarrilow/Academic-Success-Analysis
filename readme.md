<img width="435" alt="feature-importance-summary-table2" src="https://github.com/ConorWarrilow/Academic-Success-Analysis/assets/152389538/265631d8-f2c5-46bb-874c-77fd00f2ed83"># $${\color{#00D8DB}\text{Machine Learning For Early Detection of} \space \color{#00D8DB}\text{at Risk} \space \color{#00D8DB}\text{Students}}$$


The transition to higher education can be challenging for many students, and many factors can influence academic performance. As a result, some students may struggle to keep up with the demands of their coursework, leading to underperformance or dropping out. Below we will explore the usage of machine learning for early detection of students on the path to dropping out, allowing us to catch problems early and provide students with intervention strategies on a person-to-person basis.



# $${\color{#E0581C}\text{Results Summary}}$$
This dataset was part of a kaggle competition running through 1/6/2024 to 1/7/2024.

<details>
  <summary>$${\color{#72B3A2}\text{What is kaggle?}}$$</summary>
	<br/>
Kaggle is a global online platform designed for data scientists and machine learning practitioners, where individuals or teams compete to solve complex data problems. Kaggle competitions involve leaderboards, with large prices (often over $10,000) rewarded to the top participant/team.
</details>

My final model obtained an accuracy score of 83.82%, just 0.31% shy of the leaderboard highscore, putting me in the top 8% of the leaderboard out of over 2500 participants. My main focus however was on EDA and optimizing the recall score for the dropout class, which can be seen below. By altering class weights, recall scores of up to 90% are realistically possible with a small sacrifice to precison. 

To conclude, the above results indicate the potential for machine learning to be a highly effective strategy in the early detecting of students at risk of dropping out.





# $${\color{#00D8DB}\text{Goals}}$$

The main goals are to:
1. Explore and assess the features that may indicate if a student is on the path to dropping out, graduating, or remaining enrolled by the end of the normal duration of their course.
2. Construct a machine learning model aimed towards the early detection of students displaying the signs associated with dropping their course.
3. Analyze the pitfalls of a purely machine learning based approach, particuarly one that optimizes for accuracy, and determine what improvements could be made to increase our rate of early detection.

   



# $${\color{#00D8DB}\text{Table of Contents}}$$



**1. EDA**

- 1.1 The Basics
- 1.2 Correlations
- 1.3 Kde and Cumulative Kde Plots + Analysis

**2. Feature Transformations**
- 2.1 Transformation Types
- 2.2 Data Types
- 2.3 Normalization and Standardization

**3. Feature Importance and Selection**
  - 3.1 Impurity Based Feature Importance
  - 3.2 Permutation Based Feature Importance
  - 3.3 Other Methods of Evaluation

**4. Feature Engineering**

**5. Machine Learning**
- 5.1 Building a Pipeline
- 5.2 Choosing the Right Model
- 5.3 Hyperparameter Tuning
- 5.4 Ensemble Models
- 5.5 Ensemble Stacking

 **6. Results**


 **7. Further Analysis and discussion**
- 7.1 t-SNE
- 7.2 Changing our Metric
- Additional Features



# $${\color{#00D8DB}\text{The Dataset}}$$

The data set provided contains a total of 36 features relating to academic performance, age, gender, attendance style, martial status, course type, parent qualifications/occupations, and various social economic metrics.

A full list of [features and their definitons can be found here.](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

The situation is a multiclass classification, with 3 classes to predict.

**Target value Labels:**

0 - Dropout
1 - Graduate
2 - Enrolled


# $${\color{#00D8DB}\text{1. Exploratory Data Analysis}}$$

## $${\color{#00A5A8}\text{}}$$

### Initial EDA
- No missing/null values
- 8 binary/boolean columns
- 7 float columns
- 21 integer columns
- 0 categorical columns

### Data Types
Several Integer columns appear appropriate for one-hot-encoding.

### Target Class Imbalance
The target classes are imbalanced, although it's unlikely to be problematic. Various Undersampling/oversampling techniques such as SMOTE could be explored.

<details>
  <summary>$${\color{#72B3A2}\text{View Class Imbalance}}$$</summary>
<div align="center">
	<img width = "600" src="https://github.com/ConorWarrilow/Academic-Success-Analysis/assets/152389538/ca33cee2-824c-412c-bd22-0b3270f4d39a">
</div>
</details>





## $${\color{#00A5A8}\text{1.2 Correlations}}$$

### Correlation Matrix

Next, we'll take a look at the (pearson) feature correlations:

<details>
  <summary>$${\color{#72B3A2}\text{View Correlation Matrix}}$$</summary>
<div align="center">
	<img width = "600" src="https://github.com/ConorWarrilow/Academic-Success-Analysis/assets/152389538/a198ad4b-735e-40ea-92c9-6aff151d9cd2">
</div>
</details>

A few features are highly correlated, particuarly those involving curricular units. As we'll likelyt be using tree based models, this isn't necessarily a problem when it comes to model accuracy, however we still need to be mindful about these features when assessing feature importance. It would be wise to either drop certain features, aggregate highly correlated features, or to simply take these correlated features into account.


### Dendogram
We can plot a dendogram to better visualize the correlations (again, pearson in this case) between features:
<details>
  <summary>$${\color{#72B3A2}\text{View Dendogram}}$$</summary>
<div align="center">
	<img width = "600" src="https://github.com/ConorWarrilow/Academic-Success-Analysis/assets/152389538/c67b713f-236e-43ef-b79a-88cb45df28c9">
</div>
</details>

While the dendogram doesn't tell us anything new, its a useful way to visualize different 'batches' or 'groups' of features. For example we can see how the four features martial status, daytime/evening attendance, application mode, and age at enrollment have been grouped together, which isn't as visually obvious in the correlation Matrix.




### High Correlations
Finally, we can plot a table containing any features with correlation values over a certain threshold (0.8 in this case). We'll keep an eye on these later when we assess feature importance.
<details>
  <summary>$${\color{#72B3A2}\text{View High Correlations Table}}$$</summary>
<div align="center">
	<img width = "600" src="https://github.com/ConorWarrilow/Academic-Success-Analysis/assets/152389538/fac02335-f652-438b-b8b8-6627c16db881">
</div>
</details>





## $${\color{#00A5A8}\text{1.3 Kde and Cumulative Kde plots and Analysis}}$$
### Kde Plots
Next We'll look at the kde plots for our numeric features.

<details>
  <summary>$${\color{#72B3A2}\text{View Kde Plots}}$$</summary>
<div align="center">
	<img width = "600" src="https://github.com/ConorWarrilow/Academic-Success-Analysis/assets/152389538/a894ab22-bb76-4252-8916-9ee1a7a926eb">
</div>
</details>

Looking at the kde plots, two features stand out the most: Curricular units 1st sem (approved), and curricular units 2nd sem (approved). These features show distinct distributions across the target classes with very little overlap amongst the dropout and graduate classes. Looking at the dropout class, the majority of values are zero in both features. In contrast, most values range between 5 and 7 for the graduate class, while the enrolled class spans a somewhat gaussian distribution between the other classes, with its meaning falling much closer to the graduate class.

Curricular units 1st/2nd sem (grade) features also sow string separation between class distributions, particuarly between the dropout and graudate/enrolled classes.

Age at enrollment shows older students as being much more likely to dropout, with the dropout class showing a much more prominant right skew compared to the other classes.




<details>
  <summary>$${\color{#72B3A2}\text{View Cumulative Kde Plots}}$$</summary>
<div align="center">
	<img width = "600" src="https://github.com/ConorWarrilow/Academic-Success-Analysis/assets/152389538/3cd02fdd-a770-4fad-a62e-9b3a061587ba">
</div>
</details>





# $${\color{#00D8DB}\text{2. Feature Transformations}}$$








## $${\color{#00A5A8}\text{2.1 Transformation Types}}$$

## $${\color{#00A5A8}\text{2.2 Handling Categorical Data}}$$

## $${\color{#00A5A8}\text{2.3 Normalization and Standardization}}$$




# $${\color{#00D8DB}\text{3. Feature Importance Analysis}}$$





## $${\color{#00A5A8}\text{3.1 Impurity-Based Feature Importance}}$$


<details>
  <summary>$${\color{#72B3A2}\text{View Impurity-Based Results}}$$</summary>
<div align="center">
	<img width = "600" src="https://github.com/ConorWarrilow/Academic-Success-Analysis/assets/152389538/38b8f5fe-a77b-4a3b-95f2-9826be62e38f">
</div>

 <div align="center">
	<img width = "600" src="https://github.com/ConorWarrilow/Academic-Success-Analysis/assets/152389538/d5d706cf-ab81-43e7-83bc-eff12f2163cd">
</div>
</details>





## $${\color{#00A5A8}\text{3.2 Permutation-Based Feature Importance}}$$


<details>
  <summary>$${\color{#72B3A2}\text{View Permutation-Based Results}}$$</summary>
<div align="center">
	<img width = "600" src="https://github.com/ConorWarrilow/Academic-Success-Analysis/assets/152389538/fe550290-0092-4e2e-8fac-15acb9a5f2bc">
</div>
</details>



<details>
  <summary>$${\color{#72B3A2}\text{View Feature Importance Summary Table}}$$</summary>
<div align="center">
	<img width = "600" src="https://github.com/ConorWarrilow/Academic-Success-Analysis/assets/152389538/9acb8d6e-dd6f-47d7-aee9-8b997414a123">
</div>

<div align="center">
	<img width = "600" src="https://github.com/ConorWarrilow/Academic-Success-Analysis/assets/152389538/5db741a9-1c24-427b-8b44-ec842e95aee6">
</div>


 
</details>




## $${\color{#00A5A8}\text{3.3 Altermative Methods of Feature Importance}}$$



# $${\color{#00D8DB}\text{4. Feature Engineering}}$$



# $${\color{#00D8DB}\text{5. Machine Learning}}$$









## $${\color{#00A5A8}\text{5.1 Building a Pipeline}}$$







## $${\color{#00A5A8}\text{5.2 Choosing the Right Model}}$$

## $${\color{#00A5A8}\text{5.3 Hyperparameter Tuning}}$$

## $${\color{#00A5A8}\text{5.4 Ensemble Models}}$$

## $${\color{#00A5A8}\text{5.4 Stacked Ensembles}}$$


# $${\color{#00D8DB}\text{6. Results}}$$



# $${\color{#00D8DB}\text{7. Further Analysis}}$$


## $${\color{#00A5A8}\text{7.1 t-SNE}}$$

## $${\color{#00A5A8}\text{7.2 Changing our Metric}}$$

## $${\color{#00A5A8}\text{7.2 Additional Features}}$$



