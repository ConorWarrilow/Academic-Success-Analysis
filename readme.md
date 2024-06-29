# $${\color{#00D8DB}\text{Machine Learning For Early Detection of} \space \color{#00D8DB}\text{at Risk} \space \color{#00D8DB}\text{Students}}$$


The transition to higher education can be challenging for many students, and many factors can influence academic performance. As a result, some students may struggle to keep up with the demands of their coursework, leading to underperformance or dropping out. Below we will explore the usage of machine learning for early detection of students on the path to dropping out, allowing us to catch problems early and provide students with intervention strategies on a person-to-person basis.



# $${\color{#E0581C}\text{Results Summary}}$$
This dataset was part of a kaggle competition running through 1/6/2024 to 1/7/2024.

<details>
  <summary>$${\color{#72B3A2}\text{What is kaggle?}}$$</summary>
	<br/>
Kaggle is a global online platform designed for data scientists and machine learning practitioners, where individuals or teams compete to solve complex data problems. Kaggle competitions involve leaderboards, with large prices (often over $10,000) rewarded to the top participants/teams.
</details>

My final model obtained an accuracy score of 83.82%, just 0.31% shy of the leaderboard highscore, putting me in the top 8% of the leaderboard out of over 2500 participants. My main focus however was on EDA and optimizing the recall score for the dropout class, which can be seen below. By altering class weights, recall scores of up to 90% are realistically possible with a small sacrifice to precison. 

To conclude, the above results indicate the potential for machine learning to be a highly effective strategy for the early detection of students at risk of dropping out.





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
- 1.4 Binary Feature Analysis
- 1.5 Other EDA

**2. Feature Transformations**
- 2.1 Transformation Types
- 2.2 Data Types
- 2.3 Normalization and Standardization

**3. Feature Importance and Selection**
  - 3.1 Impurity Based Feature Importance
  - 3.2 Permutation Based Feature Importance
  - 3.3 Feature Importance Summary
  - 3.4 Other Methods of Evaluation

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
The target classes are imbalanced, as can be seen in the below plot.

<details>
  <summary>$${\color{#72B3A2}\text{View Class Imbalance}}$$</summary>
<div align="center">
	<img width = "600" src="https://github.com/ConorWarrilow/Academic-Success-Analysis/assets/152389538/ca33cee2-824c-412c-bd22-0b3270f4d39a">
</div>
</details>


Various Undersampling/oversampling techniques such as SMOTE could be explored, however the imbalance isn't too extreme and likely won't be a problem.


## $${\color{#00A5A8}\text{1.2 Correlations}}$$

### Correlation Matrix

Next, we'll take a look at the (pearson) feature correlations:

<details>
  <summary>$${\color{#72B3A2}\text{View Correlation Matrix}}$$</summary>
<div align="center">
	<img width = "1200" src="https://github.com/ConorWarrilow/Academic-Success-Analysis/assets/152389538/a198ad4b-735e-40ea-92c9-6aff151d9cd2">
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
	<img width = "1200" src="https://github.com/ConorWarrilow/Academic-Success-Analysis/assets/152389538/a894ab22-bb76-4252-8916-9ee1a7a926eb">
</div>
</details>

Looking at the kde plots, two features stand out the most: Curricular units 1st sem (approved), and curricular units 2nd sem (approved). These features show distinct distributions across the target classes with very little overlap amongst the dropout and graduate classes. Looking at the dropout class, the majority of values are zero in both features. In contrast, most values range between 5 and 7 for the graduate class, while the enrolled class spans a somewhat gaussian distribution between the other classes, with its meaning falling much closer to the graduate class.

Curricular units 1st/2nd sem (grade) features also sow string separation between class distributions, particuarly between the dropout and graudate/enrolled classes.

Age at enrollment shows older students as being much more likely to dropout, with the dropout class showing a much more prominant right skew compared to the other classes.



### Cumulative Kde Plots
We can also take a look at the cumulative kde plots.
<details>
  <summary>$${\color{#72B3A2}\text{View Cumulative Kde Plots}}$$</summary>
<div align="center">
	<img width = "1200" src="https://github.com/ConorWarrilow/Academic-Success-Analysis/assets/152389538/3cd02fdd-a770-4fad-a62e-9b3a061587ba">
</div>
</details>

To be honest, the cumulative Kde plots aren't particuarly useful, as most of the insights can be quite easily pulled from the Kde plots. 


## $${\color{#00A5A8}\text{1.4 Binary Feature Analysis}}$$
To Wrap up our EDA We'll take a look at our binary Features using sunburst plots. Unfortunately, they're no longer interactive as images, which is half the value (and 100% of the fun) of a sunburst plot, but we'll make do. 

<details>
  <summary>$${\color{#72B3A2}\text{View Sunburst Plots}}$$</summary>
<div align="center">
	<img width = "1200" src="https://github.com/ConorWarrilow/Academic-Success-Analysis/assets/152389538/26e5e7fa-5e61-4351-aacd-3deba90bae6d">
</div>
</details>

There seem to be many valuable insights from the binary features. The most notable include:

- 82% of scholarship holders graduate, while only 36% of those without scholarships graduate
- 94% of students without up to date fees drop out, while only 26% of students with up to date fees drop out
- Gender, debtor, and attendance style also contain large differences in their distribution of class values

## $${\color{#00A5A8}\text{1.5 Other EDA}}$$
While we covered a good few topics, we barely scratched the surface of EDA.

Further exploration such as bivariate analysis, detecting + handling outliers, and feature interactions could also be explored to better understand the data.






# $${\color{#00D8DB}\text{2. Feature Transformations}}$$
Before we can begin the machine learning stage, our features need to undergo transformations. Feature transformations are a crucial preprocessing step. For our specific case, the transformations we'll cover are:
- Applying log, sqrt, box-cox, yeojohnson, etc transformations to numerical features to reduce skew, address heteroskedasticity, and improve model performance.
- Categorical feature encoding
- Standardization and normalization



## $${\color{#00A5A8}\text{2.1 Transformation Types}}$$
Before applying log, sqrt, and other various transformations to our numerical features, we'll first take a look at some probability plots.


<details>
  <summary>$${\color{#72B3A2}\text{View Probability Plots}}$$</summary>
<div align="center">
	<img width = "900" src="https://github.com/ConorWarrilow/Academic-Success-Analysis/assets/152389538/e4efe42c-28d3-45fd-8941-243496bfd260">
</div>
</details>

Some features such as previous qualification (grade) and admission grade already fit well, with r^2 = 0.98 for both. This isn't the case for many other features however and they may benefit from transformations. Another thing to note is the abundance of zero-inflated features. We can create a binary indicator for each of these which may assist our model.

For the sake of time, a quick script was written to assess a range of transformations on selected numeric features, which excluded both boolean features and features to be later encoded.

**Note:** Some transformations require non-negative or strictly positive input values. A filter was applied to ensure transformations were only used where approprate. While each row appears to contain a transformation value, some of these values are simply a placeholder, where no transformation has been applied.

The results were as follows:

<details>
  <summary>$${\color{#72B3A2}\text{View Transformation Results}}$$</summary>
<div align="center">
	<img width = "900" src="https://github.com/ConorWarrilow/Academic-Success-Analysis/assets/152389538/ee0e2827-5b0e-4cd6-90e0-b0e2cc5c5ab9">
</div>
</details>


A few transformations are shown to provide marginal improvements. Further analysis should be performed before deciding to apply a transformation, particuarly in real world applications.

## $${\color{#00A5A8}\text{2.2 Handling Categorical Data}}$$
Most machine learning models don't play nicely with categorical data, with newer algorithms such as Catboost being the exception. While our data doesn't contain any categorical data types, it does contain categories represented as integers. Though not always the case, it's generally beneficial to encode such features. Numerous methods of encoding exist, and like most things in machine learning, experimentation is the only way to find the best representation.

As an example, rather than representing mother's/father's qualification as arbitrarily ordered numbers, we could instead encode them ordinally, with smaller numbers for lower levels of qualification, and larger for higher. We could also consider binning the qualification levels, as a parent having a masters as opposed to a phd is quite unlikely to reduce our model's predictive ability, and may even help in preventing overfitting.

On the other hand, a feature such as course is unlikely to have a specific order, and may benefit from simple one-hot-encoding.



## $${\color{#00A5A8}\text{2.3 Normalization and Standardization}}$$
lastly, we'll apply normalization and standardization, generally referred to as 'scaling'. The most common type of scaling is to center the mean of each feature at 0 (standardization), and to transform each feature to have a standard deviation of 1 (normalization). Again, this stage requires experimentation as different datasets will perform better with different scalers.



# $${\color{#00D8DB}\text{3. Feature Importance Analysis}}$$
Moving on to feature importance analysis, which is quite often misunderstood and misused. There are numerous methods of measuring feature importance resulting in slight variations in definition, but in general, feature importance simply aims to measure the extent to which a feature contributes towards a model's predictive ability. Obtaining feature importances can help us to detect biases within the data, as well as problems within our model. Feature importance is also used to understand the underlying relationships between features and our target, helping us to properly understand how the model works, and allowing us to use our results to make business decisions. It can also give an idea as to what features we may need to focus on during feature engineering.




## $${\color{#00A5A8}\text{3.1 Impurity-Based Feature Importance}}$$
Impurity-based feature importance is an embedded method found in tree-based models. Features capable of producing the pureset splits, hence contributing to a cleaning partitioning of the data, will be considered as the features wit hthe highest importance.

To obtain robust results, the following procedure was used:

1. Four tree-based models were selected (XGBoost, LightGBM, Catboost, and random forest for good measure)
2. Each model was trained using stratified cross validation and tested on the out of fold samples.
3. The feature importances were extracted from each model
4. The above steps were repeated for a total of ten iterations using different seeds and the results were averaged.

**Impurity Results:**

<details>
  <summary>$${\color{#72B3A2}\text{View Impurity-Based Results}}$$</summary>
<div align="center">
	<img width = "1200" src="https://github.com/ConorWarrilow/Academic-Success-Analysis/assets/152389538/38b8f5fe-a77b-4a3b-95f2-9826be62e38f">
</div>

 <div align="center">
	<img width = "600" src="https://github.com/ConorWarrilow/Academic-Success-Analysis/assets/152389538/d5d706cf-ab81-43e7-83bc-eff12f2163cd">
</div>
</details>

### Problems with Impurity-Based Feature Importance
The imporuty-based approach is performed on the training data, therefore if our tree-based model is overfitting and its performance of the test data is porr compared to the training data, the results for the analysis are not reliable. Another problem with the impurity-based approach is its bias towards favoring numerical feature, or features with more categories.


## $${\color{#00A5A8}\text{3.2 Permutation-Based Feature Importance}}$$
Next we'll perform permutation-based feature importance. In this method, each feature is asynchronously shuffled and the drop in model performance is recorded. A significant drop in model performance indicates the feature to be important. If not drop in performance is recorded, we know the model's performance isn't reliant on the feature (although it doesnt necessarily imply we can drop the feature).

The same procedure from the impurity-based analysis was conducted, with the measurement of interest being the drop (or lack thereof) in model performance. 

**Permutation Results:**

<details>
  <summary>$${\color{#72B3A2}\text{View Permutation-Based Results}}$$</summary>
<div align="center">
	<img width = "1200" src="https://github.com/ConorWarrilow/Academic-Success-Analysis/assets/152389538/fe550290-0092-4e2e-8fac-15acb9a5f2bc">
</div>

 <div align="center">
	<img width = "600" src="https://github.com/ConorWarrilow/Academic-Success-Analysis/assets/152389538/5db741a9-1c24-427b-8b44-ec842e95aee6">
</div>
</details>


### Problems with Permutation-Based Feature Importance
Despite being a popular method of evaluating feature importance, permutation importance is a problematic evaluation method. The following article does an excellent job about why we should be careful when using permutation importance, or to simply [stop permuting features](https://towardsdatascience.com/stop-permuting-features-c1412e31b63f) altogether.

To summarise, permutation based importance is:
- Biased toward colinear features, as well as features that have many categories. further suggest that bootstrapping exaggerates these effects
- Exaggerated by OOB measures where they overestimate the importance of colinear features
- Affected by true features being correlated with noise features




## $${\color{#00A5A8}\text{3.3 Feature Importance Summary}}$$

<details>
  <summary>$${\color{#72B3A2}\text{View Feature Importance Summary Table}}$$</summary>
<div align="center">
	<img width = "600" src="https://github.com/ConorWarrilow/Academic-Success-Analysis/assets/152389538/9acb8d6e-dd6f-47d7-aee9-8b997414a123">
</div>
</details>



## $${\color{#00A5A8}\text{3.4 Alternative Methods of Feature Importance}}$$














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
