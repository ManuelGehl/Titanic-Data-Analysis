# Titanic data analysis and survival prediction

<img src="https://github.com/ManuelGehl/titanic_kaggle/blob/main/Images/titanic.png?raw=true" width=500>

## Overview

This repository contains my data analysis project on the iconic Titanic dataset from [Kaggle](https://www.kaggle.com/competitions/titanic/overview). 

The aim is to discover the factors that impacted the survival of passengers on the Titanic and develop a predictive machine learning model based on passenger characteristics.

## Introduction

On April 14, 1912, 11:40 PM the RMS Titanic, during her maiden voyage from Southampton to New York, collided with an iceberg. This significant incident in nautical history resulted in the tragic sinking of the supposedly unsinkable vessel within just 160 minutes, causing the loss of numerous passengers and crew members. The Titanic was a monumental achievement, at 291.1 meters it was the largest moving structure of its time and had a capacity of up to 3,547 passengers. However, on the night of the sinking, there were only 2,223 people on board. It is worth mentioning that despite the capability to carry up to 64 lifeboats, the ship was only outfitted with 20, a few of which were not even completely occupied.

Today, the remains of this massive ship lie split in two, 12,600 feet below the ocean's surface, and are listed as a UNESCO World Underwater Heritage Site. The story of the legendary ship remains fascinating to this day. It is a tale of man's technological progress and his desire to overcome the limits of nature, but in the end he fails to do so.

## Results & Discussion
### [Data Exploration](https://github.com/ManuelGehl/Titanic-data-analysis/blob/main/titanic_data_exploration.ipynb)
**Description of the dataset**

The given train dataset describes a part of the passenger list of the Titanic. It consists of 891 entries and 12 features, most of which are complete, with only age, port of embarkation and cabin number missing (Tab. 1). 
<br></br>

*Table 1: Description of the training set.*

|Column name|	Feature description|	Entries|	Encoding|
|-----------|--------------------|---------|----------|
|PassengerId|	Continuous index for passengers|	891	|Integers|
|Survived	|Survival|	891|	0 = Not survived, 1 = survived|
|Pclass	|Passenger class	|891|	1 = 1st, 2 = 2nd , 3 = 3rd class| 
|Name|	Passenger name|	891|	String|
|Sex	|Gender of passenger|	891|	“male”, “female”|
|Age	|Age in years	|714	|Float|
|SibSp|	Number of siblings or spouses aboard|	891|	Integers|
|Parch|	Number of parents or children aboard|	891	|Integers|
|Ticket	|Ticket number|	891|	String|
|Fare|	Ticket price|	891|	Float|
|Cabin|	Cabin number|	204|	String|
|Embarked|	Port of embarkation|	889|	C = Cherbourg, Q = Queenstown, S = Southampton|

<br></br>

The distribution of the various features revealed a significant imbalance in virtually every feature distribution (Fig. 1). The majority of passengers did not survive the Titanic disaster, as indicated by the overall survival rate of only 38.4%. In terms of gender, the dataset is dominated by male passengers, who make up 64.8% of the total. The majority of the passengers (55.1%) were in 3rd class, the cheapest class available on the ship, with the remainder split roughly evenly between 1st class (24.2%) and 2nd class (20.7%). Similarly, the majority of passengers (72.3%) embarked from Southampton, while 18.9% embarked from Cherbourg and only 8.6% from Queenstown. The age distribution appears to be somewhat similar to a normal distribution; however, a significant number of 177 age values are missing. The Siblings/Spouses and Parents/Children features are both heavily skewed towards passengers traveling alone on the Titanic. The fare distribution is skewed to the left, with a mean fare of £32 and a median of £14. This distribution suggests that the majority of passengers paid lower fares, as evidenced by the passenger class distribution.

<br></br>
 <img src="https://github.com/ManuelGehl/Titanic-data-analysis/blob/main/Images/Histogram.png?raw=true">
 
 *Figure 1: Histograms showing the distribution of different features in the training dataset.* 
<br></br>

**Relationship between survival and different features**

Survival rates on the Titanic were strongly associated with passenger class, fare paid, and gender, and to a lesser extent with age, number of parents/children and siblings/spouses, and port of embarkation (Fig. 1A). Passenger class had the highest absolute correlation coefficient with survival (−0.34) of all numerically coded features tested. The probability of survival increased successively from third class passengers with 24.2% to second class with 47.3% and first class with 63.0% (Fig. 1B). This behavior is most likely related to the fare paid, which had the second highest absolute correlation coefficient of 0.26 (Fig. 1A). In terms of correlation coefficient, the other characteristics tested had a much smaller impact on survival. The correlation between age and survival is slightly negative, with a coefficient of −0.08, indicating that younger passengers had a slightly better chance of surviving the sinking. Survival rates for men and women differ significantly, with women having almost four times the chance of survival at 74.2% compared to 18.9% for men. Interestingly, the port of embarkation seems to play a bigger role in the survival of passengers.  Passengers departing from Cherbourg had the highest survival rate at 55.4%, while those departing from Queenstown or Southampton had similar rates at 39.0% and 33.7%, respectively (Fig. 1B). The number of parents/children had a positive correlation coefficient of 0.08, while siblings/spouses had a negative coefficient of −0.04.

<br></br>
<img src="https://github.com/ManuelGehl/Titanic-data-analysis/blob/main/Images/Correlation%20matrix%20+%20survival%20rates.png?raw=true" height=500>

*Figure 2:**(A)** Correlation matrix showing the correlation coefficients of different features in the training data set. **(B)** Average survival rates with respect to different features.* 
 

Since the survival rates were somewhat ambiguous depending on the number of siblings/spouses and parents/children, the characteristics were combined into the total number of relatives on board (Fig. 1B). The survival rates indicate three distinct categories depending on the presence of relatives (Fig 2). For individuals traveling alone, the survival rate was relatively low at 30.4%. However, the rate increased significantly for those traveling with at least one and up to three relatives, with an average survival rate of 57.9%. Traveling with more than three relatives resulted in a sharp drop in survival, with a rate of 16.1%.

<br></br>
<img src="https://github.com/ManuelGehl/Titanic-data-analysis/blob/main/Images/Survival%20rates%20relatives.png?raw=true" height=400>

*Figure 3: Average survival rates according to the number of relatives on board.*

<br></br>
 
The relationship between improved survival rates for passengers boarding at Cherbourg can be explained by secondary effects. Passengers embarking at Cherbourg paid an average fare of £60 and travelled predominantly in first class, whereas those embarking at Queenstown and Southampton paid much lower fares, averaging £13 and £27 respectively, and travelled predominantly in third class (Fig. 3).

<br></br>
<img src="https://github.com/ManuelGehl/Titanic-data-analysis/blob/main/Images/Survival%20rates%20ports.png?raw=true" height=300>

*Figure 4: Average fare paid and most common passenger class selected by passengers embarking at different ports.*
 

The role of age in terms of survival is somewhat ambiguous. On one hand, passengers aged 0 to 5 have a significantly higher survival rate of 70.5%, compared to the overall average of 38.4% (Fig. 4A). On the other hand, the survival rate for most age groups is evenly distributed around the overall average without displaying any discernible pattern. A notable exception is the group of passengers aged 10-15, who exhibit a heightened survival rate of 57.9% in comparison to the general value. This phenomenon can be explained by the synergistic effects of passenger class, paid fare, and number of relatives, as all these values are somewhat higher than the overall average of passengers (Fig. 4B). Therefore, it is probable that the higher survival rate for these passengers is a result of sampling.

<img src="https://github.com/ManuelGehl/Titanic-data-analysis/blob/main/Images/Survival%20rates%20age%20categories%20+%20age%2010-15.png?raw=true" height=400>

*Figure 5: **(A)** Average survival rate for different age groups. **(B)** Average characteristic values and fares paid by total passengers (orange) and passengers aged 10-15 (blue).*
<br></br>
Taken together, passenger class, gender, age, the number of siblings/spouses, the number of parents/children, the number of relatives, the fare paid, and the port of embarkation contributed to the survival chance of a passenger, albeit to varying degrees. The strongest contributors were gender, fare paid, and the resulting passenger class. In the next step, all these features were used in different combinations to test different machine learning models to classify whether a passenger would survive the Titanic disaster.

### [Machine Learning Model Building](https://github.com/ManuelGehl/Titanic-data-analysis/blob/main/titanic_model_building.ipynb)
**Data Transformation and Screening**
In preparation for the machine learning model, the dataset was transformed. First, the passenger ID, the ticket number, the cabin number, and the name of the passenger were dropped, as they did not provide any useful contribution to survival. Then, a pipeline was constructed that first imputed the numeric columns ("Age", "Fare") with the corresponding means and the categorical columns ("Pclass", "Sex", "SibSp", "Parch", "Embarked") with the corresponding most frequent category. After the imputation step, the number of siblings/spouses was added to the number of parents/children and transformed into a new column named "Relatives". Relatives were categorized into 3 bins with 0 relatives, 1-3 relatives, and more than 3 relatives, while age was categorized into 2 bins with 0-10 years and more than 10 years. Both categorizations were done according to the different survival rates of the categories as described above (see Fig. 3 and Fig. 5A). Finally, the numerical features were standard scaled and the categorical features were one-hot coded. Different subsets of the transformed dataset were trained with 5 different models to determine the best performing feature combination and model. The base set consisted of the features “Fare” (continuous), ”Pclass” (categorical), “Sex” (categorical), “Embarked” (categorical) and were the common features of all subsets. Subset 1 additionally contained “Age” (continuous) and “Relatives” (continuous). Subset 2 was the base set plus “Age” (categorical younger and older than 15 years) and “Relatives” (3 classes). Subset 3 extended the base set by “Parch” (continuous) and “SibSp” (continuous). For the screening a support vector classifier (SVC), a k-nearest neighbor (KNN) classifier, a random forest classifier, a Gradient boosting classifier and a Multilayer perceptron (MLP) with 2 hidden layers and 100 neurons each with default parameters were used. 

In general, all models benefited from the inclusion of the number of relatives and the age, and the best performing models were SVC, Gradient Boosting, and MLP (Tab. 2). Among the models tested on the base set, SVC achieved the highest accuracy with an average accuracy of 81.4% ± 2.9, followed closely by KNN with an accuracy of 80.4% ± 3.1. Random Forest and Gradient Boosting also showed competitive performance, with accuracies of 80.8% ± 4.6 and 80.9% ± 4.0, respectively. MLP achieved an accuracy of 81.1% ± 3.1, which was comparable to the best performing models. On Subset 1, Gradient Boosting emerged as the top performing model with an accuracy of 83.3% ± 4.3, surpassing its performance on the baseline set. SVC also showed improved performance on Subset 1, with an accuracy of 82.7% ± 3.9. KNN and Random Forest showed similar accuracies of approximately 81.0% ± 4.5 and 81.0% ± 4.9, respectively. MLP achieved a slightly lower accuracy of 80.6% ± 3.8. In Subset 2, Gradient Boosting continued to excel, achieving the highest accuracy in total of 83.3% ± 4.1. SVC and KNN reached accuracies of 82.2% ± 3.3 and 82.3% ± 4.7, respectively. Random Forest achieved an accuracy of 81.3% ± 4.4, while MLP showed a competitive performance with an accuracy of 82.4% ± 4.1. In Subset 3, all models experienced a slight decrease in performance compared to the previous subsets. SVC showed the highest accuracy among the models with a score of 80.8% ± 2.5. KNN and Gradient Boosting followed closely with accuracies of 79.8% ± 4.4 and 80.8% ± 4.7, respectively. Random Forest and MLP showed similar performance with accuracies of 79.2% ± 5.3 and 79.7% ± 4.1, respectively. Subset 3 was the only dataset that included the number of siblings/spouses and parents/children as a separate feature. This combination of features does not seem to be beneficial for all models, as indicated by the lowest average subset accuracy of 80.1% of all subsets (Tab. 2). The best performing subset was subset 2 with an average accuracy of 82.3%, indicating that the categorical versions of "Age" and "Relatives" can be learned best by the models. Therefore, Subset 2 was used for all following experiments and the three best models were selected as Gradient Boosting classifier, MLP and SVC.

*Table 2: Results for screening different data subsets and models. The accuracy of the models is shown as the mean with the corresponding standard deviation determined by 5-fold cross validation.*
|     Data Set    |     Support   Vector Classifier    |     KNN   Classifier    |     Random   Forest     Classifier    |     Gradient   Boosting     Classifier    |     Multi-Layer   Perceptron    |     Average   subset accuracy    |
|-----------------|------------------------------------|-------------------------|---------------------------------------|-------------------------------------------|---------------------------------|----------------------------------|
|     Base set    |     81.4 ± 2.9                     |     80.4 ± 3.1          |     80.8 ± 4.6                        |     80.9 ± 4.0                            |     81.1 ± 3.1                  |     80.9                         |
|     Subset 1    |     82.7 ± 3.9                     |     81.0 ± 4.5          |     81.0 ± 4.9                        |     83.3 ± 4.3                            |     80.6 ± 3.8                  |     81.7                         |
|     Subset 2    |     82.2 ± 3.3                     |     82.3 ± 4.7          |     81.3 ± 4.4                        |     83.3 ± 4.1                            |     82.4 ± 4.1                  |     82.3                         |
|     Subset 3    |     80.8 ± 2.5                     |     79.8 ± 4.4          |     79.2 ± 5.3                        |     80.8 ± 4.7                            |     79.7 ± 4.1                  |     80.1                         |

<br></br>

**Fine-tuning and evaluation on the test set**

Two approaches were used to fine-tune the models. In the first approach, the fast SVC and Gradient Boosting classifiers were optimized using a randomized search with 5-fold cross-validation. For the MLP, the number of hidden layers was set to 2, and the number of neurons, activation function, and learning rate were tuned using the Keras Hypertuner. For the SVC model, the best parameters were a regularization parameter C of 4.95, uniform class weights, and a polynomial kernel function of degree 2.  The optimized model yielded a mean test score of 84.4% ± 2.0%. The best parameters for the gradient boosting classifier were a learning rate of 0.156, a logarithmic loss, and 223 boost stages, resulting in a mean test score of 83.4% ± 2.3%. For the MLP, the best hyperparameters found were 272 units per hidden layer, ReLU as the activation function, and a learning rate of 0.0041, resulting in a validation accuracy of 83.8%.

The final evaluation was done by predicting the survival of the passengers in the given test set and submitting the predictions to Kaggle.com where they were scored. The SVC performed best with an accuracy of 78.2%, followed by the Gradient Boosting classifier with 76.8% and the MLP with 77.3%. Considering that the overall survival rate for the train dataset is 38.4%, a baseline model that always predicts non-survival would score 61.6%, the models show substantial performance. However, it is noteworthy that the accuracies on the test set are about 6% lower than those measured by k-fold cross-validation on the training and validation sets, respectively. This may indicate a sample consistency problem between the training and test data sets.

## Outlook

* Error analysis of the different models
* Ensemble of best performing models
* Check test set for missing values
* Placing how the Titanic Disaster fits into the context of other shipwrecks from the 18th to the 21st centuries

## Literature
Titanic Facts • 1,000+ Fascinating Facts and Figures. https://titanicfacts.net/ (2017).


