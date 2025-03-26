# Titanic Survival Prediction: Decision Tree vs Random Forest
Overview: This repository contains a series of machine learning experiments comparing the performance of Decision Trees and Random Forests on the Titanic dataset. The primary goal of the project is to predict the survival of passengers aboard the Titanic using features such as age, sex, class, fare, and embarkation port.

Key analyses performed include:
	•	Classification task (Decision Tree vs Random Forest)
	•	Overfitting risks (Decision Tree vs Random Forest)
	•	Hyperparameter tuning for Decision Trees and Random Forests
	•	Training time vs. accuracy for different numbers of trees in Random Forest
	•	Decision boundary visualization for Decision Trees

Dataset:
Dataset Link:https://www.kaggle.com/datasets/yasserh/titanic-dataset

The dataset used in this project is the Titanic dataset from Kaggle. It contains information about passengers on the Titanic, including:
	•	Passenger Class (Pclass)
	•	Sex (Sex)
	•	Age (Age)
	•	Siblings/Spouses Aboard (SibSp)
	•	Parents/Children Aboard (Parch)
	•	Fare (Fare)
	•	Embarked (Embarked)

Key Insights
	•	Overfitting with Decision Trees: Deep Decision Trees overfit the training data, showing large differences between training and test accuracy.
	•	Random Forests Reduce Overfitting: Random Forests mitigate overfitting by averaging the results of many trees, leading to better generalization.
	•	Training Time vs. Accuracy: As the number of trees in a Random Forest increases, the model’s accuracy improves, but the training time also increases. A balance is needed to optimize performance.
	•	Decision Boundaries Visualization: Visualizations of decision boundaries for Decision Trees with different depths highlight how deeper trees tend to overfit, creating complex and less generalizable decision boundaries.

Files
	•	Titanic-Dataset.csv: The Titanic dataset used for classification.
  • ML_Assignment.py: Python script implementing the machine learning models (Decision Tree, Random Forest) and analyses.
	•	README.md: This file.

Dependencies

The code requires the following Python libraries:
	•	pandas
	•	numpy
	•	scikit-learn
	•	matplotlib

You can install the required libraries using pip:

pip install pandas numpy scikit-learn matplotlib

Setup Instructions:
	1.	Clone this repository to your local machine:

git clone: https://github.com/your-username/ML_Assignment.git

	2.	Download the Titanic dataset from Kaggle and place it in the repository directory as Titanic-Dataset.csv.
	3.	Run the Python script ML_Assignment.py to perform the analysis.

python ML_Assignment.py

Code Walkthrough:
1. Data Preprocessing
	•	Irrelevant columns like PassengerId, Name, Ticket, and Cabin are dropped.
	•	Missing values are handled by filling in the median for numerical features (Age, Fare) and the mode for categorical features (Embarked).
	•	Categorical features (Sex, Embarked) are encoded to numerical values for model compatibility.

2. Model Training
	•	Decision Tree Classifier and Random Forest Classifier are trained on the data to predict the survival of passengers.
	•	Cross-validation is used to evaluate model performance.

3. Overfitting Analysis
	•	Learning Curves: Learning curves are plotted for both models to illustrate overfitting (large difference between training and testing accuracy) in Decision Trees.
	•	Training Time vs. Accuracy: A comparison of training times and accuracy for different numbers of trees in Random Forest.

4. Hyperparameter Tuning
	•	Grid Search is used to tune hyperparameters for both Decision Trees and Random Forests, optimizing accuracy.

5. Decision Boundaries Visualization
	•	The decision boundaries of Decision Trees are visualized for different tree depths (3, 5, and 10) to observe how depth impacts overfitting.

Results
	•	The Random Forest model consistently outperforms the Decision Tree model in terms of accuracy and generalization.
	•	The learning curve for Decision Trees shows significant overfitting, whereas Random Forests have more stable accuracy across training sizes.
	•	The optimal number of trees for Random Forest is determined by balancing accuracy and training time.
	•	Decision boundaries for shallow trees are simple, while deep trees show complex boundaries prone to overfitting.

Conclusion: This project demonstrates the power of ensemble methods like Random Forests in overcoming the limitations of Decision Trees, particularly in terms of overfitting and generalization. Hyperparameter tuning and visualization of decision boundaries further help in understanding how different model configurations impact performance.

References
	1.	Titanic Dataset - Kaggle: https://www.kaggle.com/c/titanic
	2.	Scikit-learn Documentation: https://scikit-learn.org/stable/
	3.	Understanding Random Forests: https://towardsdatascience.com/understanding-random-forest-58d6d1e1e24c
	4.	Hyperparameter Tuning with GridSearchCV - Scikit-learn: https://scikit-learn.org/stable/modules/grid_search.html
	5.	Overfitting vs Underfitting: https://towardsdatascience.com/overfitting-and-underfitting-in-machine-learning-3e498e4b4f8e
