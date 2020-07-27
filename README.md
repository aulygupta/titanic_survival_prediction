# Name:TITANIC SURVIVAL PREDICTION


## Imported libraries
```
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

```
### Load and View the data

~~~
data=pd.read_csv('titanic_data.csv')
data.head()
~~~

### Plotting the survival rate

![alt Survived](https://github.com/rahuljadli/Titanic-Survival-Prediction/blob/master/screen_shots/Survival.png)

### Survival Based on Gender

![alt Sex Survival ](https://github.com/rahuljadli/Titanic-Survival-Prediction/blob/master/screen_shots/Survival_gender.png)

### Plotting Survival Based on Passenger Class

![alt Sex Survival ](https://github.com/rahuljadli/Titanic-Survival-Prediction/blob/master/screen_shots/Survival_Pclass.png)

### Plotting Survival Based on Sibling

![alt Sex Survival ](https://github.com/rahuljadli/Titanic-Survival-Prediction/blob/master/screen_shots/Survival_sibling.png)

### Ploting Survival Based on Parents

![alt Sex Survival ](https://github.com/rahuljadli/Titanic-Survival-Prediction/blob/master/screen_shots/Survival_parch.png)

## Data filling

~~~
clean_test.Age = clean_test.Age.fillna(titanic_data['Age'].mean())
testing_data.Fare=testing_data.Fare.fillna(data.Fare.mean())
~~~

# Using Different Model's 

## For Creating Train and Test Data set

~~~
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

~~~
# Training the model

~~~
model=LogisticRegression()
model.fit(x_train,y_train)
~~~
# Making the prediction

~~~
new_prediction=model.predict(testing_data)
~~~
## For accuracy score

~~~
from sklearn.metrics import accuracy_score


acc_logreg = round(accuracy_score(prediction, y_test) * 100, 2)
print(acc_logreg)
~~~
