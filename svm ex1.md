```python
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
```


```python
dataset = pd.read_csv("SVMtrain.csv")
```


```python
dataset.columns
```




    Index(['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
           'Fare', 'Embarked'],
          dtype='object')




```python
dataset1=dataset.drop(["PassengerId"],axis=1)
```


```python
dataset1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>Male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>Male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
le=preprocessing.LabelEncoder()
```


```python
le.fit(dataset1["Sex"])
```




    LabelEncoder()




```python
print(le.classes_)
```

    ['Male' 'female']
    


```python
dataset1["Sex"]=le.transform(dataset1["Sex"])
```


```python
dataset1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
y=dataset1["Survived"]
y.head()
```




    0    0
    1    1
    2    1
    3    1
    4    0
    Name: Survived, dtype: int64




```python
X=dataset1.drop(["Survived"],axis=1)
```


```python
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
```


```python
X_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>350</th>
      <td>1</td>
      <td>0</td>
      <td>60.0</td>
      <td>0</td>
      <td>0</td>
      <td>35.0000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>124</th>
      <td>3</td>
      <td>0</td>
      <td>12.0</td>
      <td>1</td>
      <td>0</td>
      <td>11.2417</td>
      <td>1</td>
    </tr>
    <tr>
      <th>577</th>
      <td>3</td>
      <td>1</td>
      <td>60.0</td>
      <td>1</td>
      <td>0</td>
      <td>14.4583</td>
      <td>1</td>
    </tr>
    <tr>
      <th>422</th>
      <td>3</td>
      <td>1</td>
      <td>28.0</td>
      <td>1</td>
      <td>1</td>
      <td>14.4000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>118</th>
      <td>3</td>
      <td>1</td>
      <td>2.0</td>
      <td>4</td>
      <td>2</td>
      <td>31.2750</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>3</td>
      <td>1</td>
      <td>14.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.8542</td>
      <td>3</td>
    </tr>
    <tr>
      <th>158</th>
      <td>3</td>
      <td>0</td>
      <td>60.0</td>
      <td>8</td>
      <td>2</td>
      <td>69.5500</td>
      <td>3</td>
    </tr>
    <tr>
      <th>762</th>
      <td>1</td>
      <td>1</td>
      <td>36.0</td>
      <td>1</td>
      <td>2</td>
      <td>120.0000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>740</th>
      <td>1</td>
      <td>0</td>
      <td>36.0</td>
      <td>1</td>
      <td>0</td>
      <td>78.8500</td>
      <td>3</td>
    </tr>
    <tr>
      <th>482</th>
      <td>3</td>
      <td>1</td>
      <td>63.0</td>
      <td>0</td>
      <td>0</td>
      <td>9.5875</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_train.head()
```




    350    0
    124    1
    577    0
    422    0
    118    0
    Name: Survived, dtype: int64




```python
from sklearn import svm
```


```python
clf=svm.SVC(gamma=0.001,C=100) #svc=supportvector
```


```python
clf.fit(X_train,y_train)
```




    SVC(C=100, gamma=0.001)




```python
y_pred=clf.predict(X_test)
```


```python
accuracy_score(y_test,y_pred,normalize=True)
```




    0.7602996254681648




```python
confusion_matrix(y_test,y_pred)
```




    array([[124,  33],
           [ 31,  79]], dtype=int64)




```python
#203 records are correctly classified
#64  records are other records incorrectly classified
```


```python

```


```python

```
