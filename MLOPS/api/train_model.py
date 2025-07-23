import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data={   
      'math':   [78,56,89,58,71,80,90],
      'science':[45,98,63,54,63,25,97],
      'english':[30,95,36,54,58,92,96],
      'Result':['fail','pass','fail','pass','pass','fail','pass']
}
df=pd.DataFrame(data)
df['Result']=df['Result'].map({'pass':1,'fail':0})
#train the model
x=df[['math','science','english']]
y=df['Result']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
#call the model
model=LogisticRegression()
model.fit(x_train,y_train)
joblib.dump(model,'model.pkl')