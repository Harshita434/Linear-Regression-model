import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
#load the dataset
df = pd.read_csv("data.csv")
#encode location to numeric using labelencoder
print(df.head())
print(df.columns)
le=LabelEncoder()
df["LocationIndex"]=le.fit_transform(df["Location"])
print(df.head())
#define feature and target 
X=df[["Size","Bedroom","LocationIndex"]]
Y=df["Rent"]
#split the dataset
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
#train the linear model
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("r2 score:",r2_score(y_test,y_pred))
#compare
compare=pd.DataFrame({"actual":y_test,"prediction":y_pred.astype(int)})
print("\n actual vs prediction:\n",compare.head())
new_location="Mumbai"
new_location_index= le.transform([new_location])[0]
new_data=[[1200,3,new_location_index]]
predict=model.predict(new_data)
print(f"prediction rent based on this data is{predict}")