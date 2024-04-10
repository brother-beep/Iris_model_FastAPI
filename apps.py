import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI
from pydantic import BaseModel
import json

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open('iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Define FastAPI app
app = FastAPI()

# Define input data schema
class InputData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Load the trained model
with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)
    


# Define prediction endpoint
@app.post('/predict')
@app.get('/predict')
def predict_species(input_data: InputData):
    data = input_data.json()
    input_dictionary = json.loads(data)
    features = [input_dictionary['sepal_length'], input_dictionary['sepal_width'], input_dictionary['petal_length'], input_dictionary['petal_width']]
    prediction = model.predict([features])[0]
    species = iris.target_names[prediction]
    return {'species': species}

# Define root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Prediction API!"}