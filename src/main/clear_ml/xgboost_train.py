
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from clearml import Task
import xgboost as xgb
import numpy as np

# Always initialize ClearML before anything else. Automatic hooks will track as
# much as possible for you!
task = Task.init(
    project_name="Getting Started",
    task_name="XGBoost Training",
    output_uri=True  # IMPORTANT: setting this to True will upload the model
    # If not set the local path of the model will be saved instead!
)

# Training data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100
)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Setting the parameters
params = {
    'max_depth': 2,
    'eta': 1,
    'objective': 'reg:squarederror',
    'nthread': 4,
    'eval_metric': 'rmse',
}

# Make sure ClearML knows these parameters are our hyperparameters!
task.connect(params)


# Train the model
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=200,
    evals=[(dtrain, "train"), (dtest, "test")],
    verbose_eval=0,
)

# Save the model, saving the model will automatically also register it to
# ClearML thanks to the automagic hooks
bst.save_model("best_model.json")