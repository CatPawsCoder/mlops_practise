#!/bin/bash


project_path="$(dirname "$(readlink -f "$0")")"

# Step 1: Generate data
python "$project_path/data_creation.py"

# Step 2: Preprocess data
python "$project_path/model_preprocessing.py"

# Step 3: Train model
python "$project_path/model_preparation.py"

# Step 4: Test model
python "$project_path/model_testing.py"