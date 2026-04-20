import json
import os
import argparse
from typing import Dict

DEFAULT_MODEL_FILE = "model/data.json"

def parse_arguments():
    """Parses the model file path from the command line."""
    parser = argparse.ArgumentParser(description="🚀 Predict car price based on a trained model.")
    parser.add_argument(
        "model_file", 
        nargs="?", 
        default=DEFAULT_MODEL_FILE,
        help=f"Path to the trained JSON model (default: {DEFAULT_MODEL_FILE})"
    )
    return parser.parse_args()

def load_model(filepath: str) -> Dict[str, float]:
    """Loads the trained model parameters from a JSON file."""
    fallback_model = {
        "theta0": 0.0, 
        "theta1": 0.0, 
        "min_m": 0.0, 
        "max_m": 1.0
    }
    
    if not os.path.exists(filepath):
        print(f"⚠️ Warning: '{filepath}' not found. Using default values (0).")
        return fallback_model

    try:
        with open(filepath, 'r') as f:
            model = json.load(f)
            required_keys = ["theta0", "theta1", "min_m", "max_m"]
            if all(key in model for key in required_keys):
                print(f"✅ Model successfully loaded from '{filepath}'")
                return model
            else:
                raise KeyError("Missing required parameters in JSON")

    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"❌ Error: Model file is corrupted or invalid ({e}).")
        print("   Falling back to default values (0).")
        return fallback_model

def get_user_mileage() -> float:
    """Prompts the user for mileage and validates the input."""
    while True:
        try:
            line = input("Please enter the mileage (km): ").strip()
            if not line:
                continue
            
            mileage = float(line)
            if mileage < 0:
                print("❌ Error: Mileage cannot be a negative value.")
                continue
            return mileage
        except ValueError:
            print("❌ Error: Please enter a valid numerical value.")

def predict_price(mileage: float, model: Dict[str, float]) -> float:
    """Calculates the predicted price based on normalized mileage."""
    denom = model['max_m'] - model['min_m']
    norm_mileage = (mileage - model['min_m']) / denom if denom != 0 else 0.0
    
    # Prediction: y = theta0 + (theta1 * x_norm)
    prediction = model['theta0'] + (model['theta1'] * norm_mileage)
    
    # Handle negative prices as 0
    return max(0.0, prediction)

def main():
    args = parse_arguments()
    # 1. Load the model
    model = load_model(args.model_file)

    # 2. Get user input
    mileage = get_user_mileage()

    # 3. Calculate prediction
    estimated_price = predict_price(mileage, model)

    # 4. Output result
    print(f"✨ Estimated price: [ {estimated_price:,.2f} ]")

if __name__ == "__main__":
    main()