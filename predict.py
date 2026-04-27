import argparse
from model import TinyLinearRegression

# --- Configuration & UI ---
DEFAULT_MODEL_FILE = "model/data.json"
BOLD = "\033[1m"
CYAN = "\033[96m"
RESET = "\033[0m"

def parse_arguments():
    """Parses the model file path from the command line."""
    parser = argparse.ArgumentParser(description="🚀 Predict car price using a trained model.")
    parser.add_argument(
        "model_file", 
        nargs="?", 
        default=DEFAULT_MODEL_FILE,
        help=f"Path to the trained JSON model (default: {DEFAULT_MODEL_FILE})"
    )
    return parser.parse_args()

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

def main():
    """Main execution flow for prediction."""
    args = parse_arguments()
    
    # 1. Instantiate the model object
    model = TinyLinearRegression()
    
    # 2. Load the model state
    # The 'load' method inside the class handles the file opening and key checks
    if model.load(args.model_file):
        r2 = model.metrics.get('r2_score', 0.0)
        print(f"✅ Model successfully loaded from '{args.model_file}' (R²: {r2:.4f})")
    else:
        print(f"⚠️ Warning: '{args.model_file}' not found or invalid. Using default values (0).")

    # 3. Get user input
    mileage = get_user_mileage()

    # 4. Perform prediction using the class method
    # The class knows how to normalize the mileage using its internal min_m/max_m
    estimated_price = model.predict(mileage, verbose=True)

    # 5. Output result
    print(f"✨ Estimated price: [ {BOLD}{estimated_price:,.2f}{RESET} ]")

if __name__ == "__main__":
    main()