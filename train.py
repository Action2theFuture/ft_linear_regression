import json
import os
import argparse
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from preprocessing import csv_file_validator, load_csv, remove_outliers, normalize_features

# --- Constants & Help Texts ---
DATA_DIR = "data"
MODEL_DIR = "model"
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_EPOCHS = 1000
TOLERANCE = 1e-7
PLOT_FIG_SIZE = (12, 5)

YELLOW = "\033[93m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

HELP_DESCRIPTION = f"{BOLD}{CYAN}🚀 Linear Regression Training Tool for car price prediction.{RESET}"
HELP_EPILOG = textwrap.dedent(f"""
    {BOLD}{YELLOW}[ 💡 Hyperparameter & Preprocessing Guide ]{RESET}
    {BOLD}- Learning Rate (--lr):{RESET}
        * {GREEN}0.1{RESET}   : Standard default. Recommended for stable training.
        * {RED}1.0+{RESET}   : Fast training, but risk of {RED}'nan'{RESET} (Gradient Explosion).

    {BOLD}- Data Cleaning (--clean):{RESET}
        * {CYAN}Enabled{RESET} : Removes outliers using the {BOLD}IQR (1.5x){RESET} method.
    """)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=HELP_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP_EPILOG
    )
    # Use the validator from preprocessing module
    parser.add_argument("filename", type=csv_file_validator, help="Target CSV file name")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Max epochs")
    parser.add_argument("--clean", action="store_true", help="Remove outliers before training")
    return parser.parse_args()

def train_model(x_norm: np.ndarray, y: np.ndarray, lr: float, epochs: int) -> Tuple[float, float, List[float]]:
    """Performs Gradient Descent to find optimal theta0 and theta1."""
    t0, t1 = 0.0, 0.0
    m = len(x_norm)
    cost_history = []

    for epoch in range(epochs):
        predictions = t0 + (t1 * x_norm)
        errors = predictions - y
        
        t0 -= lr * (1/m) * np.sum(errors)
        t1 -= lr * (1/m) * np.sum(errors * x_norm)
        
        cost = np.sum(errors**2) / (2 * m)

        if np.isinf(cost) or np.isnan(cost):
            print(f"\n{RED}❌ Error: Model diverged. Lower your learning rate.{RESET}")
            return t0, t1, cost_history

        if epoch > 0 and abs(cost_history[-1] - cost) < TOLERANCE:
            print(f"{GREEN}✨ Early stopping at epoch {epoch} (Converged).{RESET}")
            break
            
        cost_history.append(cost)
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d}: Cost = {cost:.2f}")
            
    return t0, t1, cost_history

def calculate_metrics(y_true, y_pred):
    """Calculates R2, MAE, and RMSE scores."""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    return {"r2_score": r2, "mae": mae, "rmse": rmse}

def validate_and_save(params: Dict, filepath: str) -> bool:
    """Validates the model parameters against domain rules and saves to JSON."""
    warnings = []
    
    # Extract metrics and parameters for validation
    r2 = params.get('metrics', {}).get('r2_score', 0.0)
    theta1 = params.get('theta1', 0.0)

    # Rule 1: Car prices should generally decrease as mileage increases (Negative Correlation)
    if theta1 >= 0:
        warnings.append(f"{RED}{BOLD}- Abnormal trend:{RESET} Price increases with mileage (theta1 >= 0).")
    
    # Rule 2: Check if the model's reliability is high enough
    if r2 < 0.4:
        warnings.append(f"{YELLOW}- Low reliability:{RESET} R² Score is below 0.4 ({r2:.4f}).")

    # If issues are found, ask for user confirmation
    if warnings:
        print(f"\n{YELLOW}⚠️  [Warning] Model validation issues found:{RESET}")
        for w in warnings:
            print(w)
        
        while True:
            choice = input(f"\n👉 Force save this model to '{filepath}'? (y/n): ").lower().strip()
            if choice == 'y':
                break
            elif choice == 'n':
                print(f"{RED}🚫 Saving canceled by user.{RESET}")
                return False
            else:
                print("❌ Invalid input. Please enter 'y' or 'n'.")

    # Final saving process
    try:
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=4)
        print(f"✅ Model successfully saved at: {BOLD}{filepath}{RESET}")
        return True
    except Exception as e:
        print(f"{RED}❌ Error: Failed to save the model file ({e}).{RESET}")
        return False

def show_plots(m_raw: np.ndarray, prices: np.ndarray, t0: float, t1: float, 
               history: List[float], min_m: float, max_m: float):
    """
    Visualizes the regression result and the learning curve.
    """
    plt.figure(figsize=PLOT_FIG_SIZE)

    # Plot 1: Regression Line vs. Data Points
    plt.subplot(1, 2, 1)
    plt.scatter(m_raw, prices, color='blue', alpha=0.5, label='Actual Data')
    
    # Generate points for the regression line
    x_range = np.linspace(min_m, max_m, 100)
    # Important: We must normalize these points using the same logic as training
    x_norm = (x_range - min_m) / (max_m - min_m if max_m != min_m else 1.0)
    y_pred = t0 + (t1 * x_norm)
    
    plt.plot(x_range, y_pred, color='red', linewidth=2, label='Model Prediction')
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price')
    plt.title('Regression Result')
    plt.legend()

    # Plot 2: Learning Curve (Cost over Epochs)
    plt.subplot(1, 2, 2)
    # Only plot the first 100 epochs or until it stabilized
    display_limit = min(len(history), 100) 
    plt.plot(history[:display_limit], color='green')
    plt.xlabel('Epochs (First 100)')
    plt.ylabel('Cost (MSE)')
    plt.title('Learning Curve (Cost Reduction)')

    plt.tight_layout()
    print(f"\n{CYAN}📈 Displaying visual results...{RESET}")
    plt.show()

def main():
    args = parse_arguments()
    
    # 0. Output Path Setup
    model_name = os.path.splitext(os.path.basename(args.filename))[0]
    output_path = os.path.join(MODEL_DIR, f"{model_name}.json")
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

    # 1. Data Pipeline (Loading -> Cleaning -> Normalizing)
    raw_x, raw_y = load_csv(DATA_DIR, args.filename)
    if raw_x is None: return

    if args.clean:
        print(f"{CYAN}🔍 Analyzing dataset for outliers...{RESET}")
        raw_x, raw_y = remove_outliers(raw_x, raw_y)

    norm_x, min_m, max_m = normalize_features(raw_x)

    # 2. Train
    print(f"🚀 Training on '{args.filename}' (LR: {args.lr}, Epochs: {args.epochs})")
    t0, t1, history = train_model(norm_x, raw_y, args.lr, args.epochs)

    # 3. Evaluate
    preds = t0 + (t1 * norm_x)
    metrics = calculate_metrics(raw_y, preds)
    
    print(f"\n{BOLD}📊 Results:{RESET} R²={metrics['r2_score']:.4f}, MAE={metrics['mae']:.2f}")

    # 4. Save & Visualize
    model_params = {
        "theta0": float(t0), "theta1": float(t1),
        "min_m": float(min_m), "max_m": float(max_m),
        "metrics": metrics
    }
    
    # Using the existing validate_and_save and show_plots logic (omitted for brevity)
    if validate_and_save(model_params, output_path):
        show_plots(raw_x, raw_y, t0, t1, history, min_m, max_m)

if __name__ == "__main__":
    main()