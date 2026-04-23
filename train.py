import os
import argparse
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

# Modular imports
from model import TinyLinearRegression
from preprocessing import csv_file_validator, load_csv, remove_outliers, normalize_features

# --- UI & Configuration ---
DATA_DIR = "data"
MODEL_DIR = "model"
DEFAULT_LR = 0.1
DEFAULT_EPOCHS = 1000
TOLERANCE = 1e-7
PLOT_SIZE = (12, 5)

# ANSI Colors for terminal output
YELLOW = "\033[93m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

HELP_DESC = f"{BOLD}{CYAN}🚀 Linear Regression Training Tool for car price prediction.{RESET}"
HELP_EPILOG = textwrap.dedent(f"""
    {BOLD}{YELLOW}[ 💡 Training Guide ]{RESET}
    - {BOLD}Learning Rate (--lr):{RESET} Standard default is 0.1.
    - {BOLD}Data Cleaning (--clean):{RESET} Removes outliers using IQR (1.5x).
    - {BOLD}Early Stopping:{RESET} Automatically halts when cost stabilizes below {TOLERANCE}.
    """)

def parse_arguments():
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(
        description=HELP_DESC,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP_EPILOG
    )
    parser.add_argument("filename", type=csv_file_validator, help="Target CSV file name")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Max iterations")
    parser.add_argument("--clean", action="store_true", help="Enable outlier removal")
    return parser.parse_args()

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates model performance: $R^2$, MAE, and RMSE."""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    return {"r2_score": r2, "mae": mae, "rmse": rmse}

def validate_and_save(model: TinyLinearRegression, filepath: str) -> bool:
    """Checks if the trained model makes sense before saving to JSON."""
    warnings = []
    r2 = model.metrics.get('r2_score', 0.0)
    
    if model.theta1 >= 0:
        warnings.append(rf"{RED}{BOLD}- Abnormal trend:{RESET} Price increases with mileage ($\theta_1 \ge 0$).")
    if r2 < 0.4:
        warnings.append(f"{YELLOW}- Low reliability:{RESET} $R^2$ Score is below 0.4 ({r2:.4f}).")

    if warnings:
        print(f"\n{YELLOW}⚠️  [Warning] Model validation issues found:{RESET}")
        for w in warnings: print(w)
        
        while True:
            choice = input(f"\n👉 Force save this model to '{filepath}'? (y/n): ").lower().strip()
            if choice == 'y': break
            elif choice == 'n': return False
            else: print("❌ Invalid input. Enter 'y' or 'n'.")

    # Call the internal class save method
    model.save(filepath)
    return True

def show_results(raw_x: np.ndarray, raw_y: np.ndarray, model: TinyLinearRegression, history: List[float]):
    """Visualizes the final regression line and the training cost history."""
    plt.figure(figsize=PLOT_SIZE)

    # Subplot 1: Regression Result
    plt.subplot(1, 2, 1)
    plt.scatter(raw_x, raw_y, color='blue', alpha=0.5, label='Actual Data')
    
    x_line = np.linspace(model.min_m, model.max_m, 100)
    y_line = [model.predict(val) for val in x_line]
    
    plt.plot(x_line, y_line, color='red', linewidth=2, label='Model Prediction')
    plt.xlabel('Mileage (km)'), plt.ylabel('Price'), plt.title('Regression Analysis'), plt.legend()

    # Subplot 2: Learning Curve
    plt.subplot(1, 2, 2)
    display_limit = min(len(history), 100) 
    plt.plot(history[:display_limit], color='green')
    plt.xlabel('Epochs (Initial Phase)'), plt.ylabel('Cost (MSE)'), plt.title('Learning Curve')

    plt.tight_layout()
    print(f"\n{CYAN}📈 Displaying visual results...{RESET}")
    plt.show()

def main():
    """Main execution pipeline."""
    args = parse_arguments()
    model = TinyLinearRegression()
    
    # 1. Data Loading
    raw_x, raw_y = load_csv(DATA_DIR, args.filename)
    if raw_x is None: return

    # 2. Preprocessing
    if args.clean:
        print(f"{CYAN}🔍 Scrubbing data for outliers...{RESET}")
        raw_x, raw_y = remove_outliers(raw_x, raw_y)

    norm_x, model.min_m, model.max_m = normalize_features(raw_x)

    # 3. Training (Using the Class method)
    print(f"🚀 Training on '{args.filename}' (LR: {args.lr}, Epochs: {args.epochs})")
    history = model.fit(norm_x, raw_y, args.lr, args.epochs, TOLERANCE)

    # 4. Evaluation
    preds = np.array([model.predict(x, verbose=False) for x in raw_x])
    model.metrics = calculate_metrics(raw_y, preds)
    
    print(f"\n{BOLD}📊 Results:{RESET} R²={model.metrics['r2_score']:.4f}, MAE={model.metrics['mae']:.2f}")

    # 5. Validation & Persistence
    out_file = os.path.join(MODEL_DIR, f"{os.path.splitext(args.filename)[0]}.json")
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    
    if validate_and_save(model, out_file):
        show_results(raw_x, raw_y, model, history)

if __name__ == "__main__":
    main()