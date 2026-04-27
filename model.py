import json
import numpy as np
from typing import List

class TinyLinearRegression:
    def __init__(self):
        # Model state (Parameters)
        self.theta0 = 0.0
        self.theta1 = 0.0
        
        # Scaling info (For normalization)
        self.min_m = 0.0
        self.max_m = 1.0
        
        # Metadata
        self.metrics = {}

    def fit(self, x_norm: np.ndarray, y: np.ndarray, lr: float, epochs: int, tolerance: float = 1e-7) -> List[float]:
        """Performs Gradient Descent to update theta0 and theta1."""
        m = len(x_norm)
        cost_history = []

        for epoch in range(epochs):
            predictions = self.theta0 + (self.theta1 * x_norm)
            errors = predictions - y
            
            # Simultaneous Update
            self.theta0 -= lr * (1/m) * np.sum(errors)
            self.theta1 -= lr * (1/m) * np.sum(errors * x_norm)
            
            cost = np.sum(errors**2) / (2 * m)

            if lr <= 0:
                print(f"❌ Error: Learning rate must be a positive value.")
                return cost_history

            # Safety Guard: Divergence check - Check for numerical overflow
            if np.isinf(cost) or np.isnan(cost):
                print(f"\n❌ Error: Model diverged. Lower your learning rate.")
                return cost_history

            # Check for abnormal trend (Proactive)
            # If cost increases significantly compared to the start, it's diverging.
            if epoch > 0 and cost > cost_history[0] * 10:
                print(f"\n❌ Divergence detected: Cost is exploding!")
                return cost_history

            # Early Stopping
            if epoch > 0 and abs(cost_history[-1] - cost) < tolerance:
                print(f"✨ Early stopping at epoch {epoch} (Converged).")
                break
                
            cost_history.append(cost)
            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d}: Cost = {cost:.2f}")
        
        denom = self.max_m - self.min_m
        if denom != 0:
            self.theta0 = self.theta0 - (self.theta1 * (self.min_m / denom))
            self.theta1 = self.theta1 / denom
        
        return cost_history

    def predict(self, mileage: float, verbose: bool = False) -> float:
        """Calculates the predicted price for a single mileage value."""
        # Normalize input based on trained scale
        prediction = self.theta0 + (self.theta1 * mileage)
        
        if verbose and prediction < 0:
            print(f"⚠️  Note: The input mileage is far beyond the training data range.")
            print(f"The calculated value was negative ({prediction:.2f}), so the price is estimated as 0.")
        return max(0.0, float(prediction)) # Prices shouldn't be negative

    def save(self, filepath: str):
        """Exports model parameters and metrics to a JSON file."""
        data = {
            "theta0": float(self.theta0),
            "theta1": float(self.theta1),
            "min_m": float(self.min_m),
            "max_m": float(self.max_m),
            "metrics": self.metrics
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    def load(self, filepath: str) -> bool:
        """Imports model parameters from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.theta0 = data.get("theta0", 0.0)
                self.theta1 = data.get("theta1", 0.0)
                self.min_m = data.get("min_m", 0.0)
                self.max_m = data.get("max_m", 1.0)
                self.metrics = data.get("metrics", {})
                return True
        except Exception:
            return False