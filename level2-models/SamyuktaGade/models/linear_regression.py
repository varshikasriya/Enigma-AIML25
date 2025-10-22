# level2-models/SamyuktaGade/linear_regression.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class LinearRegressionGD:
    """Linear Regression with Gradient Descent for binary classification.
    
    Uses squared loss with optional L2 regularization to prevent overfitting.
    """
    lr: float = 0.01  # Reduced learning rate for stability
    max_epochs: int = 5000
    tol: float = 1e-6
    l2: float = 0.01  # Added regularization by default
    fit_intercept: bool = True
    standardize: bool = True
    random_state: Optional[int] = 42
    verbose: bool = False

    # Learned parameters
    w_: Optional[np.ndarray] = None
    x_mean_: Optional[np.ndarray] = None
    x_std_: Optional[np.ndarray] = None
    epochs_run_: int = 0
    converged_: bool = False
    last_loss_: float = np.inf
    train_loss_history_: List[float] = None
    val_loss_history_: List[float] = None

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add bias column to feature matrix."""
        if not self.fit_intercept:
            return X
        return np.c_[np.ones((X.shape[0], 1)), X]

    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        """Standardize features during training."""
        if not self.standardize:
            return X
        Xn = X.copy()
        start = 1 if self.fit_intercept else 0
        self.x_mean_ = Xn[:, start:].mean(axis=0)
        self.x_std_ = Xn[:, start:].std(axis=0)
        # Avoid division by zero
        self.x_std_[self.x_std_ == 0.0] = 1.0
        Xn[:, start:] = (Xn[:, start:] - self.x_mean_) / self.x_std_
        return Xn

    def _standardize_apply(self, X: np.ndarray) -> np.ndarray:
        """Apply standardization to new data."""
        if not self.standardize:
            return X
        Xn = X.copy()
        start = 1 if self.fit_intercept else 0
        Xn[:, start:] = (Xn[:, start:] - self.x_mean_) / self.x_std_
        return Xn

    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute MSE loss with optional L2 regularization."""
        y_hat = X @ self.w_
        err = y_hat - y
        loss = (err @ err) / len(y)
        
        if self.l2 > 0:
            start = 1 if self.fit_intercept else 0
            loss += self.l2 * (self.w_[start:] @ self.w_[start:])
        
        return float(loss)

    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None) -> "LinearRegressionGD":
        """Fit the linear regression model.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            X_val: Optional validation features
            y_val: Optional validation labels
        """
        rng = np.random.default_rng(self.random_state)
        
        # Prepare training data
        Xb = self._add_intercept(X.astype(float))
        Xb = self._standardize_fit(Xb)
        n, d = Xb.shape
        
        # Initialize weights with small random values
        self.w_ = rng.normal(scale=0.01, size=d)
        
        # Prepare validation data if provided
        Xb_val = None
        if X_val is not None and y_val is not None:
            Xb_val = self._add_intercept(X_val.astype(float))
            Xb_val = self._standardize_apply(Xb_val)
        
        # Training history
        self.train_loss_history_ = []
        self.val_loss_history_ = []
        prev_loss = np.inf

        for epoch in range(1, self.max_epochs + 1):
            # Forward pass
            y_hat = Xb @ self.w_
            err = y_hat - y
            
            # Compute gradient
            grad = (2.0 / n) * (Xb.T @ err)
            
            # Add L2 regularization gradient (don't regularize bias)
            if self.l2 > 0:
                start = 1 if self.fit_intercept else 0
                reg = np.zeros_like(self.w_)
                reg[start:] = 2.0 * self.l2 * self.w_[start:]
                grad += reg
            
            # Update weights
            self.w_ -= self.lr * grad
            
            # Compute losses
            train_loss = self._compute_loss(Xb, y)
            self.train_loss_history_.append(train_loss)
            
            if Xb_val is not None:
                val_loss = self._compute_loss(Xb_val, y_val)
                self.val_loss_history_.append(val_loss)
            
            # Check convergence
            self.epochs_run_ = epoch
            self.last_loss_ = train_loss
            
            if epoch % 100 == 0 and self.verbose:
                msg = f"Epoch {epoch}: train_loss={train_loss:.6f}"
                if Xb_val is not None:
                    msg += f", val_loss={val_loss:.6f}"
                print(msg)
            
            if abs(prev_loss - train_loss) < self.tol:
                self.converged_ = True
                if self.verbose:
                    print(f"Converged at epoch {epoch}")
                break
            
            prev_loss = train_loss
        
        return self

    def predict_continuous(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous values."""
        if self.w_ is None:
            raise RuntimeError("Model not fitted yet")
        Xb = self._add_intercept(X.astype(float))
        Xb = self._standardize_apply(Xb)
        return Xb @ self.w_

    def predict_classes(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary classes {0, 1}."""
        y_hat = self.predict_continuous(X)
        return (y_hat >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability-like scores (clipped to [0, 1])."""
        y_hat = self.predict_continuous(X)
        return np.clip(y_hat, 0.0, 1.0)