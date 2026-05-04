"""
Deconvolution module for recovering underlying incidence from observed data.

This module implements a Richardson-Lucy style iterative deconvolution algorithm
to recover the underlying infection/hospitalization incidence from observed
case/hospitalization counts, accounting for delay distributions using Gamma distributions.
"""

import numpy as np
from scipy.stats import gamma
from scipy.signal import savgol_filter
from typing import Optional, Tuple, Dict
import warnings


class GammaDeconvolution:
    """
    Class for performing gamma-based deconvolution using Richardson-Lucy algorithm.

    This class deconvolves observed data (e.g., hospitalizations) to recover
    the underlying incidence (e.g., cases) by accounting for delay distributions.

    Attributes:
        num_times (int): Number of time points in the data
        delay_matrix (np.ndarray): Delay kernel matrix P (shape: T x T)
        normalization_factor (np.ndarray): Normalization factor q (shape: T,)
        mean_delay (float): Mean delay (a * b)
    """

    def __init__(
        self,
        num_times: int,
        shape_param: float,
        scale_param: float,
        smooth_window: int = 7,
        min_value: float = 0.01,
    ):
        """
        Initialize the deconvolution class.

        Args:
            num_times (int): Number of time points in the time series
            shape_param (float): Shape parameter 'a' for gamma distribution
            scale_param (float): Scale parameter 'b' for gamma distribution
            smooth_window (int): Window size for smoothing (default: 7)
            min_value (float): Minimum value to prevent zeros (default: 0.01)
        """
        self.num_times = num_times
        self.shape_param = shape_param
        self.scale_param = scale_param
        self.smooth_window = smooth_window
        self.min_value = min_value
        self.mean_delay = shape_param * scale_param

        # Construct delay matrix
        self.delay_matrix, self.normalization_factor = self._construct_delay_matrix()

    def _construct_delay_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct the delay kernel matrix using gamma distribution.

        The matrix P has shape (T, T) where P[i, j] represents the probability
        that an event on day j contributes to observed data on day i.

        Returns:
            tuple: (delay_matrix, normalization_factor)
                - delay_matrix: T x T matrix (transpose of gamma PDF matrix)
                - normalization_factor: T-length array for normalization
        """
        # Create delay matrix x where x[i, j] = j - i + 1 for j >= i
        x = np.zeros((self.num_times, self.num_times))
        for i in range(self.num_times):
            x[i, i:] = np.arange(1, self.num_times - i + 1)

        # Compute gamma PDF for each delay
        # Using scipy.stats.gamma: pdf(x, a, scale=b)
        y = gamma.pdf(x, self.shape_param, scale=self.scale_param)

        # Transpose to get kernel matrix P
        # P[j, i] = probability that event on day j contributes to observation on day i
        P = y.T

        # Normalization factor: sum across columns (delays) for each row
        q = np.sum(y, axis=1)

        return P, q

    def _smooth_data(self, data: np.ndarray) -> np.ndarray:
        """
        Smooth the input data using Savitzky-Golay filter.

        Args:
            data (np.ndarray): Input time series data

        Returns:
            np.ndarray: Smoothed data
        """
        if len(data) < self.smooth_window:
            # If data is too short, use simple moving average
            return np.convolve(
                data, np.ones(self.smooth_window) / self.smooth_window, mode="same"
            )

        # Use Savitzky-Golay filter (polynomial order 3)
        window_length = (
            self.smooth_window
            if self.smooth_window % 2 == 1
            else self.smooth_window - 1
        )
        polyorder = min(3, window_length - 1)

        try:
            smoothed = savgol_filter(data, window_length, polyorder)
        except ValueError:
            # Fallback to simple moving average if Savitzky-Golay fails
            smoothed = np.convolve(
                data, np.ones(self.smooth_window) / self.smooth_window, mode="same"
            )

        return smoothed

    def _remove_anomalies(self, data: np.ndarray) -> np.ndarray:
        """
        Remove anomalies from the data by replacing outliers with local averages.

        An outlier is defined as a value > 3 * (local average excluding that point).

        Args:
            data (np.ndarray): Input time series data

        Returns:
            np.ndarray: Data with anomalies removed
        """
        cleaned_data = data.copy()

        for t in range(len(data)):
            # Compute local average excluding current point
            start_idx = max(0, t - 3)
            end_idx = min(len(data), t + 4)
            local_values = np.concatenate([data[start_idx:t], data[t + 1 : end_idx]])

            if len(local_values) > 0:
                local_avg = np.mean(local_values)
                threshold = 3 * local_avg

                if data[t] > threshold:
                    cleaned_data[t] = local_avg

        return cleaned_data

    def _initialize_lambda(self, observed_data: np.ndarray) -> np.ndarray:
        """
        Initialize lambda (underlying incidence) from observed data.

        The initialization shifts the observed data back by the mean delay
        to account for the expected delay.

        Args:
            observed_data (np.ndarray): Observed time series data

        Returns:
            np.ndarray: Initialized lambda
        """
        shift = int(np.round(self.mean_delay))
        lambda_init = np.zeros(self.num_times)

        # Shift observed data back by mean delay
        if shift < len(observed_data):
            lambda_init[: self.num_times - shift] = observed_data[
                shift : self.num_times
            ]
        else:
            # If shift is too large, use the last values
            lambda_init[: len(observed_data)] = observed_data

        # Smooth and ensure minimum value
        lambda_init = self._smooth_data(lambda_init)
        lambda_init = np.maximum(lambda_init, self.min_value)

        return lambda_init

    def _compute_chi_squared(
        self, predicted: np.ndarray, observed: np.ndarray
    ) -> float:
        """
        Compute chi-squared statistic for goodness of fit.

        Args:
            predicted (np.ndarray): Predicted values
            observed (np.ndarray): Observed values

        Returns:
            float: Chi-squared statistic
        """
        # Avoid division by zero
        mask = predicted > 0
        if np.sum(mask) == 0:
            return np.inf

        chi_sq = np.sum(
            (predicted[mask] - observed[mask]) ** 2 / predicted[mask]
        ) / len(observed)
        return chi_sq

    def richardson_lucy_update(
        self,
        lambda_old: np.ndarray,
        observed_data: np.ndarray,
        delay_matrix: Optional[np.ndarray] = None,
        normalization_factor: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Perform one iteration of the Richardson-Lucy deconvolution update.

        The update equation is:
            λ_new(t) = (λ_old(t) / q(t)) · Σ_s (D_obs(s) / D_pred(s)) · P(s, t)

        where:
            - D_pred(s) = Σ_t P(s, t) · λ_old(t) is the predicted observations
            - q(t) is the normalization factor
            - P is the delay kernel matrix

        Args:
            lambda_old (np.ndarray): Current estimate of underlying incidence
            observed_data (np.ndarray): Observed time series data
            delay_matrix (np.ndarray, optional): Delay kernel matrix. If None, uses self.delay_matrix
            normalization_factor (np.ndarray, optional): Normalization factor. If None, uses self.normalization_factor

        Returns:
            tuple: (lambda_new, predicted_data, chi_squared)
                - lambda_new: Updated estimate of underlying incidence
                - predicted_data: Predicted observations D_pred
                - chi_squared: Chi-squared statistic for this iteration
        """
        if delay_matrix is None:
            delay_matrix = self.delay_matrix
        if normalization_factor is None:
            normalization_factor = self.normalization_factor

        # Forward prediction: D_pred = P · λ
        predicted_data = delay_matrix @ lambda_old

        # Avoid division by zero
        predicted_data = np.maximum(predicted_data, self.min_value)

        # Compute ratio: D_obs / D_pred
        ratio = observed_data / predicted_data

        # Backward projection: temp = ratio · P
        temp = ratio @ delay_matrix

        # Update with normalization: λ_new = (λ_old / q) .* temp
        lambda_new = (lambda_old / normalization_factor) * temp

        # Ensure non-negativity and minimum value
        lambda_new = np.maximum(lambda_new, self.min_value)

        # Compute chi-squared statistic
        chi_sq = self._compute_chi_squared(predicted_data, observed_data)

        return lambda_new, predicted_data, chi_sq

    def deconvolve(
        self,
        observed_data: np.ndarray,
        max_iterations: int = 1000,
        chi_threshold: float = 1.0,
        delta_chi_threshold: float = 1e-4,
        remove_anomalies: bool = True,
        initialize: bool = True,
        verbose: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Perform gamma deconvolution to recover underlying incidence.

        This is the main method that applies the iterative Richardson-Lucy
        algorithm to deconvolve observed data.

        Args:
            observed_data (np.ndarray): Observed time series data (e.g., hospitalizations)
            max_iterations (int): Maximum number of iterations (default: 1000)
            chi_threshold (float): Convergence threshold for chi-squared (default: 1.0)
            delta_chi_threshold (float): Convergence threshold for change in chi-squared (default: 1e-4)
            remove_anomalies (bool): Whether to remove anomalies from input data (default: True)
            initialize (bool): Whether to initialize lambda from observed data (default: True)
            verbose (bool): Whether to print convergence information (default: False)

        Returns:
            dict: Dictionary containing:
                - 'lambda': Deconvolved underlying incidence
                - 'predicted': Predicted observations (reconstructed from lambda)
                - 'chi_squared': Final chi-squared statistic
                - 'iterations': Number of iterations performed
                - 'converged': Whether the algorithm converged
        """
        # Preprocess observed data
        if remove_anomalies:
            observed_data = self._remove_anomalies(observed_data)

        observed_data = self._smooth_data(observed_data)
        observed_data = np.maximum(observed_data, self.min_value)

        # Ensure observed_data matches num_times
        if len(observed_data) != self.num_times:
            if len(observed_data) > self.num_times:
                observed_data = observed_data[: self.num_times]
            else:
                # Pad with zeros if shorter
                padded = np.zeros(self.num_times)
                padded[: len(observed_data)] = observed_data
                observed_data = padded

        # Initialize lambda
        if initialize:
            lambda_est = self._initialize_lambda(observed_data)
        else:
            lambda_est = np.ones(self.num_times) * np.mean(observed_data)
            lambda_est = np.maximum(lambda_est, self.min_value)

        # Initialize convergence tracking
        chi = np.inf
        delta_chi = np.inf
        iteration = 0

        # Iterative update
        while (
            (chi > chi_threshold)
            and (delta_chi > delta_chi_threshold)
            and (iteration < max_iterations)
        ):
            iteration += 1
            chi_old = chi

            # Perform Richardson-Lucy update
            lambda_est, predicted_data, chi = self.richardson_lucy_update(
                lambda_est, observed_data
            )

            # Compute change in chi-squared
            delta_chi = abs(chi - chi_old)

        converged = (chi <= chi_threshold) or (delta_chi <= delta_chi_threshold)

        if verbose:
            print(f"Deconvolution completed:")
            print(f"  Iterations: {iteration}")
            print(f"  Chi-squared: {chi:.6f}")
            print(f"  Delta chi-squared: {delta_chi:.6e}")
            print(f"  Converged: {converged}")

        if not converged and iteration >= max_iterations:
            warnings.warn(
                f"Deconvolution did not converge after {max_iterations} iterations. "
                f"Final chi-squared: {chi:.6f}"
            )

        return {
            "lambda": lambda_est,
            "predicted": predicted_data,
            "chi_squared": chi,
            "iterations": iteration,
            "converged": converged,
        }

    def reconstruct(self, lambda_est: np.ndarray) -> np.ndarray:
        """
        Reconstruct observed data from deconvolved lambda.

        This performs the forward operation: D_pred = P · λ

        Args:
            lambda_est (np.ndarray): Deconvolved underlying incidence

        Returns:
            np.ndarray: Reconstructed observations
        """
        return self.delay_matrix @ lambda_est


# Example usage:
#
# # Initialize deconvolution for hospitalization to case deconvolution
# # Typical delay: shape=4, scale=5 (mean delay = 20 days)
# deconv = GammaDeconvolution(
#     num_times=len(hospitalization_data),
#     shape_param=4.0,
#     scale_param=5.0,
#     smooth_window=7
# )
#
# # Perform deconvolution
# result = deconv.deconvolve(
#     observed_data=hospitalization_data,
#     max_iterations=1000,
#     chi_threshold=1.0,
#     delta_chi_threshold=1e-4,
#     verbose=True
# )
#
# # Access results
# cases_estimated = result['lambda']  # Deconvolved case incidence
# hosp_reconstructed = result['predicted']  # Reconstructed hospitalizations
#
# # Or use the Richardson-Lucy update method directly for custom iteration
# lambda_new, predicted, chi_sq = deconv.richardson_lucy_update(
#     lambda_old=lambda_old,
#     observed_data=hospitalization_data
# )
