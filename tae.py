"""
TAE functions
"""

from datetime import date, timedelta
from scipy import stats
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.stattools import acf

#########################
########## KS ###########
#########################

def neff_lag1(series):
    """
    Estimates the effective sample size (Neff) of a time series,
    accounting for autocorrelation at lag-1.

    The method uses the following approximation:
    Neff = N * (1 - rho1) / (1 + rho1)

    Parameters:
        series (array-like): Time series data.

    Returns:
        float: Effective sample size (Neff), with a minimum value of 1.
    """
    
    rho1 = acf(series, nlags=1, fft=True)[1]  # Lag-1 autocorrelation
    Neff = len(series) * (1 - rho1) / (1 + rho1)
    
    return max(Neff, 1)  # Ensure Neff is not less than 1

def ks_2samp(samp1, samp2):
    
    """
    Performs a two-sample KS test between two time series,
    adjusting the p-value to account for autocorrelation at lag-1.

    The adjustment uses the effective sample size (Neff) for each sample,
    based on their lag-1 autocorrelation, to correct the significance estimate.

    Parameters:
        samp1 (array-like): First sample or time series.
        samp2 (array-like): Second sample or time series.

    Returns:
        dict: Dictionary with the KS statistic and the adjusted p-value:
              - 'ks_stat': float, the KS test statistic.
              - 'pvalue': float, adjusted p-value accounting for autocorrelation.
    """
    ks_stat, p_value = stats.ks_2samp(samp1, samp2)  # Original KS test
    
    Neff1 = neff_lag1(samp1)
    Neff2 = neff_lag1(samp2)

    Neff_comb = (Neff1 * Neff2) / (Neff1 + Neff2)  # Combined effective sample size
    
    adjusted_p_value = stats.kstwo.sf(ks_stat, np.round(Neff_comb))
    
    return {"ks_stat": ks_stat, "pvalue": adjusted_p_value}

def rolling_ks_2samp(samp, Nsamp, window=20, step=5):
    
    """
    Performs a rolling Kolmogorov–Smirnov (KS) test between a fixed sample 
    and moving sub-samples of another time series.

    The test is repeated over rolling windows of size `window`, advancing 
    every 5 time steps. If the KS test p-value is below 0.05, the function 
    flags that period as one of statistical divergence (emergence).

    Parameters:
        samp (iris.cube.Cube): Fixed reference sample (e.g., control run),
                               with dimensions ("time", "latitude", "longitude").
        Nsamp (iris.cube.Cube): Time series to be tested against the fixed sample,
                                with dimension "time".
        window (int, optional): Size of the rolling window in time steps. 
                                Default is 20 (e.g., 20 years if monthly data).
        step (int, optional): Step size to move the rolling window. Default is 5.

    Returns:
        list: A list with two strings representing the start and end date of the 
              first period where the KS test p-value is consistently < 0.05, 
              indicating significant emergence. Returns an empty list if no emergence 
              is detected.
    """
    time_points = Nsamp.coord("time").points
    N = len(time_points)

    has_emerged = False
    emergence_period = []

    for i in range(0, N - window, step):
        
        # Get rolling sample
        rolling_samp = Nsamp[i:i + window]
        
        # Convert time coordinates to actual dates
        tmin = str(date(1850, 1, 1) + timedelta(days=int(time_points[i])))
        tmax = str(date(1850, 1, 1) + timedelta(days=int(time_points[i + window])))

        # Apply KS test between the fixed sample and the rolling sample
        result = ks_2samp(samp.data, rolling_samp.data)
        is_emerging = result["pvalue"] < 0.05

        if not has_emerged and is_emerging:
            emergence_period = [tmin, tmax]
            has_emerged = True
        elif has_emerged and not is_emerging:
            emergence_period = []
            has_emerged = False

    if len(emergence_period)>0:
        TAE = int(emergence_period[0][:4])
    else:
        TAE = np.nan
        
    return TAE

#########################
########## SN ###########
#########################

def get_LF(x, y, method, frac):
    
    """
    Extracts the low-frequency (LF) component from a time series 
    by fitting either a 4th-degree polynomial or applying LOWESS smoothing.
    
    Parameters:
        x (numpy.ndarray): Time axis or independent variable (e.g., years).
        y (numpy.ndarray): Time series to be smoothed or fitted.
        method (str): Smoothing method to use: 'polyfit' (default) or 'lowess'.
        frac (float): Fraction of data used in each local LOWESS regression (only used if method='lowess').
    
    Returns:
        numpy.ndarray: Low-frequency component of the input series.
    """
    
    if method=="polyfit":
        coeffs = np.polyfit(x, y, 4)
        LF_component = np.polyval(coeffs, x)  
    elif method=="lowess":
        LF_component = lowess(y, x, frac=frac, return_sorted=False)
    else:
        raise ValueError("Invalid method. Use 'polyfit' or 'lowess'.")
        
    return LF_component

def get_STD(x, y):
    
    """
    Estimates the standard deviation of natural variability in a time series,
    based on the residuals from a first-degree polynomial fit that represents the 
    low-frequency component.
    
    Parameters:
        x (numpy.ndarray): Years.
        y (numpy.ndarray): Time series data.
    
    Returns:
        float: Standard deviation of the high-frequency component.
    """

    coeffs = np.polyfit(x, y, 1)
    LF_component = np.polyval(coeffs, x)
    HF_component = y - LF_component
    natural_variability_STD = np.std(HF_component)
    
    return natural_variability_STD

def get_SN_ratio(x, y, x_preind, y_preind, method='polyfit', frac=0.1):

    """
    Calculates the SN as the ratio between the absolute value 
    of the LF component of a time series and the STD of natural
    variability estimated from a pre-industrial period.
    
    Parameters:
        x (numpy.ndarray): Time axis for the period of interest.
        y (numpy.ndarray): Time series data for the period of interest.
        x_preind (numpy.ndarray): Time axis for the pre-industrial/control period.
        y_preind (numpy.ndarray): Time series data for the pre-industrial/control period.
        method (str): Method used to extract the LF component. 
                      Options are 'polyfit' (default) or 'lowess'.
        frac (float): Smoothing parameter used in LOWESS. Represents the fraction of data 
                      used when estimating each y-value (only relevant if method='lowess').

    Returns:
        numpy.ndarray: SN time series.
    """
    
    LF = get_LF(x,y,method,frac)
    STD = get_STD(x_preind,y_preind)
    SNR = np.abs(LF) / STD
    
    return SNR

def get_tae_SN(sn, years, sn_threshold=1.0):
    """
    Detects the TAE as the first year when
    SN exceeds a threshold and remains above it 
    until the end of the time series.

    Parameters:
        sn (numpy.ndarray): Signal-to-noise ratio time series.
        years (numpy.ndarray): Array of years corresponding to SNR values.
        sn_threshold (float): Threshold above which emergence is considered.

    Returns:
        int or float: Year of emergence if detected and maintained, 
        otherwise np.nan.
    """
    for i in range(len(sn)):
        if sn[i] > sn_threshold:
            if np.all(sn[i:] > sn_threshold):
                return years[i]
    return np.nan
