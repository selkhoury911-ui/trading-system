"""
pysimfin.py — Python API Wrapper for SimFin (Part 2.1)
=======================================================
Provides an object-oriented interface to the SimFin Data API v3.
Returns data as Pandas DataFrames, enforces rate limiting, and raises
descriptive exceptions so the Streamlit app can handle errors cleanly.

Usage
-----
    from pysimfin import PySimFin

    client = PySimFin(api_key="your-api-key-here")

    # Fetch 90 days of AAPL share prices
    prices = client.get_share_prices("AAPL", start="2024-01-01", end="2024-12-31")
    print(prices.head())

    # Fetch income statement
    income = client.get_financial_statement("AAPL", "2023-01-01", "2024-01-01", statement="pl")
    print(income.head())

API docs: https://simfin.readme.io/reference/getting-started-1
Rate limit: max 2 requests/second on the free tier.
"""

import time
from typing import Optional

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class SimFinError(Exception):
    """Base exception for all PySimFin errors."""


class SimFinAuthError(SimFinError):
    """Raised when the API key is invalid or missing."""


class SimFinRateLimitError(SimFinError):
    """Raised when SimFin's rate limit is exceeded (HTTP 429 or 403)."""


class SimFinNotFoundError(SimFinError):
    """Raised when a ticker or resource cannot be found (HTTP 404 or empty result)."""


class SimFinConnectionError(SimFinError):
    """Raised for network timeouts or connection failures."""


# ---------------------------------------------------------------------------
# PySimFin class
# ---------------------------------------------------------------------------

class PySimFin:
    """
    Python wrapper for the SimFin Data API v3.

    Handles authentication, rate limiting, HTTP error mapping, and
    JSON-to-DataFrame conversion in one place so callers never have
    to deal with raw requests or response parsing.

    Parameters
    ----------
    api_key : str
        Your SimFin API key. Get one free at https://simfin.com/

    Attributes
    ----------
    BASE_URL : str
        SimFin API v3 base URL.
    RATE_LIMIT_DELAY : float
        Minimum seconds between requests (0.55 s keeps us safely under
        the free-tier limit of 2 req/s).

    Examples
    --------
    >>> client = PySimFin(api_key="your-key")
    >>> df = client.get_share_prices("AAPL", "2024-01-01", "2024-06-01")
    >>> print(df.columns.tolist())
    """

    BASE_URL         = "https://backend.simfin.com/api/v3"
    RATE_LIMIT_DELAY = 0.55   # seconds — just above 0.5 to stay safely under 2 req/s

    def __init__(self, api_key: str):
        """
        Initialise the PySimFin client.

        Parameters
        ----------
        api_key : str
            SimFin API key (required). Never hardcode this — load it from
            environment variables or Streamlit secrets.

        Raises
        ------
        ValueError
            If api_key is empty or not a string.
        """
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError(
                "A valid SimFin API key string must be provided.\n"
                "Get your free key at https://simfin.com/"
            )

        self.api_key = api_key.strip()
        self.headers = {
            "Authorization": f"api-key {self.api_key}",
            "Accept":        "application/json",
        }
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        """
        Sleep if necessary to stay within SimFin's 2-request-per-second limit.

        Called automatically before every API request — no need to call this
        manually from outside the class.
        """
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def _get(self, endpoint: str, params: Optional[dict] = None) -> list:
        """
        Make a rate-limited GET request and return the parsed JSON body.

        Parameters
        ----------
        endpoint : str
            API path, e.g. "/companies/prices/compact".
        params : dict, optional
            Query-string parameters.

        Returns
        -------
        list or dict
            Parsed JSON response from SimFin.

        Raises
        ------
        SimFinAuthError
            HTTP 401 — invalid API key.
        SimFinRateLimitError
            HTTP 429 or 403 — too many requests.
        SimFinNotFoundError
            HTTP 404 — endpoint or resource not found.
        SimFinError
            Any other non-2xx status code.
        SimFinConnectionError
            Network timeout or DNS failure.
        """
        self._throttle()

        url = f"{self.BASE_URL}{endpoint}"

        try:
            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=30,
            )
        except requests.exceptions.Timeout:
            raise SimFinConnectionError(
                "Request to SimFin timed out after 30 s. "
                "The server may be slow — try again in a moment."
            )
        except requests.exceptions.ConnectionError as exc:
            raise SimFinConnectionError(
                f"Could not reach SimFin ({exc}). "
                "Check your internet connection."
            )

        # Map HTTP status codes to descriptive exceptions
        if response.status_code == 401:
            raise SimFinAuthError(
                "Authentication failed (HTTP 401). "
                "Check that your SIMFIN_API_KEY is correct."
            )
        if response.status_code in (429, 403):
            raise SimFinRateLimitError(
                f"Rate limit exceeded (HTTP {response.status_code}). "
                "SimFin free tier allows max 2 requests per second. "
                "Wait a moment and try again."
            )
        if response.status_code == 404:
            raise SimFinNotFoundError(
                f"Resource not found (HTTP 404): {url}. "
                "Check that the ticker symbol exists in SimFin."
            )
        if not response.ok:
            raise SimFinError(
                f"SimFin API error (HTTP {response.status_code}): {response.text}"
            )

        return response.json()

    @staticmethod
    def _to_dataframe(data: list) -> pd.DataFrame:
        """
        Convert SimFin's compact JSON format into a Pandas DataFrame.

        SimFin returns data in this shape:
            [{"columns": ["col1", "col2", ...], "data": [[v1, v2], [v1, v2], ...]}]

        Parameters
        ----------
        data : list
            Raw JSON from SimFin.

        Returns
        -------
        pd.DataFrame
            Tabular representation of the response. Empty if data is empty.
        """
        if not data:
            return pd.DataFrame()

        entry = data[0] if isinstance(data, list) else data

        if "columns" in entry and "data" in entry:
            df = pd.DataFrame(entry["data"], columns=entry["columns"])
        else:
            # Fallback for non-compact responses
            df = pd.DataFrame(data if isinstance(data, list) else [data])

        # Auto-parse any Date column
        for col in df.columns:
            if col.lower() == "date":
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass

        return df

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def get_share_prices(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Retrieve daily share prices for a ticker over a date range.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol, e.g. "AAPL", "MSFT".
        start : str
            Start date in "YYYY-MM-DD" format.
        end : str
            End date in "YYYY-MM-DD" format.

        Returns
        -------
        pd.DataFrame
            Columns typically include: Date, Opening Price, Highest Price,
            Lowest Price, Last Closing Price, Trading Volume, Adj. Closing Price.
            The Streamlit Go Live page renames these to Open/High/Low/Close/Volume
            before passing to the ETL transform() function.

        Raises
        ------
        ValueError
            If ticker, start, or end are missing.
        SimFinNotFoundError
            If no data is returned for the given ticker and date range.
        SimFinAuthError, SimFinRateLimitError, SimFinError
            See _get() for details.

        Example
        -------
        >>> df = client.get_share_prices("AAPL", "2024-01-01", "2024-03-31")
        >>> print(df[["Date", "Last Closing Price"]].tail())
        """
        if not ticker:
            raise ValueError("ticker cannot be empty.")
        if not start or not end:
            raise ValueError("Both start and end dates must be provided (YYYY-MM-DD).")

        params = {
            "ticker": ticker.upper().strip(),
            "start":  start,
            "end":    end,
        }

        data = self._get("/companies/prices/compact", params=params)
        df   = self._to_dataframe(data)

        if df.empty:
            raise SimFinNotFoundError(
                f"No share price data found for '{ticker}' between {start} and {end}. "
                "Make sure the ticker exists on SimFin and the date range is valid."
            )

        # Sort chronologically
        if "Date" in df.columns:
            df = df.sort_values("Date").reset_index(drop=True)

        return df

    def get_financial_statement(
        self,
        ticker: str,
        start: str,
        end: str,
        statement: str = "pl",
    ) -> pd.DataFrame:
        """
        Retrieve a financial statement for a ticker over a date range.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol.
        start : str
            Start date in "YYYY-MM-DD" format.
        end : str
            End date in "YYYY-MM-DD" format.
        statement : str, optional
            Statement type:
            - "pl"  — Profit & Loss / Income Statement (default)
            - "bs"  — Balance Sheet
            - "cf"  — Cash Flow Statement

        Returns
        -------
        pd.DataFrame
            Financial statement data with one row per reporting period.

        Raises
        ------
        ValueError
            If an invalid statement type is provided.
        SimFinNotFoundError
            If no data is returned.

        Example
        -------
        >>> df = client.get_financial_statement("AAPL", "2022-01-01", "2024-01-01", "pl")
        >>> print(df.head())
        """
        valid = {"pl", "bs", "cf"}
        if statement not in valid:
            raise ValueError(
                f"Invalid statement type '{statement}'. "
                f"Choose from: {sorted(valid)}"
            )
        if not ticker:
            raise ValueError("ticker cannot be empty.")
        if not start or not end:
            raise ValueError("Both start and end dates are required.")

        params = {
            "ticker":    ticker.upper().strip(),
            "start":     start,
            "end":       end,
            "statement": statement,
        }

        data = self._get(
            f"/companies/statements/compact?statement={statement}",
            params={k: v for k, v in params.items() if k != "statement"},
        )
        df = self._to_dataframe(data)

        if df.empty:
            raise SimFinNotFoundError(
                f"No '{statement}' statement data found for '{ticker}' "
                f"between {start} and {end}."
            )

        return df

    def get_company_info(self, ticker: str) -> pd.DataFrame:
        """
        Retrieve general company information (name, industry, employees, etc.).

        Parameters
        ----------
        ticker : str
            Stock ticker symbol.

        Returns
        -------
        pd.DataFrame
            One-row DataFrame with company metadata.

        Example
        -------
        >>> info = client.get_company_info("AAPL")
        >>> print(info[["Company Name", "Industry", "Number of Employees"]])
        """
        if not ticker:
            raise ValueError("ticker cannot be empty.")

        params = {"ticker": ticker.upper().strip()}
        data   = self._get("/companies/general/compact", params=params)
        df     = self._to_dataframe(data)

        if df.empty:
            raise SimFinNotFoundError(
                f"No company information found for ticker '{ticker}'."
            )

        return df


# ---------------------------------------------------------------------------
# Quick smoke test — run with: python src/pysimfin.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    print("PySimFin — API Wrapper (Part 2.1)")
    print("=" * 40)

    api_key = os.getenv("SIMFIN_API_KEY", "")
    if not api_key:
        print("Set the SIMFIN_API_KEY environment variable to run a live test.")
        print("  Windows : set SIMFIN_API_KEY=your-key-here")
        print("  Mac/Linux: export SIMFIN_API_KEY=your-key-here")
    else:
        client = PySimFin(api_key=api_key)

        print("\nFetching AAPL share prices (last 30 days)...")
        try:
            df = client.get_share_prices("AAPL", start="2024-11-01", end="2024-11-30")
            print(df.head())
            print(f"\nColumns returned: {df.columns.tolist()}")
        except SimFinError as e:
            print(f"Error: {e}")

        print("\nFetching AAPL income statement...")
        try:
            df_pl = client.get_financial_statement(
                "AAPL", start="2022-01-01", end="2024-01-01", statement="pl"
            )
            print(df_pl.head())
        except SimFinError as e:
            print(f"Error: {e}")
