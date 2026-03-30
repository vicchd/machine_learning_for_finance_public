"""
Fetches Walmart stock prices and Consumer Confidence Index from public sources.
"""

import yfinance as yf
import pandas as pd
import pandas_datareader.data as web

from pytrends.request import TrendReq

def fetch_walmart_stock(start="2012-01-01", end="2023-12-31"):
    """Fetch weekly Walmart (WMT) closing prices from Yahoo Finance."""
    wmt = yf.Ticker("WMT")
    weekly_prices = wmt.history(start=start, end=end, interval="1wk")[["Close"]]
    weekly_prices.index = pd.to_datetime(weekly_prices.index).tz_localize(None)
    return weekly_prices


def fetch_consumer_confidence(start="2012-09-01", end="2013-04-30"):
    """Fetch monthly Consumer Confidence Index from FRED and resample to weekly."""
    cci = web.DataReader("UMCSENT", "fred", start, end)
    cci.columns = ["consumer_confidence"]
    cci = cci.resample("W").first()
    return cci.interpolate(method="linear")

if __name__=="__main__":
    cci = fetch_consumer_confidence()
    print(cci)


KEYWORDS = ["Walmart", "grocery store", "discount", "inflation"]


def fetch_google_trends(
    keywords=None,
    start_date="2012-09-01",
    end_date="2013-04-30",
):
    """
    Fetch weekly Google Trends interest scores for the provided keywords.

    Parameters
    ----------
    keywords : list[str] | None
        List of Google Trends search terms to fetch. If None, default keywords
        are used.
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.

    Returns
    -------
    pandas.DataFrame
        Weekly DataFrame indexed by date with one column per keyword and a
        composite `consumer_search_index` column equal to the simple average
        of all keyword scores.
    """
    kw_list = keywords or KEYWORDS
    timeframe = f"{start_date} {end_date}"

    pytrends = TrendReq(hl="en-US", tz=0)
    pytrends.build_payload(kw_list=kw_list, timeframe=timeframe, geo="")
    trends = pytrends.interest_over_time()

    if trends.empty:
        columns = list(kw_list) + ["consumer_search_index"]
        empty_df = pd.DataFrame(columns=columns)
        empty_df.index.name = "date"
        return empty_df

    trends = trends.drop(columns=["isPartial"], errors="ignore")
    trends.index = pd.to_datetime(trends.index).tz_localize(None)
    trends = trends.resample("W").mean()
    trends["consumer_search_index"] = trends[kw_list].mean(axis=1)
    trends.index.name = "date"
    return trends
