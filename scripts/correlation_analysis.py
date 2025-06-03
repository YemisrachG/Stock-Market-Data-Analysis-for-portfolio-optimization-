import pandas as pd
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt 
import nltk
nltk.download('vader_lexicon', quiet=True)

def load_stock_data(data_dir='notebooks/data/', stock_files=None):
    stock_data = {}
    if stock_files is None:
        stock_files = [
            'AAPL_historical_data.csv', 'AMZN_historical_data.csv', 'GOOG_historical_data.csv',
            'META_historical_data.csv', 'MSFT_historical_data.csv', 'NVDA_historical_data.csv',
            'TSLA_historical_data.csv'
        ]
    for file_name in stock_files:
        ticker = file_name.split('_')[0]
        file_path = os.path.join(data_dir, file_name)
        try:
            df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
            df.columns = [col.capitalize() for col in df.columns]
            if 'Adj Close' in df.columns:
                df.rename(columns={'Adj Close': 'Close'}, inplace=True)
            elif 'Adjclose' in df.columns:
                df.rename(columns={'Adjclose': 'Close'}, inplace=True)
            stock_data[ticker] = df[['Close']]
            print(f"Loaded stock data for {ticker}: {df.shape}")
        except FileNotFoundError:
            print(f"Error: Stock data not found at {file_path}")
    print("\nStock data loading complete.")
    return stock_data

def load_news_data(news_file='raw_analyst_ratings.csv', data_dir='notebooks/data/'):
    news_file_path = os.path.join(data_dir, news_file)
    try:
        news_df = pd.read_csv(news_file_path)
        news_df['date'] = pd.to_datetime(news_df['date'], format='mixed', utc=True)
        print(f"\nLoaded news data: {news_df.shape}")
        print(news_df.head())
        return news_df
    except FileNotFoundError:
        print(f"Error: News data not found at {news_file_path}")
        return None

def analyze_sentiment(news_df):
    if news_df is not None:
        analyzer = SentimentIntensityAnalyzer()
        news_df['sentiment_score'] = news_df['headline'].apply(lambda headline: analyzer.polarity_scores(headline)['compound'])
        print("\nSentiment analysis complete. Sample with sentiment scores:")
        print(news_df[['date', 'headline', 'sentiment_score']].head())
    return news_df

def aggregate_daily_sentiment(news_df):
    if news_df is not None:
        daily_sentiment = news_df.groupby(news_df['date'].dt.date)['sentiment_score'].mean().reset_index()
        daily_sentiment.rename(columns={'date': 'Date', 'sentiment_score': 'avg_sentiment'}, inplace=True)
        daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])
        daily_sentiment.set_index('Date', inplace=True)
        print("\nAggregated daily sentiment:")
        print(daily_sentiment.head())
        return daily_sentiment
    return None

def calculate_correlation(stock_data, daily_sentiment):
    correlation_results = {}
    if daily_sentiment is not None:
        for ticker, df in stock_data.items():
            if df is not None:
                df['Daily_Return'] = df['Close'].pct_change()
                merged_df = pd.merge(df[['Daily_Return']], daily_sentiment, left_index=True, right_index=True, how='inner')
                correlation = merged_df['Daily_Return'].corr(merged_df['avg_sentiment'])
                correlation_results[ticker] = correlation
        print("\nCorrelation analysis complete:")
        print(correlation_results)
    return correlation_results

def visualize_correlation(correlation_results):
    """Prints the correlation results in tabular form."""
    correlation_df = pd.DataFrame(list(correlation_results.items()), columns=['Stock', 'Correlation'])
    print(correlation_df)
    print("\nCorrelation printing complete.")

if __name__ == "__main__":
    stock_data = load_stock_data()
    news_df = load_news_data()
    news_df = analyze_sentiment(news_df)
    daily_sentiment = aggregate_daily_sentiment(news_df)
    correlation_results = calculate_correlation(stock_data, daily_sentiment)
    visualize_correlation(correlation_results)