import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import requests
from textblob import TextBlob
import nltk
import time

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def get_news_api_key():
    """
    Get the News API key from secrets
    """
    try:
        return st.secrets.get("NEWSAPI_KEY", None)
    except Exception:
        # Return None if secrets aren't available at all
        return None

def get_news_for_ticker(ticker, days_back=7):
    """
    Fetch news articles related to a ticker symbol
    
    Args:
        ticker (str): Stock symbol to search for
        days_back (int): Number of days to look back for news
        
    Returns:
        list: List of articles with title, description, publish date and sentiment score
    """
    api_key = get_news_api_key()
    articles = []
    
    if api_key:
        try:
            # Format company name from ticker by removing extensions
            # For Sri Lankan stocks, extract just the company code
            if '.N' in ticker or '.X' in ticker:
                company_name = ticker.split('.')[0]
            else:
                company_name = ticker
                
            # For crypto, extract the currency name
            if '-' in ticker:
                parts = ticker.split('-')
                if parts[0] in ['BTC', 'ETH', 'XRP', 'LTC']:
                    crypto_names = {
                        'BTC': 'Bitcoin',
                        'ETH': 'Ethereum',
                        'XRP': 'Ripple',
                        'LTC': 'Litecoin'
                    }
                    company_name = crypto_names.get(parts[0], parts[0])
            
            # Calculate date range
            to_date = datetime.now().strftime('%Y-%m-%d')
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            # Use requests to directly access the News API
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": company_name,
                "from": from_date,
                "to": to_date,
                "language": "en",
                "sortBy": "relevancy",
                "pageSize": 20,
                "apiKey": api_key
            }
            
            response = requests.get(url, params=params)
            all_articles = response.json()
            
            if all_articles['status'] == 'ok' and all_articles['totalResults'] > 0:
                for article in all_articles['articles']:
                    # Perform sentiment analysis on title and description
                    title = article.get('title', '')
                    desc = article.get('description', '')
                    
                    # Combine title and description for better sentiment analysis
                    content = f"{title}. {desc}"
                    
                    # Get sentiment using TextBlob
                    sentiment = TextBlob(content).sentiment.polarity
                    
                    # Format publish date
                    publish_date = article.get('publishedAt', '')
                    if publish_date:
                        try:
                            # Convert to datetime and format
                            date_obj = datetime.strptime(publish_date, '%Y-%m-%dT%H:%M:%SZ')
                            publish_date = date_obj.strftime('%Y-%m-%d')
                        except:
                            # Keep original if parsing fails
                            pass
                    
                    articles.append({
                        'title': title,
                        'description': desc,
                        'published_at': publish_date,
                        'sentiment': sentiment,
                        'url': article.get('url', '')
                    })
                
                return articles
            else:
                return []
        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")
            return []
    else:
        # Return empty list if no API key is available
        return []

def get_sentiment_score(ticker, days_back=7):
    """
    Calculate the average sentiment score for a ticker
    
    Args:
        ticker (str): Stock symbol to get sentiment for
        days_back (int): Number of days to look back for news
        
    Returns:
        tuple: (average_sentiment, article_count, sentiment_by_day)
    """
    articles = get_news_for_ticker(ticker, days_back)
    
    if not articles:
        return 0, 0, {}
    
    # Calculate overall sentiment
    total_sentiment = sum(article['sentiment'] for article in articles)
    avg_sentiment = total_sentiment / len(articles)
    
    # Calculate sentiment by day
    sentiment_by_day = {}
    for article in articles:
        date = article['published_at']
        if date in sentiment_by_day:
            sentiment_by_day[date]['total'] += article['sentiment']
            sentiment_by_day[date]['count'] += 1
        else:
            sentiment_by_day[date] = {'total': article['sentiment'], 'count': 1}
    
    # Calculate average sentiment for each day
    for date in sentiment_by_day:
        sentiment_by_day[date] = sentiment_by_day[date]['total'] / sentiment_by_day[date]['count']
    
    return avg_sentiment, len(articles), sentiment_by_day

def get_sentiment_trend(ticker, days=30):
    """
    Generate a simulated sentiment trend when API key is not available
    
    Args:
        ticker (str): Stock symbol
        days (int): Number of days for the trend
        
    Returns:
        pandas.DataFrame: Dataframe with dates and sentiment scores
    """
    # Try to get real sentiment
    avg_sentiment, article_count, sentiment_by_day = get_sentiment_score(ticker, days_back=days)
    
    if article_count > 0:
        # Convert the sentiment_by_day dictionary to a DataFrame
        dates = []
        sentiments = []
        
        for date_str, sentiment in sentiment_by_day.items():
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d')
                dates.append(date)
                sentiments.append(sentiment)
            except:
                # Skip dates that can't be parsed
                pass
        
        if dates:
            sentiment_df = pd.DataFrame({
                'Date': dates,
                'Sentiment': sentiments
            })
            return sentiment_df
    
    # If news API isn't available, return None but don't show warning
    # The warning will be handled in display_news_sentiment
    return None

def display_news_sentiment(ticker, days_back=7):
    """
    Display news sentiment analysis in the Streamlit app
    
    Args:
        ticker (str): Stock symbol to analyze
        days_back (int): Number of days to look back for news
    """
    try:
        # Check if we can access secrets first
        api_key = get_news_api_key()
        if api_key is None:
            # Create fallback info when no API key is available
            st.warning("News API key not available. Please set up a NewsAPI key for news sentiment analysis.")
            st.markdown("""
            ### How to set up your News API key:
            
            1. Sign up for a free account at [NewsAPI.org](https://newsapi.org/register)
            2. Get your API key from your account dashboard
            3. Add your API key to the Replit Secrets by clicking on the lock icon in the sidebar
            4. Set the secret name to `NEWSAPI_KEY` and the value to your API key
            """)
            
            # Show example sentiment format for the selected ticker
            st.subheader(f"Example News Sentiment Format for {ticker}")
            st.info("This section will display real sentiment analysis once your API key is configured.")
            
            # Create sample metrics to show the user what they would see
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Sentiment", "0.00", "Neutral")
            with col2:
                st.metric("Articles Analyzed", "0")
                
            return
        
        articles = get_news_for_ticker(ticker, days_back)
        
        if not articles:
            st.info(f"No news articles found for {ticker} in the last {days_back} days.")
            return
        
        avg_sentiment, article_count, sentiment_by_day = get_sentiment_score(ticker, days_back)
        
        # Display sentiment summary
        st.subheader(f"News Sentiment Analysis for {ticker}")
        
        # Determine sentiment description
        sentiment_desc = "Neutral"
        if avg_sentiment > 0.2:
            sentiment_desc = "Strongly Positive"
        elif avg_sentiment > 0.05:
            sentiment_desc = "Positive"
        elif avg_sentiment < -0.2:
            sentiment_desc = "Strongly Negative"
        elif avg_sentiment < -0.05:
            sentiment_desc = "Negative"
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Sentiment", f"{avg_sentiment:.2f}", sentiment_desc)
        with col2:
            st.metric("Articles Analyzed", article_count)
        
        # If we have sentiment data for multiple days, show a trend
        if len(sentiment_by_day) > 1:
            st.subheader("Sentiment Trend")
            # Convert the sentiment_by_day dictionary to a DataFrame for plotting
            try:
                sentiment_df = pd.DataFrame([
                    {"Date": date, "Sentiment": score}
                    for date, score in sentiment_by_day.items()
                ])
                
                # Ensure the Date column is properly formatted
                sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"])
                sentiment_df = sentiment_df.sort_values("Date")
                
                # Plot the sentiment trend
                st.line_chart(sentiment_df.set_index("Date"))
            except Exception as trend_error:
                st.error(f"Error displaying sentiment trend: {trend_error}")
                
        # Display individual articles
        st.subheader("Recent News Articles")
        
        for i, article in enumerate(articles[:5]):  # Show only up to 5 articles
            try:
                with st.expander(f"{article['title']} ({article['published_at']})"):
                    st.write(article['description'])
                    
                    # Format sentiment - make sure sentiment is a float, not a Series
                    sentiment = float(article['sentiment']) if isinstance(article['sentiment'], (pd.Series, np.ndarray)) else article['sentiment']
                    sentiment_color = "gray"
                    if sentiment > 0.2:
                        sentiment_color = "green"
                    elif sentiment > 0.05:
                        sentiment_color = "lightgreen"
                    elif sentiment < -0.2:
                        sentiment_color = "red"
                    elif sentiment < -0.05:
                        sentiment_color = "lightcoral"
                    
                    st.markdown(f"**Sentiment Score:** <span style='color:{sentiment_color}'>{sentiment:.2f}</span>", unsafe_allow_html=True)
                    st.markdown(f"[Read full article]({article['url']})")
            except Exception as article_error:
                st.error(f"Error displaying article {i}: {article_error}")
                st.write(f"Article data: {type(article)}")
    except Exception as e:
        st.error(f"Error in news sentiment display: {str(e)}")
        # Try to provide more context about the error
        import traceback
        st.text(traceback.format_exc())
        st.info("To enable news sentiment analysis, please add a NEWSAPI_KEY in the Replit Secrets.")