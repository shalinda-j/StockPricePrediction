import trafilatura
import streamlit as st


def get_website_text_content(url: str) -> str:
    """
    This function takes a url and returns the main text content of the website.
    The text content is extracted using trafilatura and easier to understand.
    The results is not directly readable, better to be summarized by LLM before consume
    by the user.

    Args:
        url (str): The URL of the website to scrape

    Returns:
        str: The extracted text content of the website
    
    Example:
        MLB scores: https://www.mlb.com/scores/YYYY-MM-DD
    """
    try:
        # Send a request to the website
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        if text is None:
            return "No content could be extracted from this URL. The page might be protected, empty, or using a format that couldn't be parsed."
        return text
    except Exception as e:
        error_message = f"Error scraping website: {str(e)}"
        st.error(error_message)
        return error_message


def display_scraped_content(url: str, max_chars: int = 2000):
    """
    Display scraped content from a URL in a Streamlit app
    
    Args:
        url (str): The URL to scrape
        max_chars (int): Maximum characters to display
    """
    if not url:
        st.warning("Please enter a valid URL")
        return
    
    with st.spinner(f"Scraping content from {url}..."):
        content = get_website_text_content(url)
        
    if not content:
        st.error("Could not retrieve content from the provided URL")
        return
    
    # Display a portion of the content
    if len(content) > max_chars:
        display_content = content[:max_chars] + "..."
        st.info(f"Showing first {max_chars} characters of {len(content)} total characters")
    else:
        display_content = content
    
    st.subheader("Scraped Content")
    st.text_area("Content", display_content, height=300)
    
    # Allow downloading the full content
    st.download_button(
        label="Download Full Content", 
        data=content, 
        file_name="scraped_content.txt",
        mime="text/plain"
    )