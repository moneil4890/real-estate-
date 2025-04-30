import streamlit as st
import openai
from openai import OpenAI
import pandas as pd
import io
import base64
from PIL import Image
import PyPDF2
import re
import requests
from bs4 import BeautifulSoup
import json
import os

# App configuration
st.set_page_config(page_title="BURO - Your Listing Optimizer", layout="wide")

# Initialize session state variables if they don't exist
if 'seo_listing' not in st.session_state:
    st.session_state.seo_listing = ""
if 'price_report' not in st.session_state:
    st.session_state.price_report = ""
if 'accuracy_report' not in st.session_state:
    st.session_state.accuracy_report = ""
if 'property_data' not in st.session_state:
    st.session_state.property_data = {}
if 'uploaded_documents' not in st.session_state:
    st.session_state.uploaded_documents = None
if 'client' not in st.session_state:
    st.session_state.client = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'title_tag' not in st.session_state:
    st.session_state.title_tag = ""
if 'meta_description' not in st.session_state:
    st.session_state.meta_description = ""
if 'keyword_analysis' not in st.session_state:
    st.session_state.keyword_analysis = ""

# Custom CSS to match the UI in the image
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .buro-header {
        color: #4a4a4a;
        font-size: 38px;  /* Increased from 28px to make it larger */
        font-weight: 600;
        margin-left: 10px;
        margin-top: 5px;
    }
    .listing-optimizer-header {
        color: #555;
        font-size: 16px;
        text-transform: uppercase;
        font-weight: 500;
        margin-left: 10px;
        margin-bottom: 20px;
    }
    .section-header {
        color: #666;
        font-size: 14px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stExpander {
        border: 1px solid #eaeaea;
        border-radius: 4px;
        margin-bottom: 10px;
        background-color: white;
    }
    .highlight {
        background-color: #f0f7ff;
        padding: 15px;
        border-radius: 5px;
        border-left: 3px solid #1e88e5;
        margin-bottom: 15px;
    }
    .meta-box {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
        margin-bottom: 15px;
    }
    .chat-message {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 3px solid #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 3px solid #8bc34a;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to create a download link for text content
def get_download_link(content, filename, text):
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href

# AI Agent functions - Automatically initialize the OpenAI client
def init_openai_client():
    # Use a default API key - in a production app, you'd want to handle this more securely
    default_api_key = "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"  # Replace with your key or implement secure storage
    
    # Initialize the OpenAI client with the API key
    st.session_state.client = OpenAI(api_key=default_api_key)
    return True

def critic_agent(property_data, current_listing="", documents=None):
    """Coordinates between different AI agents"""
    # First, get the SEO optimized listing
    seo_listing, title_tag, meta_description, keyword_analysis = writer_agent(property_data, current_listing)
    
    # Then, get the listing reviewed
    reviewed_listing = reviewer_agent(seo_listing, property_data)
    
    # Get real estate market data for the area
    market_data = get_real_estate_data(property_data)
    
    # Then, get pricing recommendation
    price_report = pricing_agent(property_data, reviewed_listing, market_data)
    
    # Finally, check accuracy if documents are provided
    accuracy_report = ""
    if documents is not None:
        accuracy_report = accuracy_agent(property_data, documents)
    
    return reviewed_listing, title_tag, meta_description, keyword_analysis, price_report, accuracy_report

def writer_agent(property_data, current_listing=""):
    """Creates SEO optimized property listing with title tag and meta description"""
    if not st.session_state.client:
        init_openai_client()
    
    # Convert property data to a formatted string
    property_str = "\n".join([f"{k}: {v}" for k, v in property_data.items() if v])
    
    # Include current listing if provided
    current_listing_text = ""
    if current_listing:
        current_listing_text = f"\nCURRENT LISTING TO OPTIMIZE:\n{current_listing}\n"
    
    prompt = f"""
    You are an experienced real estate agent. Write a well-crafted real estate listing 
    that is optimized for SEO based on the following property details:
    
    {property_str}
    {current_listing_text}
    
    Your response should include four sections:
    
    1. TITLE TAG:
    Create a compelling title tag (50-60 characters) that includes property type, location, and a key feature
    or call to action. This will appear in search results and browser tabs.
    
    2. META DESCRIPTION:
    Write a concise meta description (150-160 characters) summarizing the property's key features and location
    to encourage clicks from search results. This should contain important keywords.
    
    3. KEYWORD ANALYSIS:
    List 5-10 primary keywords for this property based on:
    - Location keywords (city, neighborhood, landmarks)
    - Property type keywords (house, condo, townhouse, etc.)
    - Feature keywords (bedrooms, bathrooms, pool, renovated, etc.)
    - Target audience keywords (luxury, first-time buyers, investors)
    
    4. OPTIMIZED LISTING:
    Write a 300-400 word property description that:
    - Naturally incorporates the keywords identified above
    - Uses vivid, engaging language to describe the property
    - Highlights unique selling points
    - Is organized in clear paragraphs (introduction, property details, neighborhood, call to action)
    - Targets the likely buyer persona for this property type and location
    
    Format each section with clear headings.
    """
    
    try:
        response = st.session_state.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an experienced real estate agent specializing in SEO-optimized property listings."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        
        # Extract sections from the response
        title_tag_match = re.search(r"TITLE TAG:(.+?)(?=META DESCRIPTION:|$)", content, re.DOTALL)
        meta_description_match = re.search(r"META DESCRIPTION:(.+?)(?=KEYWORD ANALYSIS:|$)", content, re.DOTALL)
        keyword_analysis_match = re.search(r"KEYWORD ANALYSIS:(.+?)(?=OPTIMIZED LISTING:|$)", content, re.DOTALL)
        optimized_listing_match = re.search(r"OPTIMIZED LISTING:(.+?)$", content, re.DOTALL)
        
        title_tag = title_tag_match.group(1).strip() if title_tag_match else ""
        meta_description = meta_description_match.group(1).strip() if meta_description_match else ""
        keyword_analysis = keyword_analysis_match.group(1).strip() if keyword_analysis_match else ""
        optimized_listing = optimized_listing_match.group(1).strip() if optimized_listing_match else content
        
        return optimized_listing, title_tag, meta_description, keyword_analysis
    except Exception as e:
        return f"Error generating listing: {str(e)}", "", "", ""

def reviewer_agent(listing, property_data):
    """Reviews the SEO optimized listing for quality and suggestions"""
    if not st.session_state.client:
        init_openai_client()
    
    # Convert property data to a formatted string
    property_str = "\n".join([f"{k}: {v}" for k, v in property_data.items() if v])
    
    prompt = f"""
    You are an experienced real estate agent who reviews property listings for SEO optimization
    and marketing effectiveness. Review the following property listing and improve it if needed:
    
    PROPERTY DETAILS:
    {property_str}
    
    CURRENT LISTING:
    {listing}
    
    Evaluate the listing based on:
    1. SEO optimization (keywords for location, property type, features)
    2. Compelling headline and description
    3. Highlighting of unique selling points
    4. Language appropriate for the target buyer persona
    5. Overall structure and readability
    6. Call to action
    
    If it needs improvement, provide a revised version. If it's already excellent, return it as is.
    Simply start with an introduction paragraph, and avoid labels like "REVISED LISTING:" or "Property Details".
    """
    
    try:
        response = st.session_state.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an experienced real estate agent who reviews listings for quality, SEO, and target buyer appeal."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return listing  # Return original if error

def get_real_estate_data(property_data):
    """
    Function to get real estate data from online sources using GPT-4's browsing capability
    """
    if not st.session_state.client:
        init_openai_client()
    
    city = property_data.get('City', '')
    state = property_data.get('State', '')
    zip_code = property_data.get('ZIP Code', '')
    bedrooms = property_data.get('Bedrooms', '')
    bathrooms = property_data.get('Bathrooms', '')
    sq_ft = property_data.get('Square Footage', '')
    neighborhood = property_data.get('Neighborhood', '')
    
    location_info = f"{city}, {state} {zip_code}"
    if not city and not zip_code:
        return "Insufficient location data provided."
    
    prompt = f"""
    I need you to act as a real estate market researcher and find current pricing data for:
    
    Location: {location_info}
    Neighborhood: {neighborhood}
    Property Type: {property_data.get('Property Type', '')}
    Bedrooms: {bedrooms}
    Bathrooms: {bathrooms}
    Square Footage: {sq_ft}
    
    Using your browsing capabilities, please find and summarize:
    1. Recent sales (last 3-6 months) of similar properties in this area
    2. Currently active listings of similar properties
    3. Average price per square foot in this area
    4. Market trends for this location (increasing, decreasing, stable)
    5. Days on market statistics
    
    Focus on websites like Zillow, Redfin, and Realtor.com to gather this data.
    Format your response as structured market research that could be used by a pricing agent.
    """
    
    try:
        response = st.session_state.client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o which has web browsing capabilities
            messages=[
                {"role": "system", "content": "You are a real estate market researcher with access to current property listings and sales data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        # Fallback with simulated data
        return f"""
        ## Market Research Data (Simulated)
        
        Due to current limitations in accessing real-time web data, the following market analysis is based on typical patterns for similar properties:
        
        ### Recent Sales Summary
        - Average sale price for {bedrooms} bed/{bathrooms} bath homes in {location_info}: $350,000 - $425,000
        - Price per square foot range: $175 - $225
        - Typical days on market: 15-30 days
        
        ### Current Market Listings
        - Active listings in similar range: 12-15 properties
        - Price range: $360,000 - $450,000
        - Average time on market: 21 days
        
        ### Market Trends
        - Market appears to be moderately competitive
        - Prices trending upward approximately 5% annually
        - Inventory levels are below historical averages
        
        Note: This is simulated data. For accurate current market data, please check Zillow, Redfin, or Realtor.com for {location_info}.
        """

def pricing_agent(property_data, listing, market_data):
    """Generates pricing recommendation based on property details and market data"""
    if not st.session_state.client:
        init_openai_client()
    
    # Convert property data to a formatted string
    property_str = "\n".join([f"{k}: {v}" for k, v in property_data.items() if v])
    
    prompt = f"""
    You are an experienced real estate agent who determines listing prices for properties.
    Based on the following property details, listing, and market research data, provide a detailed pricing recommendation report:
    
    PROPERTY DETAILS:
    {property_str}
    
    PROPERTY LISTING:
    {listing}
    
    MARKET RESEARCH DATA:
    {market_data}
    
    Your pricing recommendation report should include:
    1. A recommended listing price with justification
    2. Analysis of comparable properties (recently sold and currently listed)
    3. Price per square foot analysis
    4. Market trend considerations
    5. Pricing strategy recommendations (e.g., pricing slightly below market to generate multiple offers)
    6. A suggested price range if applicable
    
    Format this as a professional report with clear sections and headings.
    """
    
    try:
        response = st.session_state.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an experienced real estate agent specializing in property pricing."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating pricing report: {str(e)}"

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF document"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def accuracy_agent(property_data, document_text):
    """Compares user-entered data with official documents for accuracy"""
    if not st.session_state.client:
        init_openai_client()
    
    # Convert property data to a formatted string
    property_str = "\n".join([f"{k}: {v}" for k, v in property_data.items() if v])
    
    prompt = f"""
    You are an Accuracy Reviewer Agent responsible for verifying property details by comparing 
    user-entered data with official records. Review the following information:
    
    USER-ENTERED PROPERTY DATA:
    {property_str}
    
    OFFICIAL DOCUMENT TEXT:
    {document_text}
    
    Your task is to:
    1. Compare key attributes (square footage, lot size, bedrooms, bathrooms, year built, etc.)
    2. Identify any discrepancies between user data and official records
    3. Create a verification report that includes:
       - Summary of comparison
       - List of confirmed data points
       - List of discrepancies found
       - Suggested corrections
       - Final accuracy rating (percentage of accuracy)
    
    Format this as a clear, professional report with sections.
    """
    
    try:
        response = st.session_state.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an Accuracy Reviewer Agent specializing in property detail verification."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error performing accuracy check: {str(e)}"

def chat_with_agent(user_message):
    """Interactive chat with the AI agent to gather more information"""
    if not st.session_state.client:
        init_openai_client()
    
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_message})
    
    # Convert property data to context
    property_context = "\n".join([f"{k}: {v}" for k, v in st.session_state.property_data.items() if v])
    
    # Prepare the conversation history for the API call
    messages = [
        {"role": "system", "content": "You are an experienced real estate agent helping to create an SEO optimized property listing. Ask for specific details about the property that would help create a better listing. Focus on unique selling points, neighborhood benefits, and details that would appeal to the target buyer persona. Keep your responses concise and focused on gathering helpful information."},
        {"role": "user", "content": f"I'm trying to create an SEO optimized listing for a property. Here's what I know so far:\n{property_context}"}
    ]
    
    # Add the conversation history
    for message in st.session_state.chat_history:
        messages.append({"role": message["role"], "content": message["content"]})
    
    try:
        response = st.session_state.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        assistant_message = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_message})
        return assistant_message
    except Exception as e:
        return f"Error in chat: {str(e)}"


# Main UI

st.markdown('<div class="buro-header">BURO</div>', unsafe_allow_html=True)
st.markdown('<div class="listing-optimizer-header">YOUR LISTING OPTIMIZER</div>', unsafe_allow_html=True)

st.markdown("Enter property details and upload official documents. You'll get an SEO optimized listing with Title Tag & Meta Description, Listing Price Report, and Listing Accuracy Report")
# Initialize OpenAI client
init_openai_client()

# Create tabs for app modes
tab_new, tab_existing = st.tabs(["Create New Listing", "Optimize Existing Listing"])

with tab_new:
    # Create expandable sections for property details input
    with st.expander("PROPERTY BASICS", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.property_data['Property Type'] = st.selectbox(
                "Property Type", 
                ["Single Family Home", "Townhouse", "Condo", "Multi-Family", "Land", "Luxury Home", "Vacation Property"]
            )
            st.session_state.property_data['Bedrooms'] = st.number_input("Bedrooms", min_value=0, max_value=20, value=3)
            st.session_state.property_data['Bathrooms'] = st.number_input("Bathrooms", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
        with col2:
            st.session_state.property_data['Square Footage'] = st.number_input("Square Footage", min_value=0, value=2000)
            st.session_state.property_data['Year Built'] = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
            
            # Add both options for lot size
            lot_size_type = st.radio("Lot Size Measurement", ("acres", "square feet"))
            if lot_size_type == "acres":
                st.session_state.property_data['Lot Size'] = st.text_input("Lot Size (acres)", value="0.25")
            else:
                st.session_state.property_data['Lot Size'] = st.text_input("Lot Size (square feet)", value="10890")

    with st.expander("LOCATION DETAILS"):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.property_data['Street Address'] = st.text_input("Street Address")
            st.session_state.property_data['City'] = st.text_input("City")
            st.session_state.property_data['State'] = st.text_input("State")
        with col2:
            st.session_state.property_data['ZIP Code'] = st.text_input("ZIP Code")
            st.session_state.property_data['Neighborhood'] = st.text_input("Neighborhood")
            st.session_state.property_data['School District'] = st.text_input("School District")
            st.session_state.property_data['Nearby Landmarks'] = st.text_input("Nearby Landmarks")

    with st.expander("INTERIOR DETAILS"):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.property_data['Kitchen Features'] = st.text_area("Kitchen Features")
            st.session_state.property_data['Living Areas'] = st.text_area("Living Areas")
            st.session_state.property_data['Flooring'] = st.text_input("Flooring")
        with col2:
            st.session_state.property_data['Heating/Cooling'] = st.text_input("Heating/Cooling")
            st.session_state.property_data['Special Interior Features'] = st.text_area("Special Interior Features")
            st.session_state.property_data['Smart Home Features'] = st.text_input("Smart Home Features")

    with st.expander("EXTERIOR DETAILS"):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.property_data['Exterior Material'] = st.text_input("Exterior Material")
            st.session_state.property_data['Roof'] = st.text_input("Roof")
            st.session_state.property_data['Garage'] = st.text_input("Garage")
        with col2:
            st.session_state.property_data['Outdoor Features'] = st.text_area("Outdoor Features")
            st.session_state.property_data['Pool/Spa'] = st.text_input("Pool/Spa")
            st.session_state.property_data['Views'] = st.text_input("Views")

    with st.expander("TARGET BUYER & ADDITIONAL DETAILS"):
        col1, col2 = st.columns(2)
        with col1:
            # Modified to allow custom target buyer
            target_buyer_options = ["First-time homebuyers", "Luxury buyers", "Investors", "Retirees/Downsizers", 
                 "Growing families", "Urban professionals", "Vacation/Second home buyers", "Custom"]
            target_buyer_selection = st.selectbox("Target Buyer", target_buyer_options)
            
            if target_buyer_selection == "Custom":
                st.session_state.property_data['Target Buyer'] = st.text_input("Enter Custom Target Buyer")
            else:
                st.session_state.property_data['Target Buyer'] = target_buyer_selection
                
            st.session_state.property_data['HOA'] = st.text_input("HOA Information")
            st.session_state.property_data['Taxes'] = st.text_input("Annual Taxes")
        with col2:
            st.session_state.property_data['Utilities'] = st.text_input("Utilities")
            st.session_state.property_data['Special Notes'] = st.text_area("Special Notes")
            st.session_state.property_data['Selling Points'] = st.text_area("Key Selling Points")

    with st.expander("ACCURACY CHECKER"):
        st.write("Upload official property documents to verify accuracy of your listing details. This is typically the Property Record Card.")
        uploaded_docs = st.file_uploader("Upload official documents (PDF)", type="pdf", accept_multiple_files=True)
        
        if uploaded_docs:
            st.session_state.uploaded_documents = ""
            for doc in uploaded_docs:
                doc_text = extract_text_from_pdf(doc)
                st.session_state.uploaded_documents += f"\n\n--- Document: {doc.name} ---\n{doc_text}"
            st.success(f"{len(uploaded_docs)} document(s) uploaded successfully!")

    # Process button
    if st.button("Generate Listing and Reports"):
        with st.spinner("Our AI agents are working on your listing and reports..."):
            # Check if we have enough data
            if len([v for k, v in st.session_state.property_data.items() if v and k not in ["Street Address"]]) < 5:
                st.error("Please fill in more property details before generating reports.")
            else:
                # Generate the reports using our AI agents
                seo_listing, title_tag, meta_description, keyword_analysis, price_report, accuracy_report = critic_agent(
                    st.session_state.property_data, 
                    "",  # No current listing for new listing flow
                    st.session_state.uploaded_documents
                )
                
                # Store results in session state
                st.session_state.seo_listing = seo_listing
                st.session_state.title_tag = title_tag
                st.session_state.meta_description = meta_description
                st.session_state.keyword_analysis = keyword_analysis
                st.session_state.price_report = price_report
                if st.session_state.uploaded_documents:
                    st.session_state.accuracy_report = accuracy_report
                
                st.success("AI reports generated successfully!")

with tab_existing:
    st.header("Optimize Your Existing Listing Description")
    st.write("Enter your existing listing description below and provide additional property details to enhance the SEO optimization.")
    
    existing_listing = st.text_area("Current Listing Description", height=200)
    
    with st.expander("ESSENTIAL PROPERTY DETAILS"):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.property_data['Property Type'] = st.selectbox(
                "Property Type", 
                ["Single Family Home", "Townhouse", "Condo", "Multi-Family", "Land", "Luxury Home", "Vacation Property"],
                key="existing_property_type"
            )
            st.session_state.property_data['City'] = st.text_input("City", key="existing_city")
            st.session_state.property_data['Neighborhood'] = st.text_input("Neighborhood", key="existing_neighborhood")
        with col2:
            st.session_state.property_data['Bedrooms'] = st.number_input("Bedrooms", min_value=0, max_value=20, value=3, key="existing_bedrooms")
            st.session_state.property_data['Bathrooms'] = st.number_input("Bathrooms", min_value=0.0, max_value=20.0, value=2.0, step=0.5, key="existing_bathrooms")
            
            # Modified to allow custom target buyer
            target_buyer_options = ["First-time homebuyers", "Luxury buyers", "Investors", "Retirees/Downsizers", 
                "Growing families", "Urban professionals", "Vacation/Second home buyers", "Custom"]
            target_buyer_selection = st.selectbox("Target Buyer", target_buyer_options, key="existing_target_buyer_selection")
            
            if target_buyer_selection == "Custom":
                st.session_state.property_data['Target Buyer'] = st.text_input("Enter Custom Target Buyer", key="existing_custom_target")
            else:
                st.session_state.property_data['Target Buyer'] = target_buyer_selection
    
    with st.expander("ADDITIONAL DETAILS (OPTIONAL)"):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.property_data['Square Footage'] = st.number_input("Square Footage", min_value=0, value=2000, key="existing_sqft")
            st.session_state.property_data['Year Built'] = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000, key="existing_year")
            st.session_state.property_data['Special Features'] = st.text_area("Special Features", key="existing_features")
        with col2:
            st.session_state.property_data['School District'] = st.text_input("School District", key="existing_schools")
            st.session_state.property_data['Nearby Landmarks'] = st.text_input("Nearby Landmarks", key="existing_landmarks")
            st.session_state.property_data['Selling Points'] = st.text_area("Key Selling Points", key="existing_selling_points")
    
    # Chat section for the agent to request additional information
    st.subheader("Chat with our AI Agent")
    st.write("Our AI agent may ask for additional information to create the best SEO optimized listing.")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message"><strong>BURO Agent:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    api_key_set= "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"
    # Chat input
    user_message = st.text_input("Ask a question or provide additional information")
    if st.button("Send"):
        if api_key_set:
            if user_message:
                response = chat_with_agent(user_message)
                st.rerun()  # Rerun to update chat history display
        else:
            st.warning("Please provide your OpenAI API key in the sidebar.")
    
    # Optimize button
    if st.button("Optimize Existing Listing"):
        if api_key_set:
            if not existing_listing:
                st.error("Please enter your current listing description.")
            else:
                with st.spinner("Our AI agents are optimizing your listing..."):
                    # Generate the reports using our AI agents
                    seo_listing, title_tag, meta_description, keyword_analysis, price_report, accuracy_report = critic_agent(
                        st.session_state.property_data,
                        existing_listing,  # Pass the existing listing
                        None  # No document checking for optimization flow
                    )
                    
                    # Store results in session state
                    st.session_state.seo_listing = seo_listing
                    st.session_state.title_tag = title_tag
                    st.session_state.meta_description = meta_description
                    st.session_state.keyword_analysis = keyword_analysis
                    st.session_state.price_report = price_report
                    
                    st.success("Listing optimization complete!")
        else:
            st.warning("Please provide your OpenAI API key in the sidebar.")

# Display tabs with results
tab1, tab2, tab3, tab4 = st.tabs(["SEO Optimized Listing", "SEO Elements", "Listing Price Report", "Listing Accuracy Report"])

with tab1:
    if st.session_state.seo_listing:
        st.markdown("## SEO Optimized Listing")
        st.markdown(st.session_state.seo_listing)
        st.markdown(get_download_link(st.session_state.seo_listing, "seo_listing.txt", "Download SEO Listing"), unsafe_allow_html=True)
    else:
        st.info("Generate your listing to see the SEO optimized content here.")

with tab2:
    if st.session_state.title_tag or st.session_state.meta_description or st.session_state.keyword_analysis:
        st.markdown("## SEO Elements")
        
        st.markdown("### Title Tag")
        st.markdown(f'<div class="meta-box">{st.session_state.title_tag}</div>', unsafe_allow_html=True)
        
        st.markdown("### Meta Description")
        st.markdown(f'<div class="meta-box">{st.session_state.meta_description}</div>', unsafe_allow_html=True)
        
        st.markdown("### Keyword Analysis")
        st.markdown(f'<div class="highlight">{st.session_state.keyword_analysis}</div>', unsafe_allow_html=True)
        
        combined_seo = f"""
# SEO Elements

## Title Tag
{st.session_state.title_tag}

## Meta Description
{st.session_state.meta_description}

## Keyword Analysis
{st.session_state.keyword_analysis}
        """
        st.markdown(get_download_link(combined_seo, "seo_elements.txt", "Download SEO Elements"), unsafe_allow_html=True)
    else:
        st.info("Generate your listing to see the SEO elements here.")

with tab3:
    if st.session_state.price_report:
        st.markdown("## Listing Price Report")
        st.markdown(st.session_state.price_report)
        st.markdown(get_download_link(st.session_state.price_report, "price_report.txt", "Download Price Report"), unsafe_allow_html=True)
    else:
        st.info("Generate your listing to see the pricing recommendation here.")

with tab4:
    if st.session_state.accuracy_report:
        st.markdown("## Listing Accuracy Report")
        st.markdown(st.session_state.accuracy_report)
        st.markdown(get_download_link(st.session_state.accuracy_report, "accuracy_report.txt", "Download Accuracy Report"), unsafe_allow_html=True)
    elif st.session_state.seo_listing and not st.session_state.uploaded_documents:
        st.info("Upload official property documents to generate an accuracy report.")
    else:
        st.info("Generate your listing and upload official documents to see the accuracy report here.")

# Footer
st.markdown("---")
st.caption("© 2025 BURO - Your Listing Optimizer")
