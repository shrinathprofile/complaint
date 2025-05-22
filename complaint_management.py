import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
import json
import uuid
from datetime import datetime
import logging
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="YOUR_OPENROUTER_API_KEY",  # Replace with your OpenRouter API key
)

# Synthetic knowledge base for RAG
knowledge_base = [
    {
        "id": 1,
        "text": "Customer complained about unauthorized transaction. Resolution: Refunded amount, issued apology, and updated security measures. Regulatory compliance: Guidelines on transparency.",
        "category": "Unauthorized Transaction",
        "keywords": ["refund", "security", "apology", "transparency"]
    },
    {
        "id": 2,
        "text": "Customer reported poor service at branch. Resolution: Provided training to staff, offered compensation. Regulatory compliance: Fairness principles.",
        "category": "Service Failure",
        "keywords": ["training", "compensation", "fairness"]
    },
    {
        "id": 3,
        "text": "Customer unable to access online services. Resolution: Investigated technical issue, restored access, and provided apology. Regulatory compliance: Guidelines on timely resolution.",
        "category": "Service Failure",
        "keywords": ["access", "technical", "apology", "timely resolution"]
    }
]

# Simple embedding function for RAG (mocked)
def get_embedding(text):
    return np.random.rand(768)  # Mock embedding (replace with real embedding API if available)

# Precompute embeddings for knowledge base
for item in knowledge_base:
    item["embedding"] = get_embedding(item["text"])

# Function to classify and prioritize complaint with retry mechanism
def classify_complaint(complaint_text, max_retries=3):
    prompt = f"""
    You are an AI assistant for a Complaint Management Program. Classify the customer complaint below into one of these categories: [Unauthorized Transaction, Service Failure, Product Issue, Other]. Assign a priority score (0-100) based on urgency and severity. Provide a brief explanation. Return a valid JSON object with fields: 'category', 'priority', and 'explanation'. 

    Complaint: {complaint_text}

    Example output:
    {{"category": "Service Failure", "priority": 80, "explanation": "The complaint about poor service indicates a need for immediate staff training."}}

    Ensure the output is valid JSON. Wrap the JSON in triple backticks (```json) for clarity.
    """
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "YOUR_SITE_URL",  # Replace with your site URL
                    "X-Title": "Complaint Management App",
                },
                model="google/gemma-3-27b-it:free",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            raw_response = completion.choices[0].message.content
            logger.debug(f"Attempt {attempt + 1} - Raw classification response: {raw_response}")
            
            # Clean response (remove backticks if present)
            raw_response = raw_response.strip()
            if raw_response.startswith("```json") and raw_response.endswith("```"):
                raw_response = raw_response[7:-3].strip()
            
            try:
                result = json.loads(raw_response)
                if not all(key in result for key in ["category", "priority", "explanation"]):
                    raise ValueError("Missing required fields in JSON response")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Attempt {attempt + 1} - JSON parsing error: {str(e)}")
                if attempt == max_retries - 1:
                    return {
                        "category": "Other",
                        "priority": 50,
                        "explanation": f"Failed to parse model response after {max_retries} attempts: {str(e)}. Defaulting to Other category."
                    }
            except ValueError as e:
                logger.error(f"Attempt {attempt + 1} - Validation error: {str(e)}")
                if attempt == max_retries - 1:
                    return {
                        "category": "Other",
                        "priority": 50,
                        "explanation": f"Invalid response format: {str(e)}. Defaulting to Other category."
                    }
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} - Classification API error: {str(e)}")
            if attempt == max_retries - 1:
                return {"error": f"Classification failed after {max_retries} attempts: {str(e)}"}
        time.sleep(1)  # Brief delay between retries

# Function to generate resolution draft using RAG
def generate_resolution(complaint_text, category, max_retries=3):
    complaint_embedding = get_embedding(complaint_text)
    similarities = [np.dot(complaint_embedding, item["embedding"]) for item in knowledge_base]
    top_idx = np.argmax(similarities)
    relevant_resolution = knowledge_base[top_idx]["text"]
    
    prompt = f"""
    You are an AI assistant for a Complaint Management Program. Draft a professional, empathetic, and regulatory-compliant resolution response for the complaint below. Use the provided past resolution as context and align with a professional, empathetic tone. Include an explanation of how regulatory requirements are met. Return a valid JSON object with fields: 'resolution' and 'regulatory_compliance'.

    Complaint: {complaint_text}
    Category: {category}
    Past Resolution Context: {relevant_resolution}

    Example output:
    {{"resolution": "We sincerely apologize for the inconvenience caused. We have resolved the issue and provided a refund.", "regulatory_compliance": "This response adheres to guidelines on transparency and fairness."}}

    Ensure the output is valid JSON. Wrap the JSON in triple backticks (```json) for clarity.
    """
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "YOUR_SITE_URL",  # Replace with your site URL
                    "X-Title": "Complaint Management App",
                },
                model="google/gemma-3-27b-it:free",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            raw_response = completion.choices[0].message.content
            logger.debug(f"Attempt {attempt + 1} - Raw resolution response: {raw_response}")
            
            # Clean response (remove backticks if present)
            raw_response = raw_response.strip()
            if raw_response.startswith("```json") and raw_response.endswith("```"):
                raw_response = raw_response[7:-3].strip()
            
            try:
                result = json.loads(raw_response)
                if not all(key in result for key in ["resolution", "regulatory_compliance"]):
                    raise ValueError("Missing required fields in JSON response")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Attempt {attempt + 1} - JSON parsing error: {str(e)}")
                if attempt == max_retries - 1:
                    return {
                        "resolution": "We apologize for the inconvenience. Our team is investigating the issue.",
                        "regulatory_compliance": f"Default response provided due to parsing error: {str(e)}."
                    }
            except ValueError as e:
                logger.error(f"Attempt {attempt + 1} - Validation error: {str(e)}")
                if attempt == max_retries - 1:
                    return {
                        "resolution": "We apologize for the inconvenience. Our team is investigating the issue.",
                        "regulatory_compliance": f"Invalid response format: {str(e)}."
                    }
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} - Resolution API error: {str(e)}")
            if attempt == max_retries - 1:
                return {"error": f"Resolution drafting failed after {max_retries} attempts: {str(e)}"}
        time.sleep(1)  # Brief delay between retries

# Streamlit app
st.set_page_config(page_title="Complaint Management App", layout="wide")
st.title("AI-Powered Complaint Management")

# Input section
st.header("Enter Complaint")
complaint_text = st.text_area("Customer Complaint", placeholder="Enter the customer complaint here...", height=200, value="Unable to access the company Q&A page")
submit_button = st.button("Process Complaint")

if submit_button and complaint_text:
    # Classify and prioritize
    with st.spinner("Classifying complaint..."):
        classification_result = classify_complaint(complaint_text)
    
    if "error" not in classification_result:
        st.header("Classification and Prioritization")
        st.json(classification_result)
        category = classification_result.get("category", "Other")
        priority = classification_result.get("priority", 0)
        
        # Generate resolution
        with st.spinner("Generating resolution draft..."):
            resolution_result = generate_resolution(complaint_text, category)
        
        if "error" not in resolution_result:
            st.header("Resolution Draft")
            st.write("**Resolution**:")
            st.write(resolution_result.get("resolution", ""))
            st.write("**Regulatory Compliance**:")
            st.write(resolution_result.get("regulatory_compliance", ""))
            
            # Download option
            output = {
                "complaint": complaint_text,
                "classification": classification_result,
                "resolution": resolution_result
            }
            st.download_button(
                label="Download Results",
                data=json.dumps(output, indent=2),
                file_name=f"complaint_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.error(resolution_result["error"])
    else:
        st.error(classification_result["error"])

# Sidebar with instructions
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Enter a customer complaint in the text area.
2. Click 'Process Complaint' to classify, prioritize, and generate a resolution draft.
3. View the results and download them as a JSON file.
""")
st.sidebar.header("About")
st.sidebar.write("This app integrates Generative AI to classify complaints and draft resolutions, supporting a general-purpose Complaint Management Program.")