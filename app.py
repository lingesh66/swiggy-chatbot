import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

# Swiggy FAQ data
data = {
    "Question": [
        "What is Swiggy Customer Care Number?",
        "Can I edit my order?",
        "I want to cancel my order",
        "Will Swiggy be accountable for quality/quantity?",
        "Is there a minimum order value?",
        "Do you charge for delivery?",
        "How long do you take to deliver?",
        "What are your delivery hours?",
        "Can I order from any location?",
        "Is single order from many restaurants possible?",
        "Do you support bulk orders?",
        "Can I order in advance?",
        "Can I change the address/number?",
        "Did not receive OTP?",
        "Did not receive referral coupon?",
        "Deactivate my account",
        "Unable to view the details in my profile",
        "What is Swiggy Money?",
        "Do you accept Sodexo, Ticket Restaurant etc.?"
    ],
    "Answer": [
        "You can reach Swiggy's customer care at 080-67466729.",
        "Your order can be edited before it reaches the restaurant. You could contact the customer support team via chat or call to do so. Once the order is placed and the restaurant starts preparing your food, you may not edit its contents.",
        "We will do our best to accommodate your request if the order is not placed to the restaurant. Please note that we will have the right to charge a cancellation fee up to the full order value to compensate our restaurant and delivery partners if your order has been confirmed.",
        "Quantity and quality of the food are the restaurant's responsibility. However, in case of issues with quality or quantity, kindly submit your feedback, and we will pass it on to the restaurant.",
        "We have no minimum order value, and you can order for any amount.",
        "Delivery fees vary from city to city and are applicable if the order value is below a certain amount. Additionally, certain restaurants might have fixed delivery fees. Delivery fees (if any) are specified on the 'Review Order' page.",
        "Standard delivery times vary by the location selected and prevailing conditions. Once you select your location, an estimated delivery time is mentioned for each restaurant.",
        "Our delivery hours vary for different locations and depend on the availability of supply from restaurant partners.",
        "We will deliver from any restaurant listed in the search results for your location. We recommend enabling your GPS location finder and letting the app auto-detect your location.",
        "We currently do not support this functionality. However, you can place orders for individual items from different restaurants.",
        "In order to provide all customers with a great selection and to ensure on-time delivery of your meal, we reserve the right to limit the quantities depending on supply.",
        "We currently do not support this functionality. All our orders are placed and executed on-demand.",
        "Any major change in the delivery address is not possible after you have placed an order with us. However, slight modifications like changing the flat number, street name, landmark, etc. are allowed. If you have received delivery executive details, you can directly call him; else you could contact our customer service team.",
        "Please check if your app is due for an update. If not, please share the details via support@swiggy.in.",
        "Referral coupon is given upon the first successful transaction of the referred person. If you still have not received it, kindly send us your details at support@swiggy.in. We will contact you within 48 hours.",
        "Please write to us at support@swiggy.in in the event that you want to deactivate your account.",
        "Please check if your app is due for an update. If not, please share the details via support@swiggy.in.",
        "Swiggy Money is a secure digital wallet where you can store digital currency and use it for faster checkouts. It prevents payment failures and gives you seamless refunds when necessary.",
        "We do not accept Sodexo vouchers, but we do accept Sodexo cards. You can select the Sodexo card option while selecting payment options at the time of order."
    ]
}

# Convert to DataFrame
faq_df = pd.DataFrame(data)
# faq_df.to_csv('faq.csv', index=False)

# Display the DataFrame
# faq_df.head()

from transformers import AutoTokenizer, AutoModel
import torch

# Load the model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to get embeddings
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

# Embed the questions
faq_df['Embeddings'] = faq_df['Question'].apply(lambda x: get_embeddings(x).numpy())

# # Display the DataFrame with embeddings
# faq_df.head()


api_key="f7f2e9e3-c8c8-4559-8721-edfa5e8901a3"

import os
import pinecone
from pinecone import ServerlessSpec, Pinecone

# Initialize Pinecone by creating an instance of the Pinecone class
pc = Pinecone(api_key=api_key)
index_name = "chatbot"

# Check if the index exists, and if not, create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=19,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

pc = Pinecone(api_key="f7f2e9e3-c8c8-4559-8721-edfa5e8901a3")
index = pc.Index("chatbot")

# Prepare vectors for upsert
vectors_to_upsert = [
    {
        "id": f"faq_{i}",
        "values": faq_df['Embeddings'][i].flatten().tolist()[:19],  # Ensure the vector is 19-dimensional
        "metadata": {"question": faq_df['Question'][i]}
    }
    for i in range(len(faq_df))
]

# Upsert vectors into Pinecone
index.upsert(
    vectors=vectors_to_upsert,
    namespace="faq_namespace"
)

# Query example
query_vector = get_embeddings("Can I edit my order?").flatten().tolist()[:19]  # Match the 19 dimensions
results = index.query(
    namespace="faq_namespace",
    vector=query_vector,
    top_k=1,
    include_values=True,
    include_metadata=True
)

from langchain_huggingface import HuggingFaceEndpoint
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import login
from langchain.llms import HuggingFaceEndpoint  # Assuming you're using LangChain

# Set your Hugging Face token
sec_key = "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"
HUGGINGFACEHUB_TOKEN=sec_key
# Log in using your Hugging Face token
login(token=sec_key)

HF_TOKEN = "<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>"
from huggingface_hub import InferenceClient
import json

# Hugging Face model setup
repo_id = "google/flan-t5-large"

llm_client = InferenceClient(
    model=repo_id,
    timeout=120,
)

# Function to call the LLM with improved control over the response
def call_llm(inference_client: InferenceClient, prompt: str):
    response = inference_client.post(
        json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.7,  # Balanced creativity to maintain factual accuracy
                "top_p": 0.9,  # Ensures varied output but controls randomness
            },
            "task": "text-generation",
        },
    )
    return json.loads(response.decode())[0]["generated_text"]

# Function to find a relevant FAQ from the database with confidence threshold
def find_relevant_faq(query, threshold=0.8):
    query_vector = get_embeddings(query).flatten().tolist()[:19]  # Match the 19 dimensions
    results = index.query(
        namespace="faq_namespace",
        vector=query_vector,
        top_k=1,
        include_values=True,
        include_metadata=True
    )

    if results["matches"]:
        best_match = results["matches"][0]
        if best_match["score"] >= threshold:  # Check if the match confidence is above the threshold
            question = best_match["metadata"]["question"]
            answer = faq_df[faq_df["Question"] == question]["Answer"].values[0]
            return question, answer
    return None, None

# Function to generate a response based on the user query with emphasis on accuracy
def generate_response(user_query):
    question, answer = find_relevant_faq(user_query)

    if question and answer:
        # Construct a prompt for the LLM that encourages creativity but emphasizes factual accuracy
        prompt = (
            f"The user asked: '{user_query}'.\n\n"
            f"Relevant FAQ:\nQ: '{question}'\nA: {answer}\n\n"
            f"Please use this FAQ as the basis of your response, but feel free to elaborate creatively to engage the user."
            f" Make sure your response is informative and accurate, but can also provide some context or extra suggestions."
        )
        response = call_llm(llm_client, prompt)

        # Double-check that the response contains relevant information from the FAQ
        if answer.lower() in response.lower():  # Ensure that the core answer is present
            return response.strip()
        else:
            # Fall back to the FAQ answer if the model strays too far
            return f"{answer} "
    else:
        # Return a fallback response when no relevant FAQ is found
        return (
            "I couldn't find an exact match for your question, but I'm here to help! "
            "Could you please provide more details, or try asking in another way?"
        )

# Example usage
query = "what is the swiggy customer care number?"
response = generate_response(query)
print(response)


# Streamlit app
st.title("Chatbot Interface")
st.write("Ask your question below:")

user_query = st.text_input("Your Question:")
if st.button("Submit"):
    if user_query:
        response = generate_response(user_query)
        st.write(f"**Response:** {response}")
    else:
        st.write("Please enter a question.")

# Example: streamlit run your_script.py
