
from langchain_huggingface import HuggingFaceEmbeddings 

import os
import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer


import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
import wfdb
import neurokit2 as nk
import scipy.stats as stats
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.summarize import load_summarize_chain


from typing import List,Any,Dict
from langchain_core.vectorstores.base import VectorStore
from langchain_core.documents import Document
from nltk.tokenize import sent_tokenize
import numpy as np

from langchain_core.retrievers import BaseRetriever

from dotenv import load_dotenv

import json

from collections import defaultdict
import ast

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field
import torch

embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name,model_kwargs={'device':'cuda'}) 
max_length = embedding_model._client.tokenizer.model_max_length - 50

eval_model_name = "google/gemma-2-2b-it"
eval_tokenizer = AutoTokenizer.from_pretrained(eval_model_name)
eval_model = AutoModelForCausalLM.from_pretrained(eval_model_name)

class diagnosisOutput(BaseModel):
    disease_name : str
    Result: bool
    Explanation: str

pipe = pipeline(
    "text-generation",
    model=eval_model,
    device_map="auto",
    tokenizer=eval_tokenizer,
    return_full_text = False
)


llm = HuggingFacePipeline(
    pipeline=pipe
)


def is_valid_page(text: str) -> bool:
    """
    Check if a page is valid based on its content.
    """
    text = text.strip()
    if len(text) < 50:
        return False
    if "TEST" in text and len(set(text.split())) <= 5:
        return False
    return True

def create_vector_db():
    """
    Create a Chroma vector store from the given database path.
    """
# --- Step 1: Delete existing DB if any
    db_path = os.path.join(os.getcwd(), "chroma_db")
    try:
        chroma_client = chromadb.PersistentClient(path=db_path)
        chroma_client.delete_collection(name='Book_embeddings')
    except Exception as e:
        print(f"DB deletion error (ignored): {e}")


    # --- Step 2: Load PDFs and filter
    pdf_paths = [
        'books/12_lead_ecg_the_art_of_interpretation.pdf',
        'books/jane-huff-ecg-workout-exercises-in-arrhythmia-interpretation.pdf'
    ]

    split_docs = []
    for pdf in pdf_paths:
        loader = PyPDFLoader(os.path.join(os.getcwd(), pdf))
        docs = loader.load()

        # Filter junk pages before splitting
        docs = [doc for doc in docs if is_valid_page(doc.page_content)]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100, length_function=len
        )
        documents = text_splitter.split_documents(documents=docs)
        split_docs.extend(documents)

    # --- Step 3: Create Chroma vector store
    vector_store = Chroma.from_documents(
        split_docs,
        embedding=embedding_model,
        persist_directory=db_path,
        collection_name='Book_embeddings'
    )
    retriever = vector_store.as_retriever()

    return retriever


def create_diagnosis_guidance(retriever: BaseRetriever, disease: str) -> dict[str]:

    diseases = {"ST/T segment change (STTC)", "myocardial infarction(MI)", "conduction disturbance (CD)", "hypertrophy(HYP)"}
    diagnosis_guidance = defaultdict(str)
    guiding_prompt = ChatPromptTemplate[(
                    ("system","You are an expert literature retriever"),
                    ("human","Create a diagnosis guidance to interpret {disease} related arrhytmias in a 12-lead ecg system."),
                    )]
    create_diagnosis_guidance = create_stuff_documents_chain(llm=llm,prompt = guiding_prompt,)

    for disease in diseases:
        query = f"How to interpret {disease} related arrhythmias in a 12-lead ecg? "
        retrieved_context = retriever.invoke(query)
        docs = [doc.page_content for doc in docs]
        # See output format
        diagnosis_guidance[disease] = create_diagnosis_guidance.invoke({"context":docs,"disease":disease})
    retrieved_context_NORM = retriever.invoke("How to interpret if a person does not have any heart arrhytmia using 12-lead ecg system")
    NORM_docs = [doc.page_content for doc in retrieved_context_NORM]
    diagnosis_guidance["NORM"] = create_diagnosis_guidance(llm = llm, prompt = "Create a diagnosis guidance to interpret if a person does not have any related arrhytmias in a 12-lead ecg system ")
    return diagnosis_guidance


def extract_full_ecg_features(record):
    """
    Extracts and describes key ECG features from each lead in the multi-lead ECG signal.
    
    Parameters:
    - record: ECG data record containing multi-lead ECG signals and sampling rate.
    
    Returns:
    - A dictionary containing descriptive statistics and fiducial features for each lead.
    """
    # Extract the ECG signal, lead names, and sampling rate
    signals = record.p_signal  # multi-lead ECG signal (time x leads)
    leads = record.sig_name    # Lead names (e.g., ['I', 'II', ..., 'V6'])
    sampling_rate = record.fs  # Sampling rate of the ECG signal (in Hz)
    all_lead_features = {}

    # Iterate over each ECG lead
    for lead_idx, lead_name in enumerate(leads):
        lead_signal = signals[:, lead_idx]  # Extract signal for the current lead

        # Basic Statistics with safety checks
        mean_val = np.mean(lead_signal) if lead_signal.size > 0 else None
        variance = np.var(lead_signal) if lead_signal.size > 0 else None
        iqr = stats.iqr(lead_signal) if lead_signal.size > 1 else None
        min_val = np.min(lead_signal) if lead_signal.size > 0 else None
        max_val = np.max(lead_signal) if lead_signal.size > 0 else None

        # Process ECG signal to extract fiducial points (R, P, T peaks)
        try:
            processed_signals, _ = nk.ecg_process(lead_signal, sampling_rate=sampling_rate)
        except Exception as e:
            print(f"Skipping lead {lead_name} due to ECG processing error: {e}")
            continue

        # Extract Peaks: R, P, T peaks
        r_peaks = np.where(processed_signals["ECG_R_Peaks"] == 1)[0]
        p_peaks = np.where(processed_signals["ECG_P_Peaks"] == 1)[0]
        t_peaks = np.where(processed_signals["ECG_T_Peaks"] == 1)[0]

        # Amplitudes of the R, P, and T peaks
        r_amplitudes = lead_signal[r_peaks].tolist() if len(r_peaks) > 0 else []
        p_amplitudes = lead_signal[p_peaks].tolist() if len(p_peaks) > 0 else []
        t_amplitudes = lead_signal[t_peaks].tolist() if len(t_peaks) > 0 else []

        # Calculate RR Intervals (time between consecutive R-peaks) and Heart Rate
        rr_intervals = np.diff(r_peaks) / sampling_rate if len(r_peaks) > 1 else []
        mean_rr = np.mean(rr_intervals) if len(rr_intervals) > 0 else None
        heart_rate = 60 / mean_rr if mean_rr else None

        # Function to calculate descriptive statistics safely
        def safe_stats(arr):
            return {
                "mean": np.mean(arr) if len(arr) > 0 else None,
                "variance": np.var(arr) if len(arr) > 0 else None,
                "iqr": stats.iqr(arr) if len(arr) > 1 else None,
                "min": np.min(arr) if len(arr) > 0 else None,
                "max": np.max(arr) if len(arr) > 0 else None
            }

        # Organize features into a dictionary with descriptive names
        all_lead_features[lead_name] = {
            "Statistical Features": {
                "Mean Voltage (mV)": mean_val,
                "Voltage Variance": variance,
                "Interquartile Range (IQR)": iqr,
                "Minimum Voltage (mV)": min_val,
                "Maximum Voltage (mV)": max_val,
            },
            "ECG Fiducial Features": {
                "R-Wave Amplitudes (mV)": safe_stats(r_amplitudes),
                "P-Wave Amplitudes (mV)": safe_stats(p_amplitudes),
                "T-Wave Amplitudes (mV)": safe_stats(t_amplitudes),
                "RR Intervals (seconds)": safe_stats(rr_intervals),
                "Mean RR Interval (seconds)": mean_rr,
                "Heart Rate (BPM)": heart_rate
            }
        }

    # Generate a descriptive text for each lead
    lead_descriptions = []
    for lead, feat in all_lead_features.items():
        parts = [f"Lead {lead}:"]

        # Iterate over all categories of features (Statistical and Fiducial)
        for category_name, category in feat.items():
            parts.append(f"  {category_name}:")
            for key, val in category.items():
                if isinstance(val, dict):
                    # If value is a dict, expand and display each sub-statistic
                    sub = ", ".join(f"{k}: {round(v, 3) if isinstance(v, float) else v}" for k, v in val.items())
                    parts.append(f"    {key} ({sub})")
                else:
                    # Format the main feature value
                    v = round(val, 3) if isinstance(val, float) else val
                    parts.append(f"    {key}: {v}")

        lead_descriptions.append("\n".join(parts))

    # Combine all lead descriptions into a final output string
    full_description = "\n\n".join(lead_descriptions)
    
    return full_description


def diagnose(record, disease:str, diagnosis_guidance:dict[str]) :
    feature_prompt = ChatPromptTemplate[(
        ("system","You are an ECG expert and can calculate fiducial points and segements across different leads."),
        ("human","Retrieve the relevant information from {features} required to interpret {disease} heart arrhythmias. Use this {diagnosis_guidance}")
    )]
    patient_features = extract_full_ecg_features(record)
    create_feature_prompt = create_stuff_documents_chain(llm = llm, prompt = feature_prompt)
    feature_prompt = create_feature_prompt.invoke({
                    "features": patient_features,
                    "disease": disease,
                    "diagnosis_guidance": diagnosis_guidance[disease]
                    
                })
    diagnosis_prompt = ChatPromptTemplate.from_template("""
        You are a medical expert in heart arrhythmia diagnosis.

        DISEASE TO DIAGNOSE: {input}
        MEDICAL CONTEXT: {context}
        PATIENT'S MEASURED FEATURES: {patient_features}

        Based on medical literature and the patient's measurements, determine whether the patient has {input}.

        Please return ONLY a JSON object with the following fields:

        - disease_name: Name of the disease (must match the input)
        - Result: true or false, depending on whether the patient has the disease
        - Explanation: Justification for the decision

        Example output:
        {{
        "disease_name": "NORM",
        "Result": False,
        "Explanation": "The ECG shows significant abnormalities including ST segment elevation, inverted T waves, and abnormal RS complexes, indicating the presence of arrhythmias and excluding a normal ECG"
        }}

        ⚠️ Do not add any text before or after the JSON block.
        Return only the JSON in valid format.
        """)
    parser = JsonOutputParser(pydantic_object=diagnosisOutput)
    final_chain = diagnosis_prompt | llm | parser
    result = final_chain.invoke({
        "input": disease,
        "context": diagnosis_guidance[disease],
        "patient_features": patient_features
    })
    return result

def diagnose_ecg(record) -> dict[str]:
    retriever = create_vector_db()
    diagnosis_guidance = defaultdict(str)
    for disease in ["ST/T segment change (STTC)", "myocardial infarction(MI)", "conduction disturbance (CD)", "hypertrophy(HYP)"]:
        diagnosis_guidance[disease] = create_diagnosis_guidance(retriever, disease)
    result = defaultdict(str)
    for disease in ["ST/T segment change (STTC)", "myocardial infarction(MI)", "conduction disturbance (CD)", "hypertrophy(HYP)"]:
        response = diagnose(record, disease, diagnosis_guidance)
        result[disease] = response
    return result


