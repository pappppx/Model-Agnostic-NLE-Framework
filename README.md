# A Model-Agnostic Framework for Natural Language Explanations of Predictive Models

This repository contains the source code and experimental framework for the Master's Thesis: **"A Model-Agnostic Framework for Natural Language Explanations of Predictive Models"**.

The framework introduces a robust, Object-Oriented architecture designed to translate the outputs of black-box predictive models into human-readable **Natural Language Explanations (NLEs)**. It integrates **Feature Importance Methods** (SHAP, LIME) and **Logic-based Approaches** (Araucana-XAI) with **Large Language Models** (Llama-3 via Unsloth). The system aims to mitigate LLM hallucinations by injecting structural logic, generating coherent, truthful, and role-adapted reports.

---

## 📌 Project Overview & Hypothesis
The core objective is to evaluate whether injecting **Logic-based (Araucana-XAI)** alongside traditional **Feature Importance (SHAP, LIME)** improves the causal reasoning of LLMs. 

By comparing a *Baseline* (Feature Weights only) against an *Enhanced Hybrid* (Weights + Decision Tree Logic), this framework demonstrates that providing explicit structural boundaries allows the LLM to generate more accurate, actionable, and causally grounded text.

To maintain a clean experimental environment, the core logic has been abstracted into a dedicated Python package: `nle-framework-pappppx`, available on PyPI.

---

## 🛠️ System Architecture

The project adheres to the **Separation of Concerns** principle to manage GPU memory efficiently and maintain modularity:

* **`NLEModel`:** Utilizes a Singleton pattern to load and cache the Llama-3 model in the GPU. This prevents Out-of-Memory errors and redundant loading times.
* **`NLEAdapter`:** Acts as a dedicated configuration profile. It handles Prompt Engineering, orchestrates the XAI extraction (SHAP/LIME/Araucana), and communicates with the `NLEModel` to generate the final text.

---

## 🚀 Installation & Usage

You do not need to clone this repository to use the framework. It can be installed directly via `pip`.

### Option A: Google Colab (Recommended)
The easiest way to replicate the experiments is using Google Colab's free GPU tier.

1. Open a new [Google Colab](https://colab.research.google.com/) notebook.
2. Go to `Runtime` > `Change runtime type` and select **T4 GPU**.
3. Install the package in the first cell:
   ```bash
   !pip install nle-framework-pappppx
   ```
4. Import the classes and start evaluating your models (see the `experiment.ipynb` notebook in this repository for a full breast cancer classification example).

### Option B: Local Environment
If running locally, ensure you have an NVIDIA GPU compatible with 4-bit quantization (e.g., RTX 30/40 series, Tesla T4/V100) and CUDA installed.

```bash
pip install nle-framework-pappppx
```

---

## 🔄 Extensibility & Domain Adaptation

This framework is highly modular. You can seamlessly switch from Oncology to other domains like Finance, HR, or Law simply by creating a new `NLEAdapter`. 

### How to Adapt to a New Domain (e.g., Finance):

1. **Train your Model:** Train any scikit-learn compatible black-box model (e.g., Random Forest) on your custom tabular dataset.
2. **Initialize the Model Manager:** Load the LLM into the GPU once.
   ```python
   from nle_core import NLEModel, NLEAdapter

   # Load the LLM into GPU memory
   nlem = NLEModel(llm_model_name="unsloth/llama-3-8b-Instruct")
   ```
3. **Define a New Persona and Create the Adapter:** Write a system prompt tailored to your domain and instantiate a new adapter.
   ```python
   # Define a Financial Analyst Persona
   finance_system_prompt = """
   SYSTEM ROLE: You are a Senior Financial Risk Analyst.
   OBJECTIVE: Explain the credit score decision based on user data.
   CONSTRAINTS: Explain the reasoning clearly without using raw code or nested lists.
   """

   # Define the user template (Universal)
   UNIVERSAL_USER_TEMPLATE = """
   PREDICTION: {prediction}
   XAI CONTEXT: {context_data}
   INPUTS: {input_features}
   Explain this:
   """

   # Create the adapter linked to your class names
   finance_adapter = NLEAdapter(
       system_prompt=finance_system_prompt,
       user_template=UNIVERSAL_USER_TEMPLATE,
       nle_model=nlem,
       class_names={0: "Loan Denied", 1: "Loan Approved"}
   )
   ```
4. **Generate the Explanation:**
   ```python
   # Generate report combining SHAP, LIME, and Araucana
   report = finance_adapter.generate_explanation(
       black_box_model=my_rf_model, 
       X_train=X_train_finance, 
       instance=client_data, 
       explainer_type="shap+lime+araucana",
       plot=False
   )
   print(report)
   ```

---

## 🔬 Experimental Reproducibility

The `notebooks/` directory contains the complete Jupyter Notebook (`experiment.ipynb`) implementing the clinical pipeline. It includes:
* **Dataset:** Breast Cancer Wisconsin Diagnostic.
* **Pipeline:** End-to-end Random Forest training and evaluation.
* **Batch Execution:** Side-by-side comparative generation of *Baseline* (Weights only) vs *Hybrid* (Logic + Weights) reports across multiple patient profiles.
