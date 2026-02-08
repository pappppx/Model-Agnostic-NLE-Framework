# A Model-Agnostic Framework for Natural Language Explanations of Predictive Models

This repository contains the source code for the Master's Thesis: **"A Model-Agnostic Framework for Natural Language Explanations of Predictive Models"**.

The framework introduces a robust, Object-Oriented architecture that integrates **Feature Importance Methods** (SHAP, LIME) and **Symbolic Logic Approaches** (Araucana-XAI) with **Large Language Models** (Llama-3 via Unsloth). The system generates coherent, truthful, and role-adapted clinical reports, specifically designed to mitigate LLM hallucinations through structural logic injection.

## 📂 Repository Contents

* `A_Model_Agnostic_Framework_for_NLE_of_Predictive_Models.ipynb`: The complete, self-contained Jupyter Notebook implementing the OOP framework and experimental pipeline.

## 🚀 Quick Start (Google Colab)

The easiest way to replicate the experiments is utilizing Google Colab's free GPU tier.

1.  Download the `.ipynb` file from this repository.
2.  Upload it to [Google Colab](https://colab.research.google.com/).
3.  **Runtime Setup:**
    * Go to `Runtime` > `Change runtime type`.
    * Select **T4 GPU** as the hardware accelerator.
4.  **Execution:**
    * Run the installation cells to set up the environment.
    * The framework automatically handles the download and caching of the Llama-3 model.

## 🛠️ System Architecture & Requirements

The project is built on Python 3.10 and utilizes a **Singleton Pattern** to manage GPU memory efficiently.

* **Core Class:** `NLEFramework` (Orchestrates XAI extraction and Prompt Generation).
* **Hardware:** NVIDIA Tesla T4 (Required for 4-bit quantization).
* **Key Libraries:**
    * `unsloth` (Efficient LLM Inference)
    * `araucanaxai` (Symbolic Decision Tree Surrogates)
    * `shap` & `lime` (Feature Attribution)
    * `scikit-learn` (Black-box modeling)

## 🔄 Extensibility & Domain Adaptation

This framework is designed with modularity in mind, using an **Adapter Pattern** to switch between different domains (e.g., Oncology, Finance, Law) without modifying the core codebase.

### How to Adapt to a New Domain:

1.  **Data Layer:**
    Replace the data loading function in Block 2 with your custom tabular dataset. Retrain the `black_box_model` (e.g., Random Forest) on your new data.

2.  **Initialization:**
    Instantiate the framework with your specific class labels:
    ```python
    nle = NLEFramework(class_names={0: "Loan Denied", 1: "Loan Approved"})
    ```

3.  **Adapter Layer (Prompt Engineering):**
    Do not edit the source code. Instead, register a new **Adapter** using the `register_adapter()` method. Define a new System Role and use the provided `UNIVERSAL_USER_TEMPLATE`.

    ```python
    # Example: Defining a Financial Analyst Persona
    finance_system = """
    SYSTEM ROLE: You are a Senior Financial Analyst.
    OBJECTIVE: Explain the credit score decision based on debt-to-income ratio.
    ...
    """
    
    # Registering the new capability
    nle.register_adapter("finance_expert", finance_system, UNIVERSAL_USER_TEMPLATE)
    ```

## 🔬 Experimental Reproducibility

The notebook includes a comparative experiment block comparing:
* **Baseline Approach:** Explanations based solely on SHAP/LIME weights.
* **Hybrid Approach (Proposed):** Explanations grounded in Araucana's symbolic logic trees.

Run **Block 5** in the notebook to generate the side-by-side comparison reports for specific test instances.
