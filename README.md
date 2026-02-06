# A Model-Agnostic Framework for Natural Language Explanations of Predictive Models

This repository contains the implementation of the Master's Thesis: **"A Model-Agnostic Framework for Natural Language Explanations of Predictive Models"**.

The framework integrates **Feature Importance Methods** (SHAP, LIME) and **Logic-Based Approaches** (Araucana-XAI) with **Large Language Models** (Llama-3 via Unsloth) to generate coherent, truthful, and role-adapted clinical reports from "black-box" models.

## 📂 Repository Contents

* `A_Model_Agnostic_Framework_for_NLE_of_Predictive_Models.ipynb`: The complete, self-contained Jupyter Notebook implementing the pipeline.

## 🚀 Quick Start (Google Colab)

The easiest way to replicate the experiments is utilizing Google Colab's free GPU tier.

1.  Download the `.ipynb` file from this repository.
2.  Upload it to [Google Colab](https://colab.research.google.com/).
3.  **Runtime Setup:**
    * Go to `Runtime` > `Change runtime type`.
    * Select **T4 GPU** as the hardware accelerator.
4.  **Execution:**
    * Run the first cell to install dependencies (Unsloth, Araucana-XAI, SHAP, LIME).
    * *Note:* A session restart might be required after installation (Colab will prompt you).

## 🛠️ System Requirements

* **Python:** 3.10+
* **GPU:** NVIDIA Tesla T4 (minimum 15GB VRAM recommended for 4-bit LLM inference).
* **Key Libraries:**
    * `unsloth` (LLM Optimization)
    * `araucanaxai` (Decision Tree Surrogates)
    * `shap` & `lime` (Explainability)
    * `scikit-learn` (Black-box modeling)

## 🔄 Extensibility

This framework is designed to be modular. To adapt it to a new domain (e.g., Credit Risk or Legal Assessment):

1.  **Data Layer:** Replace the `load_breast_cancer()` function with your dataset loading logic.
2.  **Model Layer:** Train your classifier (Random Forest, SVM, etc.) on the new data.
3.  **Adapter Layer:** Modify the `prompts_dict` dictionary in the notebook. Update the `SYSTEM ROLE` and context variables to match your new domain (e.g., changing "Oncologist" to "Financial Analyst").
