import torch
from unsloth import FastLanguageModel
import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text
from sklearn.metrics import accuracy_score
import araucanaxai
import warnings
import gc

class NLEFramework:
    """
    Main framework class for generating Natural Language Explanations.
    It integrates a Large Language Model (Llama-3) with multiple XAI backends (SHAP, LIME, Araucana).
    Implements a Singleton pattern to avoid reloading the LLM into GPU memory.
    """

    # Singleton storage to share model across instances
    _loaded_model = None
    _loaded_tokenizer = None

    def __init__(self, llm_model_name="unsloth/llama-3-8b-Instruct", load_in_4bit=True, class_names={0: "Benigno", 1: "Maligno"}):
        print("--- Initializing NLEFramework ---")

        # Memory Management: Check if model is already loaded
        if NLEFramework._loaded_model is not None:
            print(f">>> Memory Optimization: Reusing loaded model '{llm_model_name}'.")
            self.model = NLEFramework._loaded_model
            self.tokenizer = NLEFramework._loaded_tokenizer
        else:
            print(f">>> Loading LLM from scratch: {llm_model_name}...")
            # Clear GPU cache before loading
            gc.collect()
            torch.cuda.empty_cache()

            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=llm_model_name,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=load_in_4bit,
            )
            FastLanguageModel.for_inference(self.model)

            # Cache the model in the class
            NLEFramework._loaded_model = self.model
            NLEFramework._loaded_tokenizer = self.tokenizer
            print(">>> Model loaded and cached.")

        self.adapters = {}
        self.class_names = class_names

    def register_adapter(self, adapter_name, system_prompt, user_template):
        """Registers a prompt template (Adapter) for a specific persona/task."""
        self.adapters[adapter_name] = {"system": system_prompt, "user": user_template}
        print(f"Adapter '{adapter_name}' registered.")

    # --- XAI METHODS ---

    def _get_shap_context(self, black_box_model, X_train, instance, pred_class, plot=True):
        """Generates Feature Importance context using SHAP."""
        explainer = shap.TreeExplainer(black_box_model)
        shap_values_obj = explainer(instance)

        # Handle SHAP output dimensions
        if len(shap_values_obj.shape) == 3:
             shap_values_for_plot = shap_values_obj[:, :, pred_class]
        else:
             shap_values_for_plot = shap_values_obj

        if plot:
            print("\n--- SHAP Plot (Feature Importance) ---")
            plt.figure()
            shap.plots.waterfall(shap_values_for_plot[0], show=False)
            plt.show()

        # Extract text description
        vals = shap_values_for_plot.values[0]
        feature_names = X_train.columns
        feature_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['feature', 'shap_value'])
        feature_importance['abs_value'] = feature_importance['shap_value'].abs()
        feature_importance = feature_importance.sort_values(by='abs_value', ascending=False).head(5)

        context_str = "ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS (SHAP):\n"
        for _, row in feature_importance.iterrows():
            input_val = instance[row['feature']].values[0]
            direction = "AUMENTA" if row['shap_value'] > 0 else "DISMINUYE"
            context_str += f"- La variable '{row['feature']}' (valor={input_val:.2f}) {direction} el riesgo (Impacto: {row['shap_value']:.4f}).\n"
        return context_str

    def _get_lime_context(self, black_box_model, X_train, instance, pred_class, plot=True):
        """Generates Local Weighted Rules using LIME."""
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=list(X_train.columns),
            class_names=[self.class_names[0], self.class_names[1]],
            mode='classification',
            discretize_continuous=True
        )
        exp = lime_explainer.explain_instance(
            data_row=instance.values[0],
            predict_fn=black_box_model.predict_proba,
            num_features=5
        )
        if plot:
            print("\n--- LIME Plot (Local Contribution) ---")
            exp.as_pyplot_figure()
            plt.show()

        lime_list = exp.as_list()
        context_str = "ANÁLISIS LIME (Reglas Locales Ponderadas):\n"
        for feature_rule, weight in lime_list:
            direction = "AUMENTA" if weight > 0 else "DISMINUYE"
            context_str += f"- La condición '{feature_rule}' {direction} la probabilidad (Peso: {weight:.4f}).\n"
        return context_str

    def _get_araucana_context(self, black_box_model, X_train, instance, pred_class, plot=True):
        """Generates Symbolic Logic Rules using Araucana XAI."""
        warnings.filterwarnings("ignore")
        feature_names = list(X_train.columns)

        def predict_wrapper(X_numpy):
            if isinstance(X_numpy, pd.DataFrame):
                return black_box_model.predict(X_numpy)
            df_temp = pd.DataFrame(X_numpy, columns=feature_names)
            return black_box_model.predict(df_temp)

        target_instance = instance.values.reshape(1, -1)
        target_pred = predict_wrapper(target_instance)
        cat_mask = [False] * len(feature_names)

        try:
            explanation_tree = araucanaxai.run(
                x_target=target_instance,
                y_pred_target=target_pred,
                x_train=X_train.values,
                feature_names=feature_names,
                cat_list=cat_mask,
                neighbourhood_size=200,
                oversampling='uniform',
                oversampling_size=2000,
                max_depth=3,
                predict_fun=predict_wrapper
            )
            tree_obj = explanation_tree['tree']

            # Fidelity Check (Audit)
            noise = np.random.normal(0, 0.5, (100, X_train.shape[1]))
            X_audit = target_instance + noise
            y_blackbox = predict_wrapper(X_audit)
            y_surrogate = tree_obj.predict(X_audit)
            fidelity_val = accuracy_score(y_blackbox, y_surrogate)

            if plot:
                print(f"\n--- Araucana Plot (Local Logic Tree) ---")
                plt.figure(figsize=(12, 6), dpi=100)
                plot_tree(
                    tree_obj,
                    feature_names=feature_names,
                    class_names=[str(c) for c in black_box_model.classes_],
                    filled=True,
                    rounded=True,
                    fontsize=9
                )
                plt.title(f"Araucana Local Rules (Audited Fidelity: {fidelity_val:.2f})")
                plt.show()

            tree_rules = export_text(tree_obj, feature_names=feature_names)
            context_str = f"ANÁLISIS DE LÓGICA SIMBÓLICA (ARAUCANA):\n"
            context_str += f"Fidelidad del explicador: {fidelity_val:.2f} (Si es bajo, tomar con precaución).\n"
            context_str += "Reglas extraídas:\n" + tree_rules

        except Exception as e:
            print(f"Araucana Error: {e}")
            context_str = "Error al extraer reglas lógicas."

        warnings.resetwarnings()
        return context_str

    # --- MAIN GENERATION METHOD ---

    def generate_explanation(self, black_box_model, X_train, instance, y_true=None, explainer_type="shap", adapter_name="default", plot=True):
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not found.")

        # 1. Model Prediction
        prediction_idx = black_box_model.predict(instance)[0]
        pred_label_str = self.class_names.get(prediction_idx, str(prediction_idx))

        # 2. Internal Audit (Displayed to user, HIDDEN from LLM to prevent bias)
        if y_true is not None:
            true_val = y_true.values[0] if hasattr(y_true, 'values') else y_true
            true_label_str = self.class_names.get(true_val, str(true_val))

            if true_val == prediction_idx:
                status_type = "CORRECT"
                detail = "(True Positive)" if prediction_idx == 1 else "(True Negative)"
            else:
                status_type = "INCORRECT (Model Failure)"
                detail = "(False Positive)" if prediction_idx == 1 else "(False Negative)"

            print(f"\n>>> [INTERNAL AUDIT]: Model Prediction: {pred_label_str} | Ground Truth: {true_label_str}")
            print(f">>> STATUS: {status_type} {detail}")
            print(">>> (This ground truth info is HIDDEN from the LLM to evaluate pure explanatory fidelity).")

        # 3. Context Generation (Handling multiple explainers)
        explainers_to_run = explainer_type.split("+")
        combined_context = ""

        for expl in explainers_to_run:
            expl = expl.strip().lower()
            if expl == "shap":
                txt = self._get_shap_context(black_box_model, X_train, instance, prediction_idx, plot=plot)
                combined_context += f"\n--- FUENTE 1: PESO DE VARIABLES (SHAP) ---\n{txt}\n"
            elif expl == "lime":
                txt = self._get_lime_context(black_box_model, X_train, instance, prediction_idx, plot=plot)
                combined_context += f"\n--- FUENTE: REGLAS PONDERADAS (LIME) ---\n{txt}\n"
            elif expl == "araucana":
                txt = self._get_araucana_context(black_box_model, X_train, instance, prediction_idx, plot=plot)
                combined_context += f"\n--- FUENTE 2: ESTRUCTURA LÓGICA (ARAUCANA) ---\n{txt}\n"

        # 4. Prompt Construction
        adapter = self.adapters[adapter_name]
        # NOTE: We do NOT pass 'status_info' to avoid data leakage.
        user_prompt = adapter["user"].format(
            prediction=pred_label_str,
            context_data=combined_context,
            input_features=instance.to_string(index=False)
        )

        # 5. LLM Inference
        messages = [{"role": "system", "content": adapter["system"]}, {"role": "user", "content": user_prompt}]
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        outputs = self.model.generate(inputs, max_new_tokens=600, use_cache=True)
        response = self.tokenizer.batch_decode(outputs)

        return response[0].split("<|start_header_id|>assistant<|end_header_id|>")[-1].replace("<|eot_id|>", "")