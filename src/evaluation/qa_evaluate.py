import numpy as np  
from transformers import pipeline  
from datasets import load_dataset  
from rouge_score import rouge_scorer  
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  
from bert_score import score as bert_score  
import torch  
from tqdm import tqdm
import nltk  

# Download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')  
    
    
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity  
import nltk  
from nltk.corpus import stopwords  

# Download NLTK stopwords if not already present    
try:  
    nltk.data.find('corpora/stopwords')  
except LookupError:  
    nltk.download('stopwords')  



from src.configs.config import (
    DATA_PATH
)


class QAEvaluator():
    """
    Question-Answer evaluation class for measuring model performance on QA tasks.

    This evaluator computes multiple metrics including ROUGE, BLEU, and BERTScore
    to comprehensively assess the quality of generated answers.
    """
    
    def __init__(self, model, tokenizer, max_seq_length, eval_dataset=None):
        """
        Initialize the QA evaluator.

        Args:
            model: The language model to evaluate
            tokenizer: Tokenizer corresponding to the model
            max_seq_length: Maximum sequence length for generation
            eval_dataset: Optional evaluation dataset with Question/Response pairs
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device  = self.model.device
        self.max_seq_length = max_seq_length

        self.eval_dataset = eval_dataset

    def compute_metrics(self, eval_preds):
        """
        Evaluate model performance on QA dataset.

        Args:
            eval_preds: Evaluation predictions containing predictions and label_ids

        Returns:
            Dictionary containing various evaluation metrics
        """  
        assert self.model is not None, "Model is required for QA evaluation"  
        
        # Use the provided model for evaluation    
        model_to_evaluate = self.model
        
        # Get evaluation dataset from trainer or use default
        if not hasattr(self, "eval_dataset") or self.eval_dataset is None:
            print("Warning: Evaluation dataset not configured, using example data for metrics calculation")    
            eval_dataset = {  
                    "Question": [  
                        "What are the best attractions to visit in Paris?",  
                        "How can I get from London to Edinburgh by train?",  
                        "What is the local cuisine in Thailand?",  
                        "What's the best time to visit Japan?"  
                    ],  
                    "Response": [  
                        "The top attractions in Paris include the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and Montmartre.",  
                        "You can take a direct train from London King's Cross to Edinburgh Waverley. The journey takes about 4.5 hours.",  
                        "Thai cuisine features dishes like Pad Thai, Tom Yum Goong, Green Curry, and Mango Sticky Rice.",  
                        "Spring (March to May) and autumn (September to November) are the best times to visit Japan."  
                    ]  
                }  
        
        else:
            eval_dataset = self.eval_dataset 
        
                
        
        # CreateQA pipeline  
        qa_pipeline = pipeline(  
            "text-generation",  
            model=model_to_evaluate,  
            tokenizer=self.tokenizer,  
            max_length=self.max_seq_length,  
            do_sample=False  # Use greedy decoding for reproducibility    
        )  
        
        # Initialize metric calculators    
        rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)  
        smoothing = SmoothingFunction().method1  
        
        # Initialize result storage  
        results = {  
            "rouge1_f1": [],  
            "rouge2_f1": [],  
            "rougeL_f1": [],  
            "bleu1": [],  
            "bleu2": [],  
            "bleu4": [],  
            "bert_score_precision": [],  
            "bert_score_recall": [],  
            "bert_score_f1": []  
        }  
        
        generated_responses = []  
        reference_responses = []  
        
        # Generate answers for each question and compute metrics    
        for i, (question, reference) in enumerate(zip(eval_dataset["Question"], eval_dataset["Response"])):  
            # Format input prompt    
            formatted_input = f"Question: {question}\nAnswer:"  
            
            # Generate answer    
            generated_output = qa_pipeline(formatted_input)[0]["generated_text"]  
            
            # Extract generated answer part (remove original question)    
            if "Answer:" in generated_output:  
                generated_answer = generated_output.split("Answer:", 1)[1].strip()  
            else:  
                generated_answer = generated_output.replace(formatted_input, "").strip()  
            
            # Store generated and reference answers    
            generated_responses.append(generated_answer)  
            reference_responses.append(reference)  
            
            # Calculate ROUGE scores    
            rouge_scores = rouge_scorer_instance.score(reference, generated_answer)  
            results["rouge1_f1"].append(rouge_scores["rouge1"].fmeasure)  
            results["rouge2_f1"].append(rouge_scores["rouge2"].fmeasure)  
            results["rougeL_f1"].append(rouge_scores["rougeL"].fmeasure)  
            
            # Calculate BLEU scores    
            reference_tokens = nltk.word_tokenize(reference.lower())  
            generated_tokens = nltk.word_tokenize(generated_answer.lower())  
            
            results["bleu1"].append(sentence_bleu([reference_tokens], generated_tokens,   
                                                weights=(1, 0, 0, 0),   
                                                smoothing_function=smoothing))  
            results["bleu2"].append(sentence_bleu([reference_tokens], generated_tokens,   
                                                weights=(0.5, 0.5, 0, 0),   
                                                smoothing_function=smoothing))  
            results["bleu4"].append(sentence_bleu([reference_tokens], generated_tokens,   
                                                weights=(0.25, 0.25, 0.25, 0.25),   
                                                smoothing_function=smoothing))  
            
            # Print progress every 100 samples    
            if (i+1) % 100 == 0:  
                print(f"Processed {i+1}/{len(eval_dataset['Question'])} samples")    
        
        # Calculate BERTScore (batch computation for efficiency)    
        try:  
            P, R, F1 = bert_score(generated_responses, reference_responses, lang="en", rescale_with_baseline=True)  
            results["bert_score_precision"] = P.tolist()  
            results["bert_score_recall"] = R.tolist()  
            results["bert_score_f1"] = F1.tolist()  
        except Exception as e:  
            print(f"Error calculating BERTScore: {e}")    
            # Use placeholder values    
            results["bert_score_precision"] = [0.0] * len(generated_responses)  
            results["bert_score_recall"] = [0.0] * len(generated_responses)  
            results["bert_score_f1"] = [0.0] * len(generated_responses)  
        
        # Calculate average value for each metric    
        aggregated_results = {}  
        for metric_name, values in results.items():  
            aggregated_results[metric_name] = np.mean(values)  
        
        # Add sample-level results for detailed analysis    
        aggregated_results["sample_results"] = {  
            "questions": eval_dataset["Question"],  
            "references": reference_responses,  
            "generated": generated_responses,  
            "metrics": {k: v for k, v in results.items()}  
        }  
        
        # Calculate additional evaluation metrics: QA accuracy (if dataset contains multiple acceptable answers)    
        if "Acceptable_Answers" in eval_dataset:  
            correct_answers = 0  
            for i, (gen_answer, acceptable_answers) in enumerate(zip(generated_responses, eval_dataset["Acceptable_Answers"])):  
                # Check if generated answer matches any acceptable answer    
                if any(self._answer_match(gen_answer, acc_answer) for acc_answer in acceptable_answers):  
                    correct_answers += 1  
            
            aggregated_results["answer_accuracy"] = correct_answers / len(generated_responses)  
        
        return aggregated_results  
    
    
    
    
    
    
    
    def _answer_match(self, generated_answer, reference_answer, threshold=0.7):  
        """
        Check if generated answer matches reference answer using semantic similarity.

        Args:
            generated_answer: The generated answer text
            reference_answer: The reference answer text
            threshold: Similarity threshold above which answers are considered matching

        Returns:
            Boolean value indicating whether answers match
        """  

        
        # Get English stopwords    
        stop_words = set(stopwords.words('english'))  
        
        # Text preprocessing function    
        def preprocess(text):  
            tokens = nltk.word_tokenize(text.lower())  
            return ' '.join([w for w in tokens if w.isalnum() and w not in stop_words])  
        
        processed_gen = preprocess(generated_answer)  
        processed_ref = preprocess(reference_answer)  
        
        # Return False if either text is empty    
        if not processed_gen or not processed_ref:  
            return False  
        
        # Calculate TF-IDF features and cosine similarity    
        vectorizer = TfidfVectorizer()  
        try:  
            tfidf_matrix = vectorizer.fit_transform([processed_gen, processed_ref])  
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]  
            return similarity >= threshold  
        except:  
            # Return False if vectorization fails (e.g., empty vocabulary)    
            return False  