
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class ModerationClassifier:
    
    def __init__(self, model_name="ifmain/ModerationBERT-En-02"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        from transformers import BertTokenizer, BertForSequenceClassification
        # Load model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=18)
        
        # Device setup
        self.model.to(self.device)
        self.model.eval()
        
        # Define category labels
        self.cm_risk_categories = [
            'harassment', 'harassment_threatening', 'H', 'H2', 
            'SH', 'self_harm_instructions', 'self_harm_intent', 'S', 
            'S3', 'V', 'V3', 'self-harm', 
            'sexual/minors', 'hate/threatening', 'violence/graphic', 
            'self-harm/intent', 'self-harm/instructions', 'harassment/threatening'
        ]

    def predict(self, text, max_len=128, stride=32):
        # Tokenize text with overflow handling
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_len,
            stride=stride,
            return_overflowing_tokens=True,
            return_attention_mask=True,
            padding='max_length'
        )

        model_inputs = {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }

        with torch.no_grad():
            outputs = self.model(**model_inputs)
        
        # Multi-label sigmoid
        probabilities = torch.sigmoid(outputs.logits)
        
        # Aggregate across chunks
        max_probabilities, _ = torch.max(probabilities, dim=0)

        predictions = {
            self.cm_risk_categories[i]: max_probabilities[i].item()
            for i in range(len(max_probabilities))
        }

        return predictions
    
    def predict_S_V_SH(self, text):
        """Convenience wrapper to extract key moderation signals."""
        preds = self.predict(text)
        return preds.get('S', 0.0), preds.get('S3', 0.0), preds.get('V', 0.0), preds.get('SH', 0.0)
    
    def predict_S_V(self, text):
        """Convenience wrapper to extract key moderation signals."""
        preds = self.predict(text)
        return preds.get('S', 0.0), preds.get('V', 0.0)