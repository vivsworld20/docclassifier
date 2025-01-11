import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Custom Dataset
class LegalDocDataset(Dataset):
    def __init__(self, documents, labels, tokenizer, max_len):
        self.documents = documents
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        doc = self.documents[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            doc,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# The BERT model for classification
class LegalDocClassifier(nn.Module):
    def __init__(self, num_classes):
        super(LegalDocClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropped_out = self.dropout(pooled_output)
        logits = self.fc(dropped_out)
        return logits

# Training the model
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# Evaluating the model
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == labels).sum().item()

    accuracy = correct_predictions / len(dataloader.dataset)
    return total_loss / len(dataloader), accuracy

# Example data
documents = [
    "Contract for the sale of goods.",
    "Last will and testament of Jane Doe.",
    "Property transfer deed of trust.",
    "Confidentiality agreement between parties.",
    "Termination clause for contract.",
    "Residential lease agreement."
]
labels = [0, 1, 2, 0, 0, 0]  # Example Labels

# Tokenizer and Dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 128

train_docs, val_docs, train_labels, val_labels = train_test_split(documents, labels, test_size=0.2, random_state=42)

train_dataset = LegalDocDataset(train_docs, train_labels, tokenizer, max_len)
val_dataset = LegalDocDataset(val_docs, val_labels, tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(set(labels))
model = LegalDocClassifier(num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, optimizer, criterion, device)
    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
    print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

# Clause matching (just added this part)
class ClauseMatcher:
    def __init__(self):
        # Load pre-trained BERT model
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_embedding(self, text):
        # Tokenize and get embeddings
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs.last_hidden_state[:, 0, :].squeeze()

    def find_matching_clause(self, document, clause):
        # Find the matching clause in the document
        clause_embedding = self.get_embedding(clause)

        # Split document into paragraphs
        paragraphs = document.split("\n")

        best_match = None
        best_similarity = -1

        for paragraph in paragraphs:
            paragraph_embedding = self.get_embedding(paragraph)
            similarity = cosine_similarity(clause_embedding.unsqueeze(0), paragraph_embedding.unsqueeze(0))[0][0]

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = paragraph

        return best_match, best_similarity

# Example usage
document = """
This contract includes a confidentiality clause and a termination clause.
The parties agree to keep all information confidential.
Termination of the contract will occur if either party breaches the agreement.
"""

clause = "confidentiality clause"

matcher = ClauseMatcher()
matching_paragraph, similarity = matcher.find_matching_clause(document, clause)

if matching_paragraph:
    print(f"Found matching paragraph with similarity {similarity:.4f}:")
    print(matching_paragraph)
else:
    print("No matching paragraph found.")
