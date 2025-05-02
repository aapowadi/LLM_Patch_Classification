import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import os
import pdb

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data
# Assuming the data is provided as a CSV string or file; here we read from the document

file_systems = os.listdir('dataset')
data_list = []
for fs in file_systems:
    fs_path = os.path.join('dataset',fs)
    df = pd.read_csv(fs_path)  # Replace with actual file path if needed
    data_list.append(df)

data = pd.concat(data_list)
# Define feature columns
numerical_features = ['Files_Modified', 'Lines_Added', 'Lines_Removed']

# Encode categorical feature: Version
version_encoder = LabelEncoder()
data['Version'] = version_encoder.fit_transform(data['Version'])

# Normalize numerical features
data[numerical_features] = (data[numerical_features] - data[numerical_features].mean()) / data[numerical_features].std()

# Encode target variable: Patch_Type
patch_type_encoder = LabelEncoder()
print(data['Patch_Type'].value_counts())
data['Patch_Type'] = patch_type_encoder.fit_transform(data['Patch_Type'])

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Initialize BERT for text embeddings
text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text_model = AutoModel.from_pretrained("bert-base-uncased").to(device)

# Function to get text embeddings
def get_text_embeddings(texts):
    inputs = text_tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = text_model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embedding

# Prepare dataset for PyTorch
def prepare_dataset(df):
    desc_embeds = get_text_embeddings(df['Description'].tolist())
    comp_embeds = get_text_embeddings(df['Component'].tolist())
    numerical = torch.tensor(df[numerical_features].values, dtype=torch.float32).to(device)
    version = torch.tensor(df['Version'].values, dtype=torch.long).to(device)
    labels = torch.tensor(df['Patch_Type'].values, dtype=torch.long).to(device)
    return desc_embeds, comp_embeds, numerical, version, labels

# Prepare training and testing datasets
train_desc_embeds, train_comp_embeds, train_numerical, train_version, train_labels = prepare_dataset(train_data)
test_desc_embeds, test_comp_embeds, test_numerical, test_version, test_labels = prepare_dataset(test_data)

# Create TensorDatasets
train_dataset = TensorDataset(train_desc_embeds, train_comp_embeds, train_numerical, train_version, train_labels)
test_dataset = TensorDataset(test_desc_embeds, test_comp_embeds, test_numerical, test_version, test_labels)

# Create DataLoaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the classifier model
class PatchClassifier(nn.Module):
    def __init__(self, num_classes, d_model=768, num_versions=len(version_encoder.classes_)):
        super(PatchClassifier, self).__init__()
        self.version_embed = nn.Embedding(num_versions, d_model)
        self.fc_numerical = nn.Linear(len(numerical_features), d_model)
        self.classifier = nn.Linear(d_model * 3, num_classes)  # Combine desc, comp, and version+numerical

    def forward(self, desc_embeds, comp_embeds, numerical, version):
        version_embeds = self.version_embed(version)
        numerical_embeds = self.fc_numerical(numerical)
        combined = torch.cat([desc_embeds, comp_embeds, version_embeds + numerical_embeds], dim=1)
        logits = self.classifier(combined)
        return logits

# Initialize the model
num_classes = len(patch_type_encoder.classes_)
model = PatchClassifier(num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Training function
def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    for batch in train_loader:
        desc_embeds, comp_embeds, numerical, version, labels = [x.to(device) for x in batch]
        logits = model(desc_embeds, comp_embeds, numerical, version)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(train_loader)
    return avg_loss



# Assume these are defined: model, train_loader, test_loader, optimizer, device, patch_type_encoder

def evaluate(model, test_loader, return_per_class=False):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            desc_embeds, comp_embeds, numerical, version, labels = [x.to(device) for x in batch]
            logits = model(desc_embeds, comp_embeds, numerical, version)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    num_classes = len(patch_type_encoder.classes_)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    if return_per_class:
        per_class_f1 = f1_score(all_labels, all_preds, average=None, 
                              labels=range(num_classes), zero_division=0)
        return macro_f1, per_class_f1, all_labels, all_preds
    return macro_f1

def plot_class_f1_scores(class_names, per_class_f1, save_path='class_specific_f1_scores.png'):

    # Set figure size for one column (3.5 inches wide, 2.5 inches tall)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # Set font sizes to 10 pt for consistency with the paper
    plt.rc('font', size=10)          # Default text size
    plt.rc('axes', titlesize=10)     # Axes title font size
    plt.rc('axes', labelsize=10)     # X and Y label font size
    plt.rc('xtick', labelsize=10)    # X-axis tick label font size
    plt.rc('ytick', labelsize=10)    # Y-axis tick label font size
    plt.rc('legend', fontsize=10)    # Legend font size

    # Create bar plot
    bars = ax.bar(class_names, per_class_f1, color='skyblue')

    # Set axis labels and title
    ax.set_xlabel('Classes')
    ax.set_ylabel('F1 Score')
    ax.set_title('Class-Specific F1 Scores')
    ax.set_ylim(0, 1)  # Set y-axis range from 0 to 1

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')

    # # Add value labels on top of bars
    # for bar in bars:
    #     height = bar.get_height()
    #     ax.text(bar.get_x() + bar.get_width()/2., height,
    #             f'{height:.2f}', ha='center', va='bottom')

    # Adjust layout to prevent clipping
    plt.tight_layout()
    # Save the figure with high resolution
    plt.savefig(save_path, dpi=300)

def plot_conf_matrix(class_names, true_labels, predictions):
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=range(len(class_names)))

    # Set figure size for a single column (3.5 inches wide, 2.5 inches tall)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # Configure font sizes to 10 pt for consistency with the paper
    plt.rc('font', size=10)          # Default text size
    plt.rc('axes', titlesize=10)     # Axes title font size
    plt.rc('axes', labelsize=10)     # X and Y label font size
    plt.rc('xtick', labelsize=10)    # X-axis tick label font size
    plt.rc('ytick', labelsize=10)    # Y-axis tick label font size

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')

    # Add axis labels and title
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')

    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Save the figure with high resolution
    plt.savefig('confusion_matrix.png', dpi=300)



# Training loop with F1-score tracking
num_epochs = 30
train_losses = []
macro_f1_scores = []

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer)
    train_losses.append(train_loss)
    f1 = evaluate(model, test_loader)
    macro_f1_scores.append(f1)
    print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Test F1-score: {f1:.4f}")

# Plot and save F1-scores over epochs
def plot_f1_scores_over_epochs(macro_f1_scores, save_path='PatchDB_f1_scores_over_epochs.png'):
    # Set figure size for one column (3.5 inches wide, 2.5 inches tall)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # Set font sizes to 10 pt for consistency with the paper
    plt.rc('font', size=10)          # Default text size
    plt.rc('axes', titlesize=10)     # Axes title font size
    plt.rc('axes', labelsize=10)     # X and Y label font size
    plt.rc('xtick', labelsize=10)    # X-axis tick label font size
    plt.rc('ytick', labelsize=10)    # Y-axis tick label font size

    # Plot F1-scores
    epochs = range(1, len(macro_f1_scores) + 1)
    ax.plot(epochs, macro_f1_scores, marker='o', color='skyblue', linestyle='-', label='Macro F1-score')

    # Set axis labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Macro F1-score')
    ax.set_title('Macro F1-score Over Epochs')
    ax.set_ylim(0, 1)  # Set y-axis range from 0 to 1
    ax.legend()

    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Save the figure with high resolution
    plt.savefig(save_path, dpi=300)
    plt.close()

# Plot and save class-specific F1 scores and confusion matrix
print("Training completed. Plotting results.")
macro_f1, per_class_f1, all_labels, all_preds = evaluate(model, test_loader, return_per_class=True)
class_names = patch_type_encoder.classes_
plot_class_f1_scores(class_names, per_class_f1)
plot_conf_matrix(class_names, all_labels, all_preds)
plot_f1_scores_over_epochs(macro_f1_scores)