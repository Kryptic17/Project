import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import copy
torch.manual_seed(42)
df = pd.read_csv('Indian_IPO_Market_Data.csv')
print("First 5 rows: ", df.head())
print("\nLast 5 rows: ", df.tail())
print(f"We have {df.shape[0]} IPOs and {df.shape[1]} columns")
print("\nColumnInformation:")
df.info()
print("\nTarget Variable (Listing_Gains_Percent) summary:")
print(df['Listing_Gains_Percent'].describe())
print("\nSummary statistics for all numeric columns:")
print(df.describe())
df['Listing_Gains_Profit'] = (df['Listing_Gains_Percent'] > 0).astype(int)
print("First few rows showing original vs binary target:")
print(df[['Listing_Gains_Percent', 'Listing_Gains_Profit']].head(10))
print("Distribution of IPO profitability:")
target_counts = df['Listing_Gains_Profit'].value_counts()
print(target_counts)
target_percentages = df['Listing_Gains_Profit'].value_counts(normalize=True) * 100
print(f"\nAs percentages:")
print(f"Profitable IPOs (1): {target_percentages[1]:.1f}%")
print(f"Non-Profitable IPOs (0): {target_percentages[0]:.1f}%")
profitable_percentage = (df['Listing_Gains_Profit'] == 1).mean() * 100
print(f"\nExact percentage of profitable IPOs (1): {profitable_percentage:.2f}%")
print("Comparison of profitable vs non-profitable IPOs:")
print(f"\nProfitable IPOs - Average listing gain: {df[df['Listing_Gains_Profit'] == 1]['Listing_Gains_Percent'].mean():.2f}%")
print(f"\nNon-Profitable IPOs - Average listing loss: {df[df['Listing_Gains_Profit'] == 0]['Listing_Gains_Percent'].mean():.2f}%")
print(f"\nLargest Gain: {df['Listing_Gains_Percent'].max():.2f}%")
print(f"\nLargest Loss: {df['Listing_Gains_Percent'].min():.2f}%")
print("ALl columns in the dataset: ")
print(list(df.columns))
exclude_columns = ['Date ', 'IPOName', 'Listing_Gains_Percent', 'Listing_Gains_Profit']
potential_features = [col for col in df.columns if col not in exclude_columns]
print("\nPotential columns:")
print(potential_features)
print("\nNumber of potential features:")
print(len(potential_features))
plt.figure(figsize = (8,6))
target_counts = df['Listing_Gains_Profit'].value_counts()
plt.bar(['Not Profitable', 'Profitable'], target_counts.values)
plt.title('Distribution of IPO profitability')
plt.ylabel('Number of IPOs')
plt.xlabel('Outcome')
for i, count in enumerate(target_counts.values):
    plt.text(i, count - 10, str(count), ha='center', fontsize=12)
plt.show()
print(f"We have a fairly balanced dataset: {target_counts[1]} profitable vs {target_counts[0]} non-profitable IPOs:")
fig, axes = plt.subplots(2,3 , figsize = (15,10))
axes=axes.flatten()
for i, column in enumerate(potential_features):
    axes[i].hist(df[column].dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[i].set_title(f'Distribution of {column}')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Frequency')
plt.tight_layout()
plt.show()
fig, axes = plt.subplots(2,3 , figsize = (15,10))
axes=axes.flatten()
for i, column in enumerate(potential_features):
    profitable_data = df[df['Listing_Gains_Profit'] == 1][column]
    not_profitable_data = df[df['Listing_Gains_Profit'] == 0][column]
    axes[i].boxplot([not_profitable_data.dropna(), profitable_data.dropna()], tick_labels=['Not Profitable', 'Profitable'])
    axes[i].set_title(f'{column} by IPO Outcome')
    axes[i].set_ylabel(column)
plt.tight_layout()
plt.show()
numeric_features = df[potential_features].select_dtypes(include=[np.number])
correlation_matrix = numeric_features.corr()
plt.figure(figsize = (10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center = 0, square = True, fmt= '.2f', cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Matrix of IPO features')
plt.tight_layout()
plt.show()
target_correlations = df[potential_features + ['Listing_Gains_Profit']].corr()['Listing_Gains_Profit'].sort_values(ascending=False)
print("Correlation with IPO profitability:")
print(target_correlations[1:])
feature_columns = ['Subscription_QIB', 'Subscription_HNI', 'Subscription_RII', 'Issue_Price', 'Issue_Size']
X = df[feature_columns].copy()
y = df['Listing_Gains_Profit'].copy()
skewed_columns = ['Subscription_QIB', 'Subscription_HNI', 'Subscription_RII', 'Issue_Price', 'Issue_Size']
for col in skewed_columns:
    X[col] = np.log1p(X[col])
print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size = 0.2,stratify = y, random_state = 42)
print("After first split:")
print(f"Temp data (train+val): {X_temp.shape[0]} samples")
print(f"Test data: {X_test.shape[0]} samples")
print(f"Test Target Distribution: {y_test.mean():.1%} profitable")
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size = 0.25,stratify=y_temp ,random_state = 42)
print("Final split breakdown:")
print(f'Training: {X_train.shape[0]} samples ({X_train.shape[0]/len(df):.1%})')
print(f'Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(df):.1%})')
print(f'Test: {X_test.shape[0]} samples ({X_test.shape[0]/len(df):.1%})')
print("\nTarget Distributions:")
print(f"Training: {y_train.mean():.1%} profitable")
print(f"Validation: {y_val.mean():.1%} profitable")
print(f"Test: {y_test.mean():.1%} profitable")
scaler = StandardScaler()
scaler.fit(X_train)
print("Scanning parameters learned from training data: ")
print(f"Feature means: {scaler.mean_}")
print(f"Feature stds: {scaler.scale_}")
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns = feature_columns)
X_val_scaled = pd.DataFrame(X_val_scaled, columns = feature_columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns = feature_columns)
print("Scaled datasets shapes: ")
print(f" X_train_scaled: {X_train_scaled.shape}")
print(f" X_val_scaled: {X_val_scaled.shape}")
print(f" X_test_scaled: {X_test_scaled.shape}")
print("Training data after scaling (should be ~0 mean and ~1 std):")
print(f" Means: {X_train_scaled.mean().round(3).values}")
print(f" Stds: {X_train_scaled.std().round(3).values}")
print("\nValidation data after scaling (will be slightly different):")
print(f"Means: {X_val_scaled.mean().round(3).values}")
print("\nTest data after scaling (will be slightly different):")
print(f"Means: {X_test_scaled.mean().round(3).values}")
print("Everything looks good")
X_train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
print("Tensor shapes:")
print(f" X_train_tensor: {X_train_tensor.shape}")
print(f" X_val_tensor: {X_val_tensor.shape}")
print(f" X_test_tensor: {X_test_tensor.shape}")
print(f" y_train_tensor: {y_train_tensor.shape}")
print(f" y_val_tensor: {y_val_tensor.shape}")
print(f" y_test_tensor: {y_test_tensor.shape}")
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
print("Created datasets:")
print(f" Training dataset: {len(train_dataset)} samples")
print(f" Validation dataset: {len(val_dataset)} samples")
print(f" Test dataset: {len(test_dataset)} samples")
sample_features, sample_target = train_dataset[0]
print("Sample from training dataset:")
print(f' Features shape: {sample_features.shape}')
print(f' Target: {sample_target.item()}')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
print("Dataloaders created:")
print(f" Training batches: {len(train_loader)}")
print(f" Validation batches: {len(val_loader)}")
print(f" Test batches: {len(test_loader)}")
for batch_features, batch_targets in train_loader:
    print("Sample training batch:")
    print(f" Batch features shape: {batch_features.shape}" )
    print(f" Batch targets shape: {batch_targets.shape}")
    print(f" Features dtype: {batch_features.dtype}")
    print(f" Targets dtype: {batch_targets.dtype}")
    print(f" Target values in batch: {batch_targets.tolist()}")
    print(f" Profitable IPOs in batch: {(batch_targets==1).sum().item()}/{len(batch_targets)}")
    break
print(f"\nPerfect! With {len(train_loader)} training batches and {len(val_loader)} validation batches, we're ready to start building our neural network.")
input_size = X_train_tensor.shape[1]
print(f" Our model needs to handle {input_size} input features")
model = nn.Sequential(
    nn.Linear(input_size, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)
print("Model created successfully.")
print(model)
total_params = sum(param.numel() for param in model.parameters())
trainable_params =sum(param.numel() for param in model.parameters() if param.requires_grad)
print(f" Total parameters: {total_params}")
print(f" Trainable parameters: {trainable_params}")
print("\nParameter breakdown by layers:")
for i, layer in enumerate(model):
    if isinstance(layer, nn.Linear):
        layer_params = sum(param.numel() for param in layer.parameters())
        print(f" Layer {i} (Linear): {layer_params:,} parameters")
model.eval()
with torch.no_grad():
    for batch_features, batch_targets in train_loader:
        test_outputs = model(batch_features)
        print(f" Input batch shape: {batch_features.shape}")
        print(f" Output batch shape: {test_outputs.shape}")
        print(f" Sample predictions: {test_outputs.squeeze()[:5].tolist()}")
        print(f" Sample actual targets: {batch_targets[:5].tolist()}")
        break
print(f"\nPerfect! Our model can process batches of {batch_features.shape[0]} IPOs and produce probability predictions between 0 and 1.")
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.3))
optimizer = optim.Adam(model.parameters(), lr=0.003)
train_accuracies = []
val_accuracies = []
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch_features, batch_targets in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_features).squeeze()
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        train_outputs = torch.sigmoid(model(X_train_tensor).squeeze())
        train_predictions = (train_outputs > 0.45).float()
        train_accuracy = (train_predictions == y_train_tensor).float().mean().item() * 100
        val_correct = 0
        val_total = 0
        for val_features, val_targets in val_loader:
            val_outputs = torch.sigmoid(model(val_features).squeeze())
            val_predictions = (val_outputs > 0.45).float()
            val_correct += (val_predictions == val_targets).sum().item()
            val_total += val_targets.size(0)
        val_accuracy = (val_correct / val_total) * 100
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
print("Training complete")
print(f" Final training accuracy: {train_accuracies[-1]:.1f}%")
print(f" Final validation accuracy: {val_accuracies[-1]:.1f}%")
plt.figure(figsize=(10,6))
plt.plot(range(1, num_epochs + 1), train_accuracies, label="Training Accuracy", color = "blue")
plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy", color = "red")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training Progress: Accuracy Over Time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
best_val_accuracy = max(val_accuracies)
best_epoch = val_accuracies.index(best_val_accuracy) + 1
print(f"Best validation accuracy: {best_val_accuracy:.1f}% (achieved at epoch {best_epoch})")
final_gap = train_accuracies[-1] - val_accuracies[-1]
print(f"Final gap between training and validation: {final_gap:.1f}%")
if final_gap > 10:
    print("Large gap suggests possible overfitting - validation accuracy much lower than training")
elif final_gap < 5:
    print("Small gap suggests good generalization - model performs similarly on both sets")
else:
    print("Moderate gap - model might benefit from some regularization")
reg_model = nn.Sequential(
    nn.Linear(input_size, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(64, 32),
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.Dropout(0.125),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)
print("Regularized model created")
print(f" Total parameters: {sum(param.numel() for param in reg_model.parameters()),}")
print(reg_model)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.3))
optimizer = optim.Adam(reg_model.parameters(), lr=0.003)
best_val_accuracy = 0
patience = 20
patience_counter = 0
best_model_state = None
reg_train_accuracies = []
reg_val_accuracies = []
print("Setting up early stopping with patience of", patience)
print("Training will stop if validation accuracy doesn't improve for", patience, " consecutive epochs")
max_epochs = 100
print(f" Start training (up to {max_epochs} epochs with early stopping...)")
for epoch in range(max_epochs):
    reg_model.train()
    for batch_features, batch_targets in train_loader:
        optimizer.zero_grad()
        outputs = reg_model(batch_features).squeeze()
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
    reg_model.eval()
    with torch.no_grad():
        train_outputs = torch.sigmoid(reg_model(X_train_tensor).squeeze())
        train_predictions = (train_outputs > 0.45).float()
        train_accuracy = (train_predictions == y_train_tensor).float().mean().item() * 100
        val_correct = 0
        val_total = 0
        for val_features, val_targets in val_loader:
            val_outputs = torch.sigmoid(reg_model(val_features).squeeze())
            val_predictions = (val_outputs > 0.45).float()
            val_correct += (val_predictions == val_targets).sum().item()
            val_total += val_targets.size(0)
        val_accuracy = (val_correct / val_total) * 100
        reg_train_accuracies.append(train_accuracy)
        reg_val_accuracies.append(val_accuracy)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = copy.deepcopy(reg_model.state_dict())
            patience_counter = 0
            print(f" Epoch {epoch + 1}: New best validation accuracy: {val_accuracy:.1f}%")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f" Early stopping at Epoch {epoch + 1}: (no improvement for {patience} epochs)")
            break
reg_model.load_state_dict(best_model_state)
print(f" Loaded best model with validation accuracy {best_val_accuracy:.1f}%")
print("Model comparison")
print(f" Original model Final training accuracy: {train_accuracies[-1]:.1f}%")
print(f" Original mode Final validation accuracy: {val_accuracies[-1]:.1f}%")
print(f" Original model Gap: {train_accuracies[-1] - val_accuracies[-1]:.1f}%")
print(f" Regularized model Final training accuracy: {reg_train_accuracies[-1]:.1f}%")
print(f" Regularized model Final validation accuracy: {reg_val_accuracies[-1]:.1f}%")
print(f" Regularized model Gap: {reg_train_accuracies[-1] - reg_val_accuracies[-1]:.1f}%")
improvement = reg_val_accuracies[-1] - val_accuracies[-1]
print(f" \nValidation accuracy improvement: {improvement:+.1f}%")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
ax1.plot(range(1,len(train_accuracies) + 1), train_accuracies, label='Training', color='blue')
ax1.plot(range(1,len(val_accuracies) + 1), val_accuracies, label='Validation', color='red')
ax1.set_title('Original Model (Overfitting)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax2.plot(range(1, len(reg_train_accuracies) + 1), reg_train_accuracies, label='Training', color='blue')
ax2.plot(range(1, len(reg_val_accuracies) + 1), reg_val_accuracies, label='Validation', color='red')
ax2.set_title('Regularized Model (Better Generalization)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
final_gap = reg_train_accuracies[-1] - reg_val_accuracies[-1]
if final_gap < 5:
    print("Excellent very small gap between training and validation accuracy")
elif final_gap < 10:
    print("Good improvement, much better generalization than before")
else:
    print("Still some overfitting but improvement than before")
print(f" Our regularized model stopped early at epoch {len(reg_train_accuracies)} instead of running the full 100 epochs")
print("This saved us from overtraining and gave us a model that should work better on new IPOs!")
reg_model.eval()
with torch.no_grad():
    test_outputs = torch.sigmoid(reg_model(X_test_tensor).squeeze())
    test_predictions = (test_outputs > 0.45).float()
print(f" The regularized model made predictions on {len(y_test_tensor)} test IPOs")
predicted_profitable = test_predictions.sum().item()
print(f" It predicted {int(predicted_profitable)} would be profitable and {int(len(test_predictions) - predicted_profitable)} would not")
actually_profitable = y_test_tensor.sum().item()
print(f" In reality, {int(actually_profitable)} were profitable and {int(len(y_test_tensor) - actually_profitable)} were not")
print(f"\nFirst 10 test predictions vs actual:")
for i in range(10):
    pred = "Profitable" if test_predictions[i] == 1 else "Not Profitable"
    actual = "Profitable" if y_test_tensor[i] == 1 else "Not Profitable"
    match = "right" if test_predictions[i] == y_test_tensor[i] else "wrong"
    print(f" IPO {i+1}: Predicted {pred} and Actual {actual}: {match} ")
cm = confusion_matrix(y_test_tensor.numpy(), test_predictions.numpy())
print("      Predicted")
print("      Not Profitable  Profitable")
print(f"Actual Not Prof  {cm[0,0]}   {cm[0,1]}")
print(f"Actual Prof  {cm[1,0]}   {cm[1,1]}")
tn, fp, fn, tp = cm[0,0].item(), cm[0,1].item(), cm[1,0].item(), cm[1,1].item()
print(f"\nBreaking this down:")
print(f"True Negatives (TN): {tn} - Correctly predicted Not Profitable")
print(f"False Positives (FP): {fp} - Wrongly predicted Profitable")
print(f"False Negatives (FN): {fn} - Wrongly predicted Not Profitable")
print(f"True Positives (TP): {tp} - Correctly predicted Profitable")
print(f"\nOut of {tn + fp + fn + tp} total predictions:")
print(f"We got {tn + tp} right and {fp + fn} wrong.")
precision = tp / (tp + fp)
recall = tp / (tp + fn)
print("Key Business Metrics:")
print(f"Precision: {precision:.3f} ({precision * 100:.1f}%)")
print(f" When we predict 'profitable', we're right {precision*100:.1f}% of the time")
print(f"Recall: {recall:.3f} ({recall * 100:.1f})%")
print(f" We successfully identify {recall*100:.1f}% of all profitable IPOs")
test_accuracy = (tn+tp)/(tp+tn+fp+fn) * 100
print(f"\nTest Accuracy: {test_accuracy:.1f}%")
print(f"Validation Accuracy (best): {best_val_accuracy:.1f}%")
accuracy_diff = abs(test_accuracy - best_val_accuracy)
if accuracy_diff < 3:
    print(f"Excellent Test and Validation accuracy are very close ({accuracy_diff:.1f}% difference) ")
    print("This suggests our model generalizes well to completely new data")
elif accuracy_diff < 7:
    print(f"Good test and validation accuracy are reasonably close ({accuracy_diff:.1f}% difference) ")
else:
    print(f" Test accuracy differs from validation by {accuracy_diff:.1f}% might indicate some overfitting")
print("Business impact analysis:")
print("="*50)
print(f"\nFalse positives ({fp} cases): We predicted profitable but IPO lost money")
print(" Business impact: We invest in bad IPOs and lose money.")
print("This hurts our returns directly, we put money into losing investments")
print(f"\nFalse negatives ({fn} cases): We predicted not profitable but IPO made money")
print(" Business impact: We miss out on good investment opportunities.")
print("This is opportunity cost, we could've made money but didn't.")
print(f"\nFor a typical investment firm:")
if fp > fn:
    print("We're making more 'bad investment' errors than 'missed opportunity' errors")
    print("Consider raising our threshold above 0.5 to be more selective")
elif fn > fp:
    print("We're missing more opportunities than making bad investments")
    print("Consider lowering our threshold below 0.5 to catch more profitable IPOs")
else:
    print("We have a balanced error profile")
print(f" Precision tells our clients, 'when we recommend an IPO, we're right {precision*100:.1f}% of the time' ")
print(f"Recall tells us, 'We find {recall*100:.1f}% of all the profitable opportunities in the market' ")
if precision > 0.65 and recall > 0.65:
    print(f" Solid performance! Decent precision and decent recall")
elif precision > 0.7:
    print(f" Conservative but reliable, when we say buy, we're usually right")
elif recall > 0.7:
    print(f"\n Great at finding opportunities, we dont miss many profitable IPOs ")
else:
    print(f"Room for improvement, but this is a solid foundation to build on")
print(f" \nBottom line: Our model correctly predicts IPO outcomes {test_accuracy:.1f}% of the time")


