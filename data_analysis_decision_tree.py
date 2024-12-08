import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Step 1: Load the dataset
file_path = './dataset/winequality-red.csv'
df = pd.read_csv(file_path, sep=';')

# Step 2: Preprocess the data
X = df.drop('quality', axis=1)  # Features
y = df['quality']  # Target variable

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Step 6: Visualize the tree (save as image)
plt.figure(figsize=(240, 100))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=sorted(y.unique().astype(str)),
    filled=True,
    fontsize=8
)
plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
plt.savefig('decision_tree.svg', format="svg") 
print("Decision tree plot saved as 'decision_tree.svg'")
