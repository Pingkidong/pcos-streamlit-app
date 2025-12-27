import pandas as pd

df = pd.read_csv('pcos.csv')
# print(df.head())
# print(df.info())
# print(df.shape)
# print(df.columns)
# print(df.describe())
# print(df['Menstrual_Irregularity'].unique())

# outliner & correlation check
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,8))
df.hist()
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df[['BMI','Testosterone_Level(ng/dL)','Antral_Follicle_Count']])
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# distribution check
print(df['PCOS_Diagnosis'].value_counts())
print(df['PCOS_Diagnosis'].value_counts(normalize=True))

# stratified split
from sklearn.model_selection import train_test_split

X = df.drop('PCOS_Diagnosis', axis = 1)
y = df['PCOS_Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42,stratify=y)

# scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000
)

model.fit(X_train_scaled, y_train)

# evaluation
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test_scaled)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# cross-validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='recall')

print(scores)
print(scores.mean())

# export
import joblib

joblib.dump(model, 'pcos_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
