import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("Data/data/train.csv")

# Check dataset
print("First five rows:")
print(data.head())

print("\nDataset information:")
data.info()

print("\nDataset shape:")
print(data.shape)

# Check missing values
print("\nMissing values:")
print(data.isnull().sum())

# Handle missing values
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(
    data["Embarked"].mode()[0]
)

# Survival count
print("\nSurvival count:")
print(data["Survived"].value_counts())

# Survival percentage
survival_rate = data["Survived"].mean() * 100
print("\nSurvival percentage:", round(survival_rate, 2), "%")

# Survival by gender
print("\nSurvival by gender:")
print(data.groupby("Sex")["Survived"].mean() * 100)

# Survival by passenger class
print("\nSurvival by passenger class:")
print(data.groupby("Pclass")["Survived"].mean() * 100)

# Average age of passengers
print("\nAverage age:")
print(round(data["Age"].mean(), 2))

# Average fare
print("\nAverage fare:")
print(round(data["Fare"].mean(), 2))

# Graph 1: Survival count
sns.countplot(data=data, x="Survived")
plt.title("Titanic Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()

# Graph 2: Survival by gender
sns.countplot(data=data, x="Sex", hue="Survived")
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.show()

# Graph 3: Survival by passenger class
sns.countplot(data=data, x="Pclass", hue="Survived")
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.show()

