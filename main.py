import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Data/data/train.csv")

print("First five rows:")
print(data.head())

print("\nDataset information:")
data.info()

print("\nDataset shape:")
print(data.shape)

print("\nMissing values:")
print(data.isnull().sum())

data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(
    data["Embarked"].mode()[0]
)

print("\nSurvival count:")
print(data["Survived"].value_counts())

survival_rate = data["Survived"].mean() * 100
print("\nSurvival percentage:", round(survival_rate, 2), "%")

print("\nSurvival by gender:")
print(data.groupby("Sex")["Survived"].mean() * 100)

print("\nSurvival by passenger class:")
print(data.groupby("Pclass")["Survived"].mean() * 100)

print("\nAverage age:")
print(round(data["Age"].mean(), 2))

print("\nAverage fare:")
print(round(data["Fare"].mean(), 2))

sns.countplot(data=data, x="Survived")
plt.title("Titanic Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()

sns.countplot(data=data, x="Sex", hue="Survived")
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.show()

sns.countplot(data=data, x="Pclass", hue="Survived")
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.show()

