# Student Placement Prediction Using Machine Learning

## 📌 Project Overview

This project predicts whether a student is likely to be **Placed** or **Not Placed** using Machine Learning.

The system uses student academic performance, technical skills, internships, projects, communication skills, and other factors to make the prediction.

## 🎯 Objective

To develop an end-to-end Machine Learning system that:

- Analyzes student placement data
- Performs data preprocessing and EDA
- Trains multiple Machine Learning models
- Compares model performance
- Selects the best model
- Predicts placement status for a new student
- Deploys the model as a web application using FastAPI and Render

## 📊 Dataset

The dataset contains **100,000 student records** and **26 columns**.

Important features include:

- Age
- Gender
- CGPA
- Branch
- College Tier
- Internships Count
- Projects Count
- Certifications Count
- Coding Skill Score
- Aptitude Score
- Communication Skill Score
- Logical Reasoning Score
- Hackathons Participated
- GitHub Repositories
- LinkedIn Connections
- Mock Interview Score
- Attendance Percentage
- Backlogs
- Extracurricular Score
- Leadership Score
- Volunteer Experience
- Sleep Hours
- Study Hours Per Day

### Target Variable

`placement_status`

Possible outputs:

- `Placed`
- `Not Placed`

## 🤖 Machine Learning Models

Three models were implemented:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

The best-performing model was automatically selected based on accuracy.

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Data Preprocessing
   ↓
Machine Learning Models
   ↓
Model Comparison
   ↓
Best Model Selection
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
New Student Prediction
   ↓
FastAPI Deployment
   ↓
Render