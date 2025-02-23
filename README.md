# GenAi Finals - HR Screening System

Welcome to the **GenAi Finals - HR Screening System** repository! This project contains the running code for an intelligent HR screening system that helps match job seekers' resumes to a desired goal or job description. Users can input their desired job role or goal and upload resumes in PDF format. The system then analyzes the resumes and calculates how closely they align with the input goal, providing a score to indicate the match.

## 📋 Features

- **Goal Input**: Users can input their desired job goal (e.g., "Data Scientist", "Software Engineer") or job description.
- **Resume Upload**: Users can upload PDF resumes for analysis.
- **Match Scoring**: The system calculates a score based on how closely the resumes match the specified goal or job description.
- **PDF Parsing**: Resumes are parsed from PDF format and analyzed to extract relevant information.
- **User-Friendly Interface**: Simple interaction flow for input and result display.

## ⚙️ Technologies Used

The system uses the following technologies:

- **Python** (for core functionality)
- **Natural Language Processing (NLP)** (for analyzing text from resumes and goals)
- **PDFMiner** (for extracting text from PDF resumes)
- **Flask** (for building the web interface)
- **Scikit-learn** (for machine learning and scoring algorithms)
- **TF-IDF** (for text vectorization and comparison)

## 🧑‍💻 How to Use

### Prerequisites

Before running the system, ensure you have the following installed:

- Python 3.x
- Required libraries (listed in `requirements.txt`)

### Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/chrisjallaine/GenAi-Finals.git
