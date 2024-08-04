Data Science Hackathon
Hackathon Topic :– Ecommerce Product Categorization


Ecommerce Product Categorization
This project focuses on predicting the category of products typically available on e-commerce platforms such as Flipkart. Machine Learning and Deep Learning models have been developed and trained using a comprehensive Flipkart e-commerce dataset, which is accessible for analysis and application.


Codebase Structure
1. **Notebooks**: Contains all Jupyter Notebooks used for Exploratory Data Analysis (EDA), model training, and testing. Each notebook is organized to demonstrate the step-by-step development and evaluation of both Machine Learning and Deep Learning models.
2. **requirements.txt**: Lists all the dependencies needed to recreate the project's development   environment. Ensure to install these packages to run the notebooks and scripts without issues.
3. **Dataset**: Includes the primary datasets used for the project:
     - `train_product_data.csv`: The training dataset.
     - `test_data.csv`: The test dataset.
4. **Dataset2**: Contains additional dataset versions for various sampling techniques:
      - `unbalanced_dataset.csv`: The original, unbalanced dataset
      - `undersampling_data.csv`: The dataset after applying undersampling techniques.
      - `oversampling_dataset.csv`: The dataset after applying oversampling techniques.
5. **Report**: Includes a comprehensive report summarizing all observations, methodologies, results, and conclusions derived from the project. This document provides insights into the analysis and model performance.


### Methodlogy

For the Multiclass Classification of e-commerce products based on their descriptions, the following steps were undertaken:

1. **Data Visualization and Cleaning**: The dataset and its hidden parameters were visualized using libraries like Seaborn, Matplotlib, and Yellowbrick. This process aided in data cleaning, where non-contributory words identified through Word Cloud analysis were removed from the corpus to enhance product classification.

2. **Category Selection**: It was determined to use only the root of the Product Category Tree as the primary label for classification, simplifying the categorization process.

3. **Data Preparation**: Data cleaning, preprocessing, and resampling techniques were employed to balance the dataset, ensuring a fair representation of all categories.

4. **Category Refinement**: Detailed analysis and visualization led to the decision to categorize products into 13 primary categories. Miscellaneous categories with fewer than 10 products were excluded to reduce noise. The categories are:
   - Clothing
   - Jewellery
   - Sports & Fitness
   - Electronics
   - Babycare
   - Home Furnishing & Kitchen
   - Personal Accessories
   - Automotive
   - Pet Supplies
   - Tools & Hardware
   - Ebooks
   - Toys & School Supplies
   - Footwear

5. **Machine Learning Implementation**: Various Machine Learning algorithms were applied using the scikit-learn library. These included:
   - Logistic Regression (Binary and Multiclass variants)
   - Linear Support Vector Machine
   - Multinomial Naive Bayes
   - Decision Tree
   - Random Forest Classifier
   - K-Nearest Neighbors

6. **Deep Learning Implementation**: Despite achieving good accuracy with ML models, Deep Learning models were also implemented using the PyTorch framework to explore further improvements. The models included:
   - Transformer-based models:
     - Bidirectional Encoder Representations from Transformers (BERT)
     - RoBERTa
     - DistilBERT
     - XLNet
   - Recurrent Neural Network-based models:
     - Long Short-Term Memory (LSTM) networks

Each of these steps was meticulously documented and evaluated to ensure the robustness and accuracy of the final classification model.



### Methodology

#### STEP 1: Exploratory Data Analysis and Data Preprocessing
- **Exploratory Data Analysis (EDA)**:
  - An in-depth analysis of the dataset was conducted using Word Clouds, Bar Graphs, and t-SNE Visualizations. This helped identify the most frequent unigrams in the product descriptions, understand the distribution of products and brands across different categories, and analyze the length of the descriptions.
  - EDA provided insights into the data, revealing patterns and trends crucial for guiding the subsequent data preprocessing steps.

- **Data Cleaning**:
  - Contraction mapping and the removal of custom stopwords and URLs were performed.
  - Tokenization and Lemmatization were applied to normalize the text data.
  - These steps ensured that the product descriptions were clean, consistent, and ready for model training.

- **Data Balancing**:
  - Given the clear imbalance in the dataset, both oversampling and undersampling techniques were applied.
  - Balanced datasets were saved as CSV files for further use in training models.

#### STEP 2: Machine Learning Models for Product Categorization
- **Model Training**:
  - Six Machine Learning algorithms were applied to the imbalanced, oversampled, and undersampled datasets:
    - Logistic Regression (Binary and Multiclass variants)
    - Linear Support Vector Machine (SVM)
    - Multinomial Naive Bayes
    - Decision Trees
    - Random Forest Classifier
    - K-Nearest Neighbors (KNN)

- **Model Evaluation**:
  - Evaluation metrics included Classification Reports, Confusion Matrices, Accuracy Scores, ROC Curves, and AUC Scores.
  - The table below summarizes the validation accuracy of the ML algorithms on different datasets:

| ML Algorithm                  | Imbalanced Dataset | Balanced Dataset (Oversampling) | Balanced Dataset (Undersampling) |
|-------------------------------|--------------------|---------------------------------|----------------------------------|
| Logistic Regression (Binary)  | 0.9654             | 0.9756                          | 0.9486                           |
| Logistic Regression (Multiclass) | 0.9735          | 0.9893                          | 0.9654                           |
| Naive Bayes                   | 0.9096             | 0.9602                          | 0.9054                           |
| Linear SVM                    | 0.9799             | 0.9958                          | 0.9749                           |
| Decision Trees                | 0.7017             | 0.6883                          | 0.7561                           |
| Random Forest Classifier      | 0.9209             | 0.9367                          | 0.9235                           |
| K-Nearest Neighbors           | 0.9564             | 0.9800                          | 0.9453                           |

  - The Linear Support Vector Machine algorithm outperformed others across all datasets, achieving the highest accuracy.

#### STEP 3: Deep Learning Models for Product Categorization
- **Model Training and Evaluation**:
  - Deep Learning models were trained and evaluated on the undersampled balanced dataset.
  - Various Transformer-based models and Recurrent Neural Network (RNN) based models were considered, including:
    - Bidirectional Encoder Representations from Transformers (BERT)
    - RoBERTa
    - DistilBERT
    - XLNet
    - Long Short-Term Memory (LSTM) networks

- **Best Performing Model**:
  - After a detailed evaluation, BERT (uncased, base, with all layers frozen except the last one) emerged as the best performer, achieving an f1-score of 0.98.

This comprehensive approach ensured a robust analysis and classification of e-commerce products, leveraging both traditional Machine Learning and state-of-the-art Deep Learning techniques. Each step was meticulously documented to provide a clear understanding of the process and the rationale behind the decisions made.



Future Work
•	Advanced Feature Extraction:
o	Perform feature extraction on the Product Category Tree column to identify more granular classes for product categorization. This will enable a finer classification, helping to pinpoint specific subcategories within broader categories.
•	Enhanced Data Balancing Techniques:
o	Implement advanced data balancing techniques such as Synthetic Minority Over-sampling Technique (SMOTE) to address class imbalance more effectively. This could improve the performance and generalizability of the models.
•	Comprehensive Deep Learning Evaluation:
o	Train and evaluate Deep Learning models on datasets other than the undersampled one. This involves experimenting with balanced datasets created through various techniques, including oversampling and SMOTE.
o	Test these models on diverse e-commerce datasets available online to assess their scalability and robustness in handling real-world data variations. This will provide insights into the models' performance across different e-commerce platforms and product categories.





numpy
simpletransformers
transformers
tokenizers
torchtext
torchvision
sklearn
matplotlib
regex
scikit-learn
wordcloud
Pillow
nltk
Keras
Keras-Applications
Keras-Preprocessing
pandas
pytorch-nlp
