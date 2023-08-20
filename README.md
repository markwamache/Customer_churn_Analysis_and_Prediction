Analyzing Customer Churn: From Data Exploration to Predictive Modeling
Customer churn, also known as customer attrition, is a major concern for businesses across industries. Losing customers can lead to significant revenue loss and increased costs to acquire new customers. In this article, we'll embark on a journey to analyze customer churn data, explore insights, perform pre-processing and develop predictive models using machine learning.
Understanding Customer Churn
Customer churn occurs when customers stop using a company's product or service within a specified period. It's essential for businesses to understand why customers leave and when they leave to design effective retention strategies.
To understand well the customer churn analysis, I came up with the following Research questions.
1. What is the churn rate among customers? How many customers have churned (Yes) versus those who haven't (No)?
2. How does the distribution of churn ('Yes' and 'No') vary across different categories of the predictor variable?
3. What are the distributions and summaries of monthly charges, and total charges for the customer base, and are there any notable outliers in these distributions?
4. What is the average monthly charges and total charges for customers who churned and those who didn't?
5. What is the churn rate based on the tenure of customers with the company?
6. What is the distribution of monthly charges for customers who churned versus those who did not churn?
7. How does the distribution of total charges incurred by customers relate to churn rates?
8. Is there a correlation between MonthlyCharges, TotalCharges, and customer churn?
9. How do the various combinations of factors like tenure, monthly charges, total charges, payment method, contract term, etc. affect churn?

Data Preparation and Exploration
The first and foremost step is to do the necessary installations of the packages and modules required for our project. Its advisable to create a virtual environment for the the required installations.

After installations what follows is a redoing the necessary importations.

After the above steps I then went ahead and started loading the three datasets which are stored in different locations into python.
The first dataset (LP2_Telco_churn_first_3000) is stored in a database and I read it using the following codes.
# Load environment variables from .env file into a dictionary
environment_variables = dotenv_values('.env')

# Get the values for the credentials you set in the '.env' file
database = environment_variables.get("DATABASE")
server = environment_variables.get("SERVER")
username = environment_variables.get("USERNAME")
password = environment_variables.get("PASSWORD")

connection_string = f"DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
query = "Select * from dbo.LP2_Telco_churn_first_3000"
data = pd.read_sql(query, connection)


The second dataset, which is the test dataset is then loaded into pandas using the following code since it’s an excel file
excel_file = "C:/Users/OneDrive/Azubi1/LP2/Telco-churn-second-2000.xlsx"

df_test = pd.read_excel(excel_file, engine='openpyxl')

# Save the DataFrame as a CSV file
df_test.to_csv('df_test.csv', index=False)

I then loaded the last dataset which was hosted on a github repository. I downloaded it to my local machine then loaded it to python using the following code.
data3 = pd.read_csv("LP2_Telco-churn-last-2000.csv")

In these initial steps, I loaded the dataset, explored the first few rows, checked the shape, column names, data types, and obtained summary statistics. This gives us a solid foundation to proceed with our analysis.
Data Cleaning
Data cleaning was done separately on the three datasets. I started with cleaning the first dataset followed by the test dataset and lastly the third datasets.
Checking for duplicates
I checked for duplicates in the first dataset and dropped all those rows that were duplicated in the dataset.

data.duplicated().sum() # checking for duplicate columns 
# Next drop duplicated columns 
data.drop_duplicates(subset=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges', 'Churn'], inplace=True)

Handling Missing Data
Before diving into analysis, it's crucial to handle missing data. I used various techniques to manage missing values by first checking for the percentage of missing values.
data.isnull().sum() # checking for null values 
missing_values_percent = data.isna().mean().round(4) * 100
print(missing_values_percent)
We visualize the percentage of null values per column and generated the following figure  By imputing missing values, dropping duplicates, and filling categorical missing values, we ensure our data is ready for analysis.
columns_to_fill = ['MultipleLines',  'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Churn']

for column in columns_to_fill:
    data[column].fillna(data[column].mode()[0], inplace=True)
I then changed the various columns datatypes to match the other columns in the other dataset.
data['SeniorCitizen'] = data['SeniorCitizen'].astype(int)

cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']

# Convert bool to str in df2
for col in cols:
    data[col] = data[col].astype(str)

# Consistent representation
for col in cols:
    data[col] = data[col].map({'True': 'Yes', 'False': 'No'})

Checked for outliers in the Total Charges column
Q1 = data['TotalCharges'].quantile(0.25)
Q3 = data['TotalCharges'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Detect outliers
outliers = data[(data['TotalCharges'] < lower_bound) | (data['TotalCharges'] > upper_bound)]

print(f'There are {outliers.shape[0]} outliers in TotalCharges.')

 
I repeated the above steps to clean my test dataset and third dataset in which I converted the TotalCharges column to numeric using the following code 
# converting the totalcharges to numeric data type 
df_test['TotalCharges'] = pd.to_numeric(df_test['TotalCharges'],errors = 'coerce')
After the above steps we remove the columns that are not needed, i.e those are irrelevant to our objectives.
df_test.drop(columns= ['customerID','tenure'], axis=1, inplace=True)

After Cleaning Concatenate First Dataset and Third Dataset. Do not include the Test Dataset and save in csv format as df.csv
df = pd.concat([data, data3])
df.to_csv('df.csv', index=False)

After concatenating the datasets, I explored the first few rows, checked the shape, column names, data types, and obtained summary statistics
df.head()
df.shape()
df.describe()

I then generated the profile report for trainset and testset.
profile = ProfileReport(df, title='Train Dataset', html={'style':{'full_width':True}})
profile.to_notebook_iframe()
profile.to_file("[Trainset] Pandas-Profiling_Report.html")

Exploratory Data Analysis (EDA)
EDA involves visualizing data to uncover patterns, relationships, and insights. I based on the following questions to do visualizations and gain a deep insight of of my datasets.
1. What is the churn rate among customers? How many customers have churned (Yes) versus those who haven't (No)?
2. How does the distribution of churn ('Yes' and 'No') vary across different categories of the predictor variable?
3. What are the distributions and summaries of monthly charges, and total charges for the customer base, and are there any notable outliers in these distributions?
4. What is the average monthly charges and total charges for customers who churned and those who didn't?
5. What is the churn rate based on the tenure of customers with the company?
6. What is the distribution of monthly charges for customers who churned versus those who did not churn?
7. How does the distribution of total charges incurred by customers relate to churn rates?
8. Is there a correlation between MonthlyCharges, TotalCharges, and customer churn?
9. How do the various combinations of factors like tenure, monthly charges, total charges, payment method, contract term, etc. affect churn?
1. What is the churn rate among customers? How many customers have churned (Yes) versus those who haven't (No)?

 
2. How does the distribution of churn ('Yes' and 'No') vary across different categories of the predictor variable?
for i, predictor in enumerate(df.drop(columns=['Churn','PaymentMethod', 'TotalCharges', 'MonthlyCharges'])):
    plt.figure(i)
    sns.countplot(data=df, x=predictor, hue='Churn', palette=['skyblue', 'yellowgreen'])
    plt.title(f'Distribution of Churn by {predictor}')
    plt.show()

# Separate plot for 'PaymentMethod'
plt.figure(figsize=(8,6))
sns.countplot(data=df, y='PaymentMethod', hue='Churn', palette=['skyblue', 'yellowgreen'])
plt.title('Distribution of Churn by Payment Method')
plt.show()

      
3. What are the distributions and summaries of monthly charges, and total charges for the customer base, and are there any notable outliers in these distributions?

 
4. What is the average monthly charges and total charges for customers who churned and those who didn't?  
This suggests that customers who churn tend to be those with higher monthly charges but lower total charges, possibly indicating that these customers tend to leave the company relatively earlier in their tenure. This is a valuable insight for the company, as it might need to review its pricing strategy, particularly for new customers or customers in the early stages of their tenure.
5. What is the churn rate based on the tenure of customers with the company?  


The churn rate is calculated as the percentage of customers that have churned in each group. From the data, we can observe that the churn rate decreases as the tenure increases. For example, the churn rate for customers who stayed between 1 to 12 months is about 48%, while it's about 7% for customers who stayed between 61 to 72 months. This suggests that newer customers are more likely to churn compared to long-time customers.
6. What is the distribution of monthly charges for customers who churned versus those who did not churn?  
Churn is high when Monthly Charges are high
7. How does the distribution of total charges incurred by customers relate to churn rates?
 
Higher Churn at lower Total Charges. 
8. Is there a correlation between MonthlyCharges, TotalCharges, and customer churn?  
9. How do the various combinations of factors like tenure, monthly charges, total charges, payment method, contract term, etc. affect churn?  
The above process is followed by separating the features and target variable using the code 
x = df.drop('Churn', axis=1)
y = df['Churn']

  After separating we perform one hot encoding of the categorical variables, scale the data using Standard scaler and transform the columns using column Transformer. This can be well done by creating a pipeline which can be use to transform your dataset. We then resample the data using smote and then separate it into x_train and y_train which will be used in modelling.
Model Selection and Training
I decided to train eight models then choose the best performing models for hyper parameter tuning. I trained the following models
SVC(), GaussianNB(), DecisionTreeClassifier(), RandomForestClassifier(),
 XGBClassifier(), GradientBoostingClassifier(), AdaBoostClassifier(), LogisticRegression()
The model selection was done based on accuracy, precision recall trade-off and F1 score.
RandomForestClassifier: This model seems to have a good balance between precision and recall for both classes. It has a high F1-score for both classes as well.

After this I generated the confusion matrices for the models
models = [SVC(), GaussianNB(), DecisionTreeClassifier(), RandomForestClassifier(),
          XGBClassifier(), GradientBoostingClassifier(), AdaBoostClassifier(), LogisticRegression()]

for model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    print("Model: ", model.__class__.__name__)
    print("Classification Report: \n", classification_report(y_test, y_pred))
    
    fig = px.imshow(confusion_matrix(y_test, y_pred), color_continuous_scale='tropic',
                    title="Confusion Matrix of " + model.__class__.__name__,
                    labels=dict(x="Predicted", y="Actual", color="Counts"),
                    color_continuous_midpoint=0.8, width=400, height=400,
                    template="plotly_dark", text_auto=True)
    fig.show()
    print("--- -------------------------------------------------------")

 
This was then followed by visualizing the Area Under The curve AUC-ROC.
models =[RandomForestClassifier(), XGBClassifier(), GradientBoostingClassifier(),AdaBoostClassifier()]

for model in models:
    # Fit the model on your training data before making predictions
    model.fit(x_train, y_train)
    
    # ROC Curve:
    y_prob = model.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label="AUC = %0.2f" % roc_auc_score(y_test, y_prob))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.legend(loc='lower right')
    plt.title("ROC Curve of " + model.__class__.__name__, size=14)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

    
I decided to work with Random forest, XGB and Gradient boost classifier since they have the best AUC-ROC

Model Tuning
Model tuning is done to improve the performance of the model and also to determine the best parameter combinations which can produce the best performing model.

# Define models
gbc = GradientBoostingClassifier()
rfc = RandomForestClassifier()
xgb = XGBClassifier()

# Define hyperparameter grids for each model
gbc_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rfc_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

xgb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}
# Create lists of models and their corresponding parameter grids
models = [gbc, rfc, xgb]
params = [gbc_params, rfc_params, xgb_params]

 #Perform grid search for each model
for model, param_grid in zip(models, params):
    grid_cv = GridSearchCV(model, param_grid, scoring='f1', n_jobs=-1, verbose=2)
    grid_cv.fit(x_train, y_train)
    
    print(model.__class__.__name__, "Best params:", grid_cv.best_params_)


We tuned three models and Random Forest classifier was our best performing model which I decided to choose as our final model for predictions.
Model	Accuracy Score	F1 Score	ROC AUC Score	precision_score	recall_score
1	RandomForestClassifier	0.865359	0.866890	0.865359	0.857143	0.876861
2	XGBClassifier	0.853857	0.856383	0.853857	0.841830	0.871448
0	GradientBoostingClassifier	0.849120	0.850034	0.849120	0.844920	0.855210


I visualized the prediction results using the code below
#visualizing prediction results:

rfc = RandomForestClassifier(bootstrap=True, max_depth=20, max_features='sqrt', min_samples_leaf=1, min_samples_split=2, n_estimators=300)

rfc.fit(x_train, y_train)

y_pred = gbc.predict(x_test)
y_pred = y_pred.reshape(-1, 1)

# Convert y_test Series to a NumPy array
y_test_array = np.array(y_test).reshape(-1, 1)

sns.displot(x=y_test_array.flatten(), y=y_pred.flatten(), kind="kde", fill=True)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()

 
Testing predictions
Additionally, we tested predictions on new unseen data
The predictions are tested to determine the final performance of our best model 

# below we are doing OneHotEncoding of the categorical variables 
categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']
# below is defining the columns containing the numerical features to be scaled 
numerical_features = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges']

categorical_transformer = OneHotEncoder(drop='first')
numerical_transformer = StandardScaler()
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ]
)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    
])
x1_test_transformed = pipeline.fit_transform(x1_test)


y1_pred = rfc.predict(x1_test_transformed)
unique_values, counts = np.unique(y1_pred, return_counts=True)

# Map label values to descriptive strings
label_mapping = {0: "Not Churned", 1: "Churn"}

churn_counts = {label_mapping[val]: count for val, count in zip(unique_values, counts)}
print(churn_counts)

After testing the predictions you now save your model key components using pickle and export it to a local directory.

components={
'imputer': imputer,
'encoder' : categorical_transformer,
'smote' : smote,
'scaler': numerical_transformer,
'models' : models
}
destination =os.path.join(".","Export2")

with open(os.path.join(destination,"ml.pkl"),"wb") as f:
    pickle.dump(components, f)

Conclusion. 
Random Forest was our best performing model model which can be used for future predictions.
