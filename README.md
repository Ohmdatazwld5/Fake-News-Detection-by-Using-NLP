# Fake-News-Detection-by-Using-NLP
- Importing necessary libraries:

->pandas for data manipulation and analysis.

->re for regular expression operations.

->nltk for natural language processing tasks.

->nltk.download('stopwords') to download the stopwords corpus from NLTK

- Loading the dataset:

->The code reads a CSV file named "train.csv" using the pd.read_csv() function and assigns it to the DataFrame df.

->df.head(5) displays the first 5 rows of the DataFrame.

->df.columns displays the column names of the DataFrame.

- Data preprocessing:

->df.drop(['id', 'title', 'author'], axis=1) drops the 'id', 'title', and 'author' columns from the DataFrame.

->Importing stopwords from NLTK and creating an instance of the PorterStemmer class.

- Stemming function:

-> The function stemming() performs text preprocessing by applying stemming, removing non-alphabetic characters, converting text to lowercase, removing stopwords, and joining the processed words back into a string.

- Applying stemming to the 'text' column:

->The 'text' column of the DataFrame is transformed by applying the stemming() function using df['text'].apply(stemming).

- Splitting the data into training and testing sets:

->The train_test_split() function from the sklearn.model_selection module is used to split the preprocessed text data (x) and the corresponding labels (y) into training and testing sets. The testing set size is set to 20% of the total data.

- Creating TF-IDF vectors:

->The TfidfVectorizer() from sklearn.feature_extraction.text is used to convert the text data into TF-IDF vectors.

->The fit_transform() method is applied to the training set (x_train) to learn the vocabulary and transform the text into feature vectors.

->The transform() method is applied to the testing set (x_test) to transform the text into feature vectors using the vocabulary learned from the training set

- Training a decision tree classifier:

->An instance of DecisionTreeClassifier() from sklearn.tree is created.

->The fit() method is used to train the classifier on the training data (x_train and y_train).

- Making predictions:

->The trained model is used to predict the labels for the testing set using the predict() method.

->The predicted labels are stored in the variable prediction.

- Model evaluation:

->The accuracy score of the model is calculated using the score() method, which compares the predicted labels (prediction) with the actual labels (y_test).

- Saving the model and vectorizer:

->The trained vectorizer and model are saved using the pickle.dump() function, so they can be used later for making predictions.

- Loading the saved model and vectorizer:

->The saved vectorizer and model are loaded using the pickle.load() function and assigned to the variables vector_form and load_model, respectively.

- Fake news detection function:

->The function fake_news(news) takes a news article as input.

->It preprocesses the input article using the stemming() function and transforms it into a feature vector using the loaded vectorizer (vector_form).

->The loaded model (load_model) is then used to predict the label for the input feature vector.
