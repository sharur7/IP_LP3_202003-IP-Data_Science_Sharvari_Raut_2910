{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# To build a model to accurately classify a piece of news as REAL or FAKE.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Using sklearn,  build a TfidfVectorizer on the provided dataset. Then, initialize a PassiveAggressive Classifier and fit the model. In the end, the accuracy score and the confusion matrix tell us how well our model fares."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 121,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Importing Libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.preprocessing import Normalizer\n",
    "from sklearn.linear_model import PassiveAggressiveClassifier\n",
    "from sklearn import svm, datasets\n",
    "from sklearn.metrics import plot_confusion_matrix\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "import sklearn.metrics as metrics\n",
    "from itertools import product\n",
    "import itertools"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Importing DataSet"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Reading Dataset\n",
    "\n",
    "df=pd.read_csv('news.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(6335, 4)"
      ]
     },
     "execution_count": 123,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Shape of the dataset\n",
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>title</th>\n",
       "      <th>text</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8476</td>\n",
       "      <td>You Can Smell Hillary’s Fear</td>\n",
       "      <td>Daniel Greenfield, a Shillman Journalism Fello...</td>\n",
       "      <td>FAKE</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>10294</td>\n",
       "      <td>Watch The Exact Moment Paul Ryan Committed Pol...</td>\n",
       "      <td>Google Pinterest Digg Linkedin Reddit Stumbleu...</td>\n",
       "      <td>FAKE</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3608</td>\n",
       "      <td>Kerry to go to Paris in gesture of sympathy</td>\n",
       "      <td>U.S. Secretary of State John F. Kerry said Mon...</td>\n",
       "      <td>REAL</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>10142</td>\n",
       "      <td>Bernie supporters on Twitter erupt in anger ag...</td>\n",
       "      <td>— Kaydee King (@KaydeeKing) November 9, 2016 T...</td>\n",
       "      <td>FAKE</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>875</td>\n",
       "      <td>The Battle of New York: Why This Primary Matters</td>\n",
       "      <td>It's primary day in New York and front-runners...</td>\n",
       "      <td>REAL</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0                                              title  \\\n",
       "0        8476                       You Can Smell Hillary’s Fear   \n",
       "1       10294  Watch The Exact Moment Paul Ryan Committed Pol...   \n",
       "2        3608        Kerry to go to Paris in gesture of sympathy   \n",
       "3       10142  Bernie supporters on Twitter erupt in anger ag...   \n",
       "4         875   The Battle of New York: Why This Primary Matters   \n",
       "\n",
       "                                                text label  \n",
       "0  Daniel Greenfield, a Shillman Journalism Fello...  FAKE  \n",
       "1  Google Pinterest Digg Linkedin Reddit Stumbleu...  FAKE  \n",
       "2  U.S. Secretary of State John F. Kerry said Mon...  REAL  \n",
       "3  — Kaydee King (@KaydeeKing) November 9, 2016 T...  FAKE  \n",
       "4  It's primary day in New York and front-runners...  REAL  "
      ]
     },
     "execution_count": 124,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Printing first few lines\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>title</th>\n",
       "      <th>text</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8476</td>\n",
       "      <td>You Can Smell Hillary’s Fear</td>\n",
       "      <td>Daniel Greenfield, a Shillman Journalism Fello...</td>\n",
       "      <td>FAKE</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>10294</td>\n",
       "      <td>Watch The Exact Moment Paul Ryan Committed Pol...</td>\n",
       "      <td>Google Pinterest Digg Linkedin Reddit Stumbleu...</td>\n",
       "      <td>FAKE</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3608</td>\n",
       "      <td>Kerry to go to Paris in gesture of sympathy</td>\n",
       "      <td>U.S. Secretary of State John F. Kerry said Mon...</td>\n",
       "      <td>REAL</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>10142</td>\n",
       "      <td>Bernie supporters on Twitter erupt in anger ag...</td>\n",
       "      <td>— Kaydee King (@KaydeeKing) November 9, 2016 T...</td>\n",
       "      <td>FAKE</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>875</td>\n",
       "      <td>The Battle of New York: Why This Primary Matters</td>\n",
       "      <td>It's primary day in New York and front-runners...</td>\n",
       "      <td>REAL</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0                                              title  \\\n",
       "0        8476                       You Can Smell Hillary’s Fear   \n",
       "1       10294  Watch The Exact Moment Paul Ryan Committed Pol...   \n",
       "2        3608        Kerry to go to Paris in gesture of sympathy   \n",
       "3       10142  Bernie supporters on Twitter erupt in anger ag...   \n",
       "4         875   The Battle of New York: Why This Primary Matters   \n",
       "\n",
       "                                                text label  \n",
       "0  Daniel Greenfield, a Shillman Journalism Fello...  FAKE  \n",
       "1  Google Pinterest Digg Linkedin Reddit Stumbleu...  FAKE  \n",
       "2  U.S. Secretary of State John F. Kerry said Mon...  REAL  \n",
       "3  — Kaydee King (@KaydeeKing) November 9, 2016 T...  FAKE  \n",
       "4  It's primary day in New York and front-runners...  REAL  "
      ]
     },
     "execution_count": 125,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Set Index\n",
    "df.set_index('Unnamed: 0')\n",
    "\n",
    "#Printing few lines to check index\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Extracting the training data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Setting y label\n",
    "y = df.label"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>title</th>\n",
       "      <th>text</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8476</td>\n",
       "      <td>You Can Smell Hillary’s Fear</td>\n",
       "      <td>Daniel Greenfield, a Shillman Journalism Fello...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>10294</td>\n",
       "      <td>Watch The Exact Moment Paul Ryan Committed Pol...</td>\n",
       "      <td>Google Pinterest Digg Linkedin Reddit Stumbleu...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3608</td>\n",
       "      <td>Kerry to go to Paris in gesture of sympathy</td>\n",
       "      <td>U.S. Secretary of State John F. Kerry said Mon...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>10142</td>\n",
       "      <td>Bernie supporters on Twitter erupt in anger ag...</td>\n",
       "      <td>— Kaydee King (@KaydeeKing) November 9, 2016 T...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>875</td>\n",
       "      <td>The Battle of New York: Why This Primary Matters</td>\n",
       "      <td>It's primary day in New York and front-runners...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6330</th>\n",
       "      <td>4490</td>\n",
       "      <td>State Department says it can't find emails fro...</td>\n",
       "      <td>The State Department told the Republican Natio...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6331</th>\n",
       "      <td>8062</td>\n",
       "      <td>The ‘P’ in PBS Should Stand for ‘Plutocratic’ ...</td>\n",
       "      <td>The ‘P’ in PBS Should Stand for ‘Plutocratic’ ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6332</th>\n",
       "      <td>8622</td>\n",
       "      <td>Anti-Trump Protesters Are Tools of the Oligarc...</td>\n",
       "      <td>Anti-Trump Protesters Are Tools of the Oligar...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6333</th>\n",
       "      <td>4021</td>\n",
       "      <td>In Ethiopia, Obama seeks progress on peace, se...</td>\n",
       "      <td>ADDIS ABABA, Ethiopia —President Obama convene...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6334</th>\n",
       "      <td>4330</td>\n",
       "      <td>Jeb Bush Is Suddenly Attacking Trump. Here's W...</td>\n",
       "      <td>Jeb Bush Is Suddenly Attacking Trump. Here's W...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>6335 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      Unnamed: 0                                              title  \\\n",
       "0           8476                       You Can Smell Hillary’s Fear   \n",
       "1          10294  Watch The Exact Moment Paul Ryan Committed Pol...   \n",
       "2           3608        Kerry to go to Paris in gesture of sympathy   \n",
       "3          10142  Bernie supporters on Twitter erupt in anger ag...   \n",
       "4            875   The Battle of New York: Why This Primary Matters   \n",
       "...          ...                                                ...   \n",
       "6330        4490  State Department says it can't find emails fro...   \n",
       "6331        8062  The ‘P’ in PBS Should Stand for ‘Plutocratic’ ...   \n",
       "6332        8622  Anti-Trump Protesters Are Tools of the Oligarc...   \n",
       "6333        4021  In Ethiopia, Obama seeks progress on peace, se...   \n",
       "6334        4330  Jeb Bush Is Suddenly Attacking Trump. Here's W...   \n",
       "\n",
       "                                                   text  \n",
       "0     Daniel Greenfield, a Shillman Journalism Fello...  \n",
       "1     Google Pinterest Digg Linkedin Reddit Stumbleu...  \n",
       "2     U.S. Secretary of State John F. Kerry said Mon...  \n",
       "3     — Kaydee King (@KaydeeKing) November 9, 2016 T...  \n",
       "4     It's primary day in New York and front-runners...  \n",
       "...                                                 ...  \n",
       "6330  The State Department told the Republican Natio...  \n",
       "6331  The ‘P’ in PBS Should Stand for ‘Plutocratic’ ...  \n",
       "6332   Anti-Trump Protesters Are Tools of the Oligar...  \n",
       "6333  ADDIS ABABA, Ethiopia —President Obama convene...  \n",
       "6334  Jeb Bush Is Suddenly Attacking Trump. Here's W...  \n",
       "\n",
       "[6335 rows x 3 columns]"
      ]
     },
     "execution_count": 127,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Dropping the label column\n",
    "df.drop(\"label\", axis =1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Making Training and Test Sets\n",
    "X_train, X_test, y_train, y_test = train_test_split (df['text'], y, test_size=0.33, random_state=53)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Building Vectorizer Classifiers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize the `count_vectorizer` \n",
    "count_vectorizer = CountVectorizer(stop_words='english')\n",
    "\n",
    "# Fit and transform the training data \n",
    "count_train = count_vectorizer.fit_transform(X_train) \n",
    "\n",
    "# Transform the test set \n",
    "count_test = count_vectorizer.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize the `tfidf_vectorizer` \n",
    "tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7) \n",
    "\n",
    "# Fit and transform the training data \n",
    "tfidf_train = tfidf_vectorizer.fit_transform(X_train) \n",
    "\n",
    "# Transform the test set \n",
    "tfidf_test = tfidf_vectorizer.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['حلب', 'عربي', 'عن', 'لم', 'ما', 'محاولات', 'من', 'هذا', 'والمرضى', 'ยงade']\n",
      "['00', '000', '0000', '00000031', '000035', '00006', '0001', '0001pt', '000ft', '000km']\n"
     ]
    }
   ],
   "source": [
    "# Get the feature names of `tfidf_vectorizer` \n",
    "print(tfidf_vectorizer.get_feature_names()[-10:])\n",
    "\n",
    "# Get the feature names of `count_vectorizer` \n",
    "print(count_vectorizer.get_feature_names()[:10])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Intermezzo: Count versus TF-IDF Features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "metadata": {},
   "outputs": [],
   "source": [
    "count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "metadata": {},
   "outputs": [],
   "source": [
    "tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 134,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "set()"
      ]
     },
     "execution_count": 134,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "difference = set(count_df.columns) - set(tfidf_df.columns)\n",
    "difference"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 135,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "False\n"
     ]
    }
   ],
   "source": [
    "print(count_df.equals(tfidf_df))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>00</th>\n",
       "      <th>000</th>\n",
       "      <th>0000</th>\n",
       "      <th>00000031</th>\n",
       "      <th>000035</th>\n",
       "      <th>00006</th>\n",
       "      <th>0001</th>\n",
       "      <th>0001pt</th>\n",
       "      <th>000ft</th>\n",
       "      <th>000km</th>\n",
       "      <th>...</th>\n",
       "      <th>حلب</th>\n",
       "      <th>عربي</th>\n",
       "      <th>عن</th>\n",
       "      <th>لم</th>\n",
       "      <th>ما</th>\n",
       "      <th>محاولات</th>\n",
       "      <th>من</th>\n",
       "      <th>هذا</th>\n",
       "      <th>والمرضى</th>\n",
       "      <th>ยงade</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 56922 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   00  000  0000  00000031  000035  00006  0001  0001pt  000ft  000km  ...  \\\n",
       "0   0    0     0         0       0      0     0       0      0      0  ...   \n",
       "1   0    0     0         0       0      0     0       0      0      0  ...   \n",
       "2   0    0     0         0       0      0     0       0      0      0  ...   \n",
       "3   0    0     0         0       0      0     0       0      0      0  ...   \n",
       "4   0    0     0         0       0      0     0       0      0      0  ...   \n",
       "\n",
       "   حلب  عربي  عن  لم  ما  محاولات  من  هذا  والمرضى  ยงade  \n",
       "0    0     0   0   0   0        0   0    0        0      0  \n",
       "1    0     0   0   0   0        0   0    0        0      0  \n",
       "2    0     0   0   0   0        0   0    0        0      0  \n",
       "3    0     0   0   0   0        0   0    0        0      0  \n",
       "4    0     0   0   0   0        0   0    0        0      0  \n",
       "\n",
       "[5 rows x 56922 columns]"
      ]
     },
     "execution_count": 136,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "count_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 137,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>00</th>\n",
       "      <th>000</th>\n",
       "      <th>0000</th>\n",
       "      <th>00000031</th>\n",
       "      <th>000035</th>\n",
       "      <th>00006</th>\n",
       "      <th>0001</th>\n",
       "      <th>0001pt</th>\n",
       "      <th>000ft</th>\n",
       "      <th>000km</th>\n",
       "      <th>...</th>\n",
       "      <th>حلب</th>\n",
       "      <th>عربي</th>\n",
       "      <th>عن</th>\n",
       "      <th>لم</th>\n",
       "      <th>ما</th>\n",
       "      <th>محاولات</th>\n",
       "      <th>من</th>\n",
       "      <th>هذا</th>\n",
       "      <th>والمرضى</th>\n",
       "      <th>ยงade</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 56922 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "    00  000  0000  00000031  000035  00006  0001  0001pt  000ft  000km  ...  \\\n",
       "0  0.0  0.0   0.0       0.0     0.0    0.0   0.0     0.0    0.0    0.0  ...   \n",
       "1  0.0  0.0   0.0       0.0     0.0    0.0   0.0     0.0    0.0    0.0  ...   \n",
       "2  0.0  0.0   0.0       0.0     0.0    0.0   0.0     0.0    0.0    0.0  ...   \n",
       "3  0.0  0.0   0.0       0.0     0.0    0.0   0.0     0.0    0.0    0.0  ...   \n",
       "4  0.0  0.0   0.0       0.0     0.0    0.0   0.0     0.0    0.0    0.0  ...   \n",
       "\n",
       "   حلب  عربي   عن   لم   ما  محاولات   من  هذا  والمرضى  ยงade  \n",
       "0  0.0   0.0  0.0  0.0  0.0      0.0  0.0  0.0      0.0    0.0  \n",
       "1  0.0   0.0  0.0  0.0  0.0      0.0  0.0  0.0      0.0    0.0  \n",
       "2  0.0   0.0  0.0  0.0  0.0      0.0  0.0  0.0      0.0    0.0  \n",
       "3  0.0   0.0  0.0  0.0  0.0      0.0  0.0  0.0      0.0    0.0  \n",
       "4  0.0   0.0  0.0  0.0  0.0      0.0  0.0  0.0      0.0    0.0  \n",
       "\n",
       "[5 rows x 56922 columns]"
      ]
     },
     "execution_count": 137,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tfidf_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Comparing Models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 138,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Normalized confusion matrix\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Text(0.5, 15.0, 'Predicted label')"
      ]
     },
     "execution_count": 138,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAagAAAEmCAYAAAA3CARoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjMsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+AADFEAAAgAElEQVR4nO3de7xVdZ3/8dcbEBIQb6ChgICASUogF8lKSy0vM2E5lqhNWqZpWlmpY9lkWj7Umi7j6PzUyuuM4D0pTZq8oPkQBUVURLkrBzRMG00dLkc+vz++3wOLzTmcjcI+ax/ez8djP1hr7e9a6/vdh7Pe+/td66yliMDMzKxsOrR1BczMzJrjgDIzs1JyQJmZWSk5oMzMrJQcUGZmVkoOKDMzK6WqAkrSoZKelzRP0jnNvN9F0k35/Ucl9c/Lj5P0ZOG1WtJwSV0l3SXpOUmzJF1c2NZuku6V9JSkByT1qdhXD0lLJF1WWPZArl/TfnbKy0+Q9Eph+Vfy8uGSHsn7fkrS0YVtnZ7bEZJ6Vuz743k7syRNKSz/pqRn8vIzCsuHS5qa15kuaUxevr2kO/K+H5O0V2Gdb+XtPCNpgqT3VdThPyS9WZg/RdLTeR9/ljS09Z+oWd07FHgemAesd0wCdgPuBZ4CHgCajiPDgUeAWfm9owvrPAQ8mV9Lgd/m5QIuzft6CtinsM47hXUmNVOP/wDebGb5UUAAo/L8cYXtPAmsznUFOAZ4Ou/7HqDpuPSjvOxJ4I/ALlXU9x7gf4HfV9RnAPAoMBe4Ceicl+8PPAE05joXtdT2lj7HjRcRG3wBHYH5wMBc6ZnA0IoyXwOuyNPjgZua2c7ewII83RX4RJ7unBt0WJ6/BTg+Tx8I3FCxnX8HbgQuKyx7ABjVzD5PKJYrLB8CDM7TuwAvAdvl+RFAf2AR0LOwznbAs0C/PL9T/ncv4Jncpk7Anwrb/mOhXYcDD+TpnwLn5ekPAPfm6V2BhcDWef5m4IRCHUYBNwBvFpb1KEyPA+5p7Wfql191/uoYEfMjYmBEdI6ImRExtKLMLRFxfJ4+MCKajiNDImJwnt4lIl6KiO2a2cdtEfHFPH14RPwhIhQRYyPi0UK5N5tZt+k1Ku+3ssw2EfFgREzNZSrX2zsiFuTpThGxLCKajkU/iYgf5ukehXW+ERFXVFHfgyLi0xHx+4p93hwR4/P0FRFxap7uHxHDIuL6iDiqYp0Ntb25z3GjX9X0oMYA8/IHthKYCBxRUeYI4Lo8fStwkCRVlDkGmAAQEW9HxP15eiUpoZu+4QwlffMBuL+4L0kjgZ3zgf9di4g5ETE3Ty8FlgG98vyMiFjUzGrHArdHxIu53LK8fE9gam5TIzAF+GzTroAeeXpb0reJddoYEc8B/SXtnN/rBGwtqRMp9JbmtnckBdvZFW15ozDbLe/TrD0bQ+odLABaOia1dByZQ+olQPrdWvO7X7AN6ctx0zf/I4DrSb9bU0lfVnu3Usdmf1+zHwE/AZa3sO6aYyWpNyTS77ZIx5Om40hLv/sbqu+9wN8r9idSe2/N89cBn8nTi0i9sNUt1HVDKj/HjVZNQO0KLC7MN+RlzZbJB+nXgR0ryhzN2g99DUnbAZ9m7X+mmcA/5enPAttI2lFSB+BnwFkt1POaPMz1rxXh+E95KO1WSX2b2f8YUi9ufgvbbTIE2D4PJz4u6Yt5+TPA/rmOXUk9pab9nAH8VNJi4N+A7xbaeGRh/7sBfSJiSS73IqlX93pENIXx6cCkiHipmTacJmk+6T/9N1pph1m9q+aYtN5xhPWPSS397n+WdDxqCoAN7e99wHRSEHymUOZ00rBX5e/rCNLxoXKIrah4rFwFnEoa4ltKCt7fFMpemOt2HPCDKurbnB1Jw36NVZZv0lLbm1R+jhutmoCq7AnB+t/SN1hG0r7A2xHxzDorpV7CBODSiFiQF58JHCBpBnAAsAT4JSn0PhARxQ++yXGkb0nbkL6xfD8v/x3QPyKGkYberiuuJKk3acjsSxHR2jeETsBI4B+AQ4B/lTQkImYDlwD/QxrfncnaH/SpwLcioi/wLdb+x7qYFHZPAl8HZgCNkrYnffsZQBp67CbpC5J2AT5HGs9eT0RcHhG7A/9SaLtZe1XNMelM0vGjeBxpLLy/5nef9XsHxR5Ma/vrRxp6P5Z0nNqd9Lvb3O9rB+AXwHea2V6TfYG3SV98AbYiHUdG5O0+xdovugDnkgLvv0mh2Fp9m7Ox5Zs01/aiys9xo1UTUA2s7RFAGopb2lKZHDrbAq8V3h9P8xW9CpgbEb9sWhARSyPiyIgYQfrwAX5FOk+1g6RFpF7GFwsXV3wIGAwMIn1Qp+VtvRoRKwrbGNm0H0k9gLuA70fE1FY+g6Y23hMRb0XEX4EH836JiN9ExD4RsX9ud9MQwvHA7Xn6FtI3NiLijYj4UkQMB75IGmJYCBwMLIyIVyJiVV53P9J/zkHAvNz+rpLmNVPHiTT/TcasPanmmLSUNEpRPI68nv9d87tP+vZftCPp9/SuKvfX9O8C0rnwERR+X0lDZF3z9Dakc9YP5OVjSb2spgslYP1jZdOFEvNJoXEz6ZhQ6UbW9hir+XyK/koaBuxUZfkmzbW9SXOf40arJqCmAYMlDZDUmfQBVl6tMol0MIZ0pcd9EREAeWjuc6SD5xqSfkwKsjMqlvfM60D6pnB1RDxIuhBjTkT0J307uj4izsmBeDRpzLUT6ZwQknrnHlKTccDs/F5n4I68jVuq+AwA7gQ+JqlTHsrbt7C9pqsG+5F+KZr+gy0lfXuDNBY7N5fbLtcB4CvAg/lc0ovA2HyVo4CDgNkRcVdEvD+fsOxP6o0OytsaXKjjP7A2HM3aq2mkL6QDSEN0zR2TerL2+PZd4Oo8veZ3n/SlsdLnSMNvxfNDk0hfJEUKlddJQ3fbA10K+/sI6UKqu4D3ky626k/qEQ3K6/UsLJ9KOi5Nz9to7li5hDSs13Se7JPk407+DJqMA55rpb4tCdIIVNNVeseTjncb0lLbmzT3OW405RzZcCHpcFLPpCMpMC6UdAEwPSIm5UuhbyAl6GvA+KYhO0kfBy6OiLGF7fUhjZE+BzT1cC6LiF9LOgq4iPShPQicFhErlC5d/31E7CXpBNJVe6dL6kb6If41b+tPpG8p/0L6RjGO1LV/jdRV3p90WWr/Dh06qEuX9Bn379+frl27smzZMl5++WVWrVrFVlttRY8ePejfvz8AL7/8Mq+++ioAPXv2ZOed03UNzz//PI2NjUiiT58+9OiRrot48803Wbx4MRGBJPr160e3bt148803WbRoEQBbb701u+22G506pS8vS5cu5bXXXkMSXbt2ZbfddqNDh3W/R8yYMYMRI9KXlcWLF/PGG28giY4dO9KvXz+23nrrVn+mZvXsIx/5CN/+9rfp2LEjkyZN4uqrr+arX/0qs2fP5sEHH+Sggw7itNNOIyKYMWMGl1xyCatWreKwww7jvPPOY/78taedzj//fObMmQPAlVdeybXXXssjjzyyzv7OPvts9ttvP5YvX87555/P7NmzGTZsGN/73vdYvXo1HTp0YMKECdx55/rH9QcffJD9999/veVXXnklv/zlL5k9O+XNyJEj+cpXvtI4evTorSqKngJ8k3Q+6gXS1cmvArcBe5CGKF/I5ZaQguky0qX4b5OGMZtC8CHSlcPd8zZOBCaTrtKeCOxAGhb9Aul4OpoU6NuTwuZl4IOkXtyVed8dSPlQPDf2AOlUxj3rNXwjVBVQZVAMqGbeuwu4KCL+nOfvBc6OiMc3tM1Ro0bF9OnTN1TEzKxmJD0eEaNaL7llaC93ktjYMVczMyu59hJQk0gXTUjSWNLl2RsaczUzs5Lr1HqRtidpAvBxoKekBuA80uWXRMQVwN2kvz+ax9oxVzMzq2N1EVARcUwr7wf50nIzM2sf2ssQn5mZtTMOKDMzKyUHlJmZlZIDyszMSskBZWZmpeSAMjOzUnJAmZlZKTmgzMyslBxQZmZWSg4oMzMrJQeUmZmVkgPKzMxKyQFlZmal5IAyM7NSckCZmVkpOaDMzKyUHFBmZlZKDigzMyslB5SZmZWSA8rMzErJAWVmZqXkgDIzs1JyQJmZWSk5oMzMrJQcUGZmVkoOKDMzKyUHlJmZlZIDyszMSskBZWZmpeSAMjOzUnJAmZlZKTmgzMyslBxQZmZWSg4oMzMrJQeUmZmVkgPKzMxKyQFlZmal5IAyM7NSckCZmVkp1UVASTpU0vOS5kk6p5n3+0m6X9IMSU9JOrwt6mlmZptO6QNKUkfgcuAwYChwjKShFcW+D9wcESOA8cB/1raWZma2qZU+oIAxwLyIWBARK4GJwBEVZQLokae3BZbWsH5mZrYZdGrrClRhV2BxYb4B2LeizA+BP0r6OtANOLg2VTMzs82lHnpQamZZVMwfA1wbEX2Aw4EbJDXbNkknS5ouaforr7yyiatqZmabSj0EVAPQtzDfh/WH8E4EbgaIiEeA9wE9m9tYRFwVEaMiYlSvXr02Q3XNzGxTqIeAmgYMljRAUmfSRRCTKsq8CBwEIGlPUkC5e2RmVsdKH1AR0QicDkwGZpOu1psl6QJJ43Kx7wAnSZoJTABOiIjKYUAzM6sj9XCRBBFxN3B3xbIfFKafBT5S63qZmdnmU/oelJmZbZkcUGZmVkoOKDMzKyUHlJmZlZIDyszMSskBZWZmpeSAMjOzUnJAmZlZKTmgzMyslBxQZmZWSg4oMzMrJQeUmZmVkgPKzMxKyQFlZmal5IAyM7NSckCZmVkpOaDMzKyUHFBmZlZKDigzMyslB5SZmZWSA8rMzErJAWVmZqXkgDIzs1JyQJmZWSk5oMzMrJQcUGZmVkoOKDMzKyUHlJmZlZIDyszMSskBZWZmpeSAMjOzUnJAmZlZKTmgzMyslBxQZmZWSg4oMzMrJQeUmZmVkgPKzMxKyQFlZmal5IAyM7NSckCZmVkp1UVASTpU0vOS5kk6p4Uyn5f0rKRZkm6sdR3NzGzT6tTWFWiNpI7A5cAngQZgmqRJEfFsocxg4LvARyLib5J2apvampnZplIPPagxwLyIWBARK4GJwBEVZU4CLo+IvwFExLIa19HMzDaxegioXYHFhfmGvKxoCDBE0sOSpko6tKWNSTpZ0nRJ01955ZXNUF0zM9sUajLEJ6nHht6PiDc2tHpzq1TMdwIGAx8H+gAPSdorIv63mX1dBVwFMGrUqMrtmJlZSdTqHNQsUqgUw6ZpPoB+G1i3AehbmO8DLG2mzNSIWAUslPQ8KbCmvcd6m5lZG6lJQEVE39ZLtWgaMFjSAGAJMB44tqLMb4FjgGsl9SQN+S14D/s0M7M2VvNzUJLGS/penu4jaeSGykdEI3A6MBmYDdwcEbMkXSBpXC42GXhV0rPA/cBZEfHq5muFmZltboqo3WkYSZcBWwH7R8SeknYAJkfE6JpVomDUqFExffr0tti1mdl6JD0eEaPauh5lUeu/g9ovIvaRNAMgIl6T1LnGdTAzszpQ6yG+VZI6kK/Ck7QjsLrGdTAzszpQ64C6HLgN6CXpfODPwCU1roOZmdWBmg7xRcT1kh4HDs6LPhcRz9SyDmZmVh/a4l58HYFVpGG+eriThZmZtYGaBoSkc4EJwC6kP7i9UdJ3a1kHMzOrD7XuQX0BGBkRbwNIuhB4HLioxvUwM7OSq/UQ2wusG4qd8B0fzMysGbW6WewvSOec3gZmSZqc5z9FupLPzMxsHbUa4mu6Um8WcFdh+dQa7d/MzOpMrW4W+5ta7MfMzNqPml4kIWl34EJgKPC+puURMaSW9TAzs/Kr9UUS1wLXkJ4DdRhwM+kR7mZmZuuodUB1jYjJABExPyK+D3yixnUwM7M6UOu/g1ohScB8SaeQHkC4U43rYGZmdaDWAfUtoDvwDdK5qG2BL9e4DmZmVgdqfbPYR/Pk34F/ruW+zcysvtTqD3XvID8DqjkRcWQt6mFmZvWjVj2oy2q0HzMzaydq9Ye699ZiP2Zm1n74eUxmZlZKDigzMyulNgkoSV3aYr9mZlY/av1E3TGSngbm5vkPSfqPWtbBzMzqQ617UJcC/wi8ChARM/GtjszMrBm1DqgOEfFCxbJ3alwHMzOrA7W+1dFiSWOAkNQR+Dowp8Z1MDOzOlDrHtSpwLeBfsBfgLF5mZmZ2TpqfS++ZcD4Wu7TzMzqU62fqPsrmrknX0ScXMt6mJlZ+dX6HNSfCtPvAz4LLK5xHczMrA7UeojvpuK8pBuA/6llHczMrD609a2OBgC7tXEdzMyshGp9DupvrD0H1QF4DTinlnUwM7P6ULOAkiTgQ8CSvGh1RLT4EEMzM9uy1WyIL4fRHRHxTn45nMzMrEW1Pgf1mKR9arxPMzOrQzUZ4pPUKSIagY8CJ0maD7wFiNS5cmiZmdk6anUO6jFgH+AzNdqfmZnVuVoN8QkgIuY392p1ZelQSc9Lmiepxav+JB0lKSSN2pSVNzOz2qtVD6qXpG+39GZE/Lyl9/Jdzy8HPgk0ANMkTYqIZyvKbQN8A3h001TZzMzaUq16UB2B7sA2Lbw2ZAwwLyIWRMRKYCJwRDPlfgT8BFi+qSptZmZtp1Y9qJci4oJ3ue6urHu/vgZg32IBSSOAvhHxe0lnbmhjkk4GTgbo16/fu6ySmZltbjU9B7UJ113zN1SSOgC/AL5TzcYi4qqIGBURo3r16vUeqmVmZptTrQLqoPewbgPQtzDfB1hamN8G2At4QNIi0kMQJ/lCCTOz+laTgIqI197D6tOAwZIGSOpMeuDhpMK2X4+InhHRPyL6A1OBcREx/T1V2szM2lRb3828VfkPfE8HJgOzgZsjYpakCySNa9vamZnZ5lLrBxa+KxFxN3B3xbIftFD247Wok5mZbV6l70GZmdmWyQFlZmal5IAyM7NSckCZmVkpOaDMzKyUHFBmZlZKDigzMyslB5SZmZWSA8rMzErJAWVmZqXkgDIzs1JyQJmZWSk5oMzMrJQcUGZmVkoOKDMzKyUHlJmZlZIDyszMSskBZWZmpeSAMjOzUnJAmZlZKTmgzMyslBxQZmZWSg4oMzMrJQeUmZmVkgPKzMxKyQFlZmal5IAyM7NSckCZmVkpOaDMzKyUHFBmZlZKDigzMyslB5SZmZWSA8rMzErJAWVmZqXkgDIzs1JyQJmZWSk5oMzMrJQcUGZmVkoOKDMzK6W6CChJh0p6XtI8Sec08/63JT0r6SlJ90rarS3qaWZmm07pA0pSR+By4DBgKHCMpKEVxWYAoyJiGHAr8JPa1tLMzDa10gcUMAaYFxELImIlMBE4olggIu6PiLfz7FSgT43raGZmm1g9BNSuwOLCfENe1pITgT9s1hqZmdlm16mtK1AFNbMsmi0ofQEYBRzQ4sakk4GTAfr167cp6mdmZptBPfSgGoC+hfk+wNLKQpIOBs4FxkXEipY2FhFXRcSoiBjVq1evTV5ZMzPbNOohoKYBgyUNkNQZGA9MKhaQNAK4khROy9qgjmZmtomVPqAiohE4HZgMzAZujohZki6QNC4X+ynQHbhF0pOSJrWwOTMzqxP1cA6KiLgbuLti2Q8K0wfXvFJmZrZZlb4HZWZmWyYHlJmZlZIDyszMSskBZWZmpeSAMjOzUnJAmZlZKTmgzMyslBxQZmZWSg4oMzMrJQeUmZmVkgPKzMxKyQFlZmal5IAyM7NSckCZmVkpOaDMzKyUHFBmZlZKDigzMyslB5SZmZWSA8rMzErJAWVmZqXkgDIzs1JyQJmZWSk5oMzMrJQcUGZmVkoOKDMzKyUHlJmZlZIDyszMSskBZWZmpeSAMjOzUnJAmZlZKTmgzMyslBxQZmZWSg4oMzMrJQeUmZmVkgPKzMxKyQFlZmalVG1AHQo8D8wDzmnm/d2Ae4GngAeAPhXv9wCWAJcVlh0DPJ3XuQfoWbHOmUAUln8AeARYkd9r0he4H5gNzAK+WbGdr+e6zwJ+kpd1Bq6ZOHFiM00xs7K455572GOPPRg0aBAXX3zxeu+vWLGCo48+mkGDBrHvvvuyaNEiABYtWsTWW2/N8OHDGT58OKeccsqadc4991z69u1L9+7d19nWtddeS69evdas8+tf/3rNe2effTYf/OAH2XPPPfnGN75BRPD3v/99Tdnhw4fTs2dPzjjjDACuuOIK9t57b4YPH85HP/pRnn322TXbuuiiixg0aBB77LEHkydPXrP8y1/+MsCHJD1TrJekH0paIunJ/Do8L99K0nWSnpY0W9J38/I9CmWflPSGpDPyez+S9FRe/kdJu+TlknSppHn5/X0K+79H0v9K+n1Fva6VtLCwn+F5+baSfidppqRZkr5UsV6P3J5iHjQvIlp7dYyI+RExMCI6R8TMiBhaUeaWiDg+Tx8YETdUvP/vEXFjRFyW5ztFxLKI6JnnfxIRPyyU7xsRkyPihUKZnSJidERcGBFnFsr2joh98vQ2ETGnUL9PRMSfIqJLYRtExGkRcc3IkSPDzMqpsbExBg4cGPPnz48VK1bEsGHDYtasWeuUufzyy+OrX/1qRERMmDAhPv/5z0dExMKFC+ODH/xgs9t95JFHYunSpdGtW7d1ll9zzTVx2mmnrVf+4Ycfjv322y8aGxujsbExxo4dG/fff/965fbZZ5+YMmVKRES8/vrra5bfeeedccghh0RExKxZs2LYsGGxfPnyWLBgQQwcODAaGxsjImLKlCkBPAs8E4XjJ/BDoHjMa1p+LDAxT3cFFgH9K8p0BF4GdsvzPQrvfQO4Ik8fDvwBEDAWeLRQ7iDg08DvK7Z9LXBUM/X6HnBJnu4FvAZ0Lrz/78CNwGWV61a+qulBjSH1nBYAK4GJwBEVZYaSelCQejPF90cCOwN/LCxTfnXL//YAlhbe/wVwNqkH1WQZMA1YVbHvl4An8vTfST2pXfP8qcDFpF5X0zYq62tmJfTYY48xaNAgBg4cSOfOnRk/fjx33nnnOmXuvPNOjj/+eACOOuoo7r333qaDYIvGjh1L7969q66HJJYvX87KlStZsWIFq1atYuedd16nzNy5c1m2bBkf+9jHAOjRo8ea99566y0kranv+PHj6dKlCwMGDGDQoEE89thjAOy///4AjVVXLB0fu0nqBGxNOj6/UVHmIGB+RLwAEBHF97ux9hh7BHB9ztSpwHaSeud17iUdWzemXtsoNbo7KaAaASQ1lwctqiagdgUWF+YbWBsATWYC/5SnPwtsA+yYt/8z4KyK8qtI4fE0KZiGAr/J740jDQfOrKYBFfoDI4BH8/wQ4GN5fgowulDfIzp27PgudmFmtbBkyRL69u27Zr5Pnz4sWbKkxTKdOnVi22235dVXXwVg4cKFjBgxggMOOICHHnqoqn3edtttDBs2jKOOOorFi9Nh78Mf/jCf+MQn6N27N7179+aQQw5hzz33XGe9CRMmcPTRR68JIoDLL7+c3XffnbPPPptLL7206ja14PQ89Ha1pO3zsluBt0hf0l8E/i0iXqtYbzwwobhA0oWSFgPHAT/Ii6s5zjfnwlyvX0jqkpddBuxJOrY/DXwzIlZLaikPWlRNQKmZZZVfUc4EDgBm5H+XkBLza8DdrNtwgK1IATUC2IV0Huq7pG7quaz90FIFpEMlPS9p3u233/7R9Sooddlhhx1ue/rpp5874YQTlkvaIb/VCdie1GU9C7g5t+dqoOH666+vovlm1haa6wkVA2BDZXr37s2LL77IjBkz+PnPf86xxx7LG29Udi7W9elPf5pFixbx1FNPcfDBB6/pmc2bN4/Zs2fT0NDAkiVLuO+++3jwwQfXWXfixIkcc8wx6yw77bTTmD9/Ppdccgk//vGPq25TM/4fsDswnBRGP8vLxwDvkI6hA4DvSBpY2G5n0hf+W4obi4hzI6Iv8N/A6U3Fm9nvhrui6Zj9AdIX/x2Af8nLDwGezPUaDlwmqQc5DyKiMg9aVE1ANZAuRGjSh3WH48jzR5IC59y87HXgw6QPYBHwb8AXSUNuw3OZ+aQP4WZgP9IPYQCph7MI6BMRT+y6665XAIcBQxcuXLjXrFmzdiruvFu3bifdd999Y/bee+/vXXfddT8ALinU/fa8j8eA1aSLLhqBbx133HFVNN/M2kKfPn3W9GIAGhoa2GWXXVos09jYyOuvv84OO+xAly5d2HHHHQEYOXIku+++O3PmzNng/nbccUe6dEmdgJNOOonHH38cgDvuuIOxY8fSvXt3unfvzmGHHcbUqVPXrDdz5kwaGxsZOXJks9sdP348v/3tb6tuU6WI+EtEvBMRq4FfkYIJ0jmoeyJiVUQsAx4GRhVWPQx4IiL+0sKmb2TtyFc1x/nKer2UhwRXANcU6vUl4Pb83jxgISnIPkzqCS4i54Gk9a98KagmoKYBg0nB0ZnUZZxUUaZnYVvfJfVQIHUh+5GG3s4EriddBbiENKzXK5f7JOnc0dPATrl8f6Dh0EMPPXnJkiXPR8SCiFg5YMCAZ5544om9CvvWxIkTv9u1a9eHgZ+Tur0H5fHP3wIH5nJDcv3/Suqpdaui7WbWRkaPHs3cuXNZuHAhK1euZOLEiYwbN26dMuPGjeO6664D4NZbb+XAAw9EEq+88grvvPMOAAsWLGDu3LkMHDhwvX0UvfTSS2umJ02atGYYr1+/fkyZMoXGxkZWrVrFlClT1hnimzBhwnq9p7lz566Zvuuuuxg8ePCa+k6cOJEVK1awcOFC5s6dy5gxY9iQpnNB2WeBpqv8XgQOzFfgdSONFD1XKHsM6w/vDS7MjiuUn0QKDEkaC7weES+xAU31ysfaz1TU66D83s7AHsCCiDguIvpFRH9yHkREc1eFr91HaycUs8OBX5KuCLkauBC4AJieG3YUcBGpp/IgcBprL0xocgIp3Zu6lKeQLglfBbyQ33+1Yp1Fo0ePPn/69OkfiYjvA9NXrly5Y2NjY4euXbu+Qgq5YcBDK1eufK5z584rAI488sid77jjjr3zCcGrST22lcCZkgYNGTLk9MmTJw9Zvnx559dee+3NE044YdHcuXNXApx11lm9TjnllJ0aGxujoaFh5bHHHrvwL3/5yzvFSjU0NOw9atSo2S+//HJj3759O0e5n7oAAAguSURBVE2bNm1ot27dOkZEvP3226v33HPPZ0aPHt118uTJe8yZM+f/Vq9eDcB555235Oabb369S5cuuummm/oPHTq066pVq1afc845Db/73e/+PmTIkM6TJ08esnr16nj55ZdXvZd6VXyOPUnBvCVwW9uXbUlfVt8htfVl0tDRW6RRGpG+PHfNZeaTfte3I51DifxamstD6h3sQDrVsCpvd2kuv10u30g60C7P6+xGOuFP3k5DoY57A3MLZSH1Rnq0sK33s/bPZ15k7YUNA0inJN4B/gKcFxG/kXQD6RgWpJGlr0bES5K6k3ouQ/PncE1E/BRAUlfSqZWBEdHUbiTdRgqM1aTj7ikRsSSHzGWkPyl6G/hSREzP6zxE6gF1Jx2jT4yIyZLuI3UyRBrSOyUi3syXrl8L9M7vXRwR/1X4bJB0AjAqIk5nA6oNqDYj6XPAIRHxlTz/z8CYiPh6ocysXKYhz8/PZSoDr3Lb0yNi1IbKtAdbSjvBbW2PtpR2wpbV1mrUw50kqhkbXVMmX3K5LenSRjMzq1P1EFDTgMGSBuSrUpo7BzYJOD5PHwXcF2XvGpqZ2QZ1ausKtCYiGiWdDkwmnwOLiFmSLgCmR8Qk0t9Q3SBpHqnnNL7KzV+1WSpdPltKO8FtbY+2lHbCltXWVpX+HJSZmW2Z6mGIz8zMtkAOKDMzK6V2H1DF2yRJWu+PwiR1kXRTfv9RSf1rX8tNo4q2flvSs/neWfdK2q0t6rkptNbWQrmjJIWkurx0t5p2Svp8/rnOknRjreu4qVTx/7efpPslzcj/hw9vi3q+V/l+estU8ViNwvtSC4++2OK0drvzen6RLqqYDwwk3UViJjC0oszXWHvL+fHATW1d783Y1k8AXfP0qe25rbncNqQ/HJ9K+qPANq/7ZviZDibdA3P7PL9TretZw7ZeBZyap4cCi9q63u+yrfsD+1DxWI3C+y0++mJLe7X3HtQYYF7k2yTR/KNCjgCuy9PF2yTVm1bbGhH3R8TbeXYq6z9Ysl5U83MF+BHpIZXLm3mvHlTTzpOAyyPibwCR7slWj6ppa5DuzgDpbx03eK+4soqIB9nw32m2+OiLLU17D6hqbiG/pkxENJJuY7JjTWq3aW3s7fJPJH1Lq0ettlXSCKBvRKzzFNA6U83PdAgwRNLDkqZKOrRmtdu0qmnrD4EvSGogPSXh67RP7/bRF+1O6f8O6j2q5hby7+Y282VUdTskfYF0X8QDNmuNNp8NtjU/d+YXpPs71rNqfqadSMN8Hyf1iB+StFdE/O9mrtumVk1bjwGujYifSfow6W8f94p0l+/2pL0ck96z9t6D2pJuk1TV7fIlHUx6JMq4SLfJr0ettXUbYC/ggXxr/7HApDq8UKLa/793RnrkwkLgeVJg1Ztq2noi6dE8RMQjwPtYe9PV9mSjH33RXrX3gNqSbpPUalvzsNeVpHCq13MV0EpbI+L1iOgZEf0j3dp/KqnN09umuu9aNf9/f0u6+AVJPUlDfgtqWstNo5q2Fh/jsCcpoF6paS1rY6MffdFeteshvti8t0kqlSrb+lPSLfNvydeBvBgR41rcaElV2da6V2U7JwOfkvQs6TENZ0Urd/Evoyrb+h3gV5K+RRryOqEev0xKmkAaku2Zz6edR3r0BxFxBen82uHAPPKjL9qmpm3PtzoyM7NSau9DfGZmVqccUGZmVkoOKDMzKyUHlJmZlZIDyszMSskBZXVH0juSnpT0jKRbJHV9D9v6uKTf5+lxrdwZfTtJX3sX+/ihpDOrXV5R5lpJR23Evvq3dJdss3rjgLJ69H8RMTwi9gJWAqcU38x/4LjR/7cjYlJEXLyBItuR7n5vZjXggLJ69xAwKPccZkv6T+AJoK+kT0l6RNITuafVHdY8d+g5SX8GjmzakKQTJF2Wp3eWdIekmfm1H3AxsHvuvf00lztL0rT83J7zC9s6Nz/b6E/AHq01QtJJeTszJd1W0Ss8WNJDkuZI+sdcvqOknxb2/dX3+kGalY0DyupWvnfiYcDTedEepMcUjADeAr4PHBwR+wDTgW9Leh/wK+DTwMeA97ew+UuBKRHxIdKze2YB5wDzc+/tLEmfIt33bgwwHBgpaX9JI0l3JBlBCsDRVTTn9ogYnfc3m3TfuSb9STf2/QfgityGE0m3wBmdt3+SpAFV7MesbrTrWx1Zu7W1pCfz9EOk21XtAryQn58D6QaxQ4GH822dOgOPAB8AFkbEXABJ/wWc3Mw+DgS+CBAR7wCvS9q+osyn8mtGnu9OCqxtgDuanr0lqZpbL+0l6cekYcTupFv+NLk537F7rqQFuQ2fAoYVzk9tm/c9p4p9mdUFB5TVo/+LiOHFBTmE3iouAv4nIo6pKDecTffoAgEXRcSVFfs4413s41rgMxExU9IJpHu1NancVuR9fz0iikGGpP4buV+z0vIQn7VXU4GPSBoEIKmrpCHAc8AASbvncse0sP69wKl53Y6SegB/J/WOmkwGvlw4t7WrpJ1Ij5n/rKStJW1DGk5szTbAS5K2Ao6reO9zkjrkOg8kPVJjMnBqLo+kIZK6VbEfs7rhHpS1SxHxSu6JTJDUJS/+fkTMkXQycJekvwJ/Jj07qtI3gasknUi6S/ipEfGI0pNrnwH+kM9D7Qk8kntwbwJfiIgnJN0EPAm8QBqGbM2/Ao/m8k+zbhA+D0wBdgZOiYjlkn5NOjf1hNLOXwE+U92nY1YffDdzMzMrJQ/xmZlZKTmgzMyslBxQZmZWSg4oMzMrJQeUmZmVkgPKzMxKyQFlZmal9P8BOu0MqWbMGWUAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    " def plot_confusion_matrix(cm, classes, Normalizer=False, title='Confusion matrix',cmap = plt.cm.Blues):\n",
    "     plt.imshow(cm, interpolation='nearest', cmap=cmap)\n",
    "     plt.title(title)\n",
    "     plt.colorbar()\n",
    "     tick_marks = np.arange(len(classes))\n",
    "     plt.xticks(tick_marks, classes, rotation=45)\n",
    "     plt.yticks(tick_marks, classes)\n",
    "if Normalizer:\n",
    "        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]\n",
    "        print(\"Normalized confusion matrix\")\n",
    "else:\n",
    "        print('Confusion matrix, without normalization')\n",
    "thresh = cm.max() / 2.\n",
    "for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):\n",
    "        plt.text(j, i, cm[i, j],\n",
    "                 horizontalalignment=\"center\",\n",
    "                 color=\"white\" if cm[i, j] > thresh else \"black\")\n",
    "plt.tight_layout()\n",
    "plt.ylabel('True label')\n",
    "plt.xlabel('Predicted label')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 139,
   "metadata": {},
   "outputs": [],
   "source": [
    "clf = MultinomialNB() "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 140,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "accuracy:   0.857\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "clf.fit(tfidf_train, y_train)\n",
    "pred = clf.predict(tfidf_test)\n",
    "score = metrics.accuracy_score(y_test, pred)\n",
    "print(\"accuracy:   %0.3f\" % score)\n",
    "CM = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])\n",
    "plot_confusion_matrix(CM, classes=['FAKE', 'REAL'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 141,
   "metadata": {},
   "outputs": [],
   "source": [
    "linear_clf = PassiveAggressiveClassifier()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 142,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "accuracy:   0.933\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "linear_clf.fit(tfidf_train, y_train)\n",
    "pred = linear_clf.predict(tfidf_test)\n",
    "score = metrics.accuracy_score(y_test, pred)\n",
    "print(\"accuracy:   %0.3f\" % score)\n",
    "cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])\n",
    "plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
