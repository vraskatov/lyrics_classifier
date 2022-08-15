'''Functions that are being used inside of lyrics.py for getting songs,
building corpora, balancing datasets, performing NLP techniques and
fitting models as well as typing in own strings to test their performance.
'''

import os
import time
import re
import random
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def get_songs():
    '''A function that gets songs by bandname from lyrics.com avoiding duplicates,
    cleans the filenames and removes files that are too small from the folders.'''
    time.sleep(2.4)

    print('\nLet\'s scrape some lyrics then.')
    # Type in name of band
    band = input('Please type in the name of a band:\n\n')

    # Create folder for band
    band_folder = band.replace(' ', '_')
    try:
        os.makedirs(f'lyrics/{band_folder}', exist_ok=False)
    except FileExistsError:
        print(f'\nFolder for {band} already exists.')

    # Get response and links for band
    band_rep = band.replace(' ', '-')
    site = requests.get(f'https://www.lyrics.com/artist/{band_rep}').text
    linkparts = re.findall(r'href="(\/lyric.+?)">', site)

    print(f'\nThe subsite of {band} has {len(linkparts)} links.')
    time.sleep(2)
    print('Removing duplicates now.\n')
    time.sleep(2)

    # Uniforming of links
    digits_pattern = r'\/lyric\/[0-9]{8}'
    uniform = [x for x in linkparts if re.match(digits_pattern, x)]

    # Remove of lf links
    reduced = []
    for link in uniform:
        if 'lyric-lf/' not in link:
            reduced.append(link)

    # Forward remove
    parts = []
    lyrics_part = []
    start_index = len(band_rep)+17
    for lyric in reduced:
        if lyric[start_index:(start_index+5)] not in parts:
            print(lyric)
            print(lyric[start_index:(start_index+5)])
            lyrics_part.append(lyric)
            parts.append(lyric[27:32])
    # Backward remove
    parts = []
    lyrics = []
    for lyric in lyrics_part:
        if lyric[-5:-1] not in parts:
            lyrics.append(lyric)
            parts.append(lyric[-5:-1])

    links = []
    for link in lyrics:
        links.append('https://www.lyrics.com' + link)
    time.sleep(2)
    if len(lyrics) > 10:
        print(f'Downloading {len(lyrics)} lyrics of {band}:\n')
        band_link = band.replace(' ', '+')
        for link in links:
            try:
                name = link.split(f'{band_link}/')[1]
                name = name.replace('+', ' ')
                name = name.replace('%21', '')
                name = name.replace('%26', '')
                name = name.replace('%27', '')
                name = name.replace('%28', '')
                name = name.replace('%29', '')
                name = name.replace('%2A', '')
                name = name.replace('%2C', '')
                name = name.replace('%3C', '')
                name = name.replace('%3A', '')
                name = name.replace('%A4', '')
                name = name.replace('%A5', '')
                name = name.replace('%B6', '')
                name = name.replace('%3F', '')
                name = name.replace('%BA', '')
                name = name.replace('%B1', '')
                name = name.replace('%E2%80%99', '')
                name = name.replace('%C3%A8', 'e')
                name = name.replace('%C3', '')
                name = re.sub(r'%5B.+', '', name)
                response = requests.get(link)
                with open(f'lyrics/{band_folder}/{name}.txt', 'w') as song:
                    expression = r'<pre[\S\s]+pre>'
                    songstring = re.findall(expression, response.text)[0]
                    soup = BeautifulSoup(songstring, "html.parser")
                    song.write(soup.get_text())
                    print(name)
            except IndexError:
                print(f'Could not download {name}')
        time.sleep(2)
        print('\nRemoving files under 100 Bytes.')

        for entity in os.listdir(f'lyrics/{band_folder}'):
            file = os.path.join(f'lyrics/{band_folder}', entity)
            size = os.path.getsize(file)
            if size < 100:
                os.remove(file)
                print(f'Removed: "{file}" with a size of {size} bytes.')
    else:
        print('Sorry, the artist has under 10 songs, which is not enough.')
        time.sleep(2)
        print('Please check the spelling of artist/band or try it with some other name.')
        time.sleep(2)

def folder_view():
    '''A function that offers the user to get an overview over the lyrics folder
    containing all of the bands subfolders.'''
    view_wish = ''
    while view_wish != 'yes' or view_wish != 'no':
        view_wish = input('Please type in yes or no:\n\n')
        if view_wish == 'yes':
            print(f'\n{os.listdir("lyrics")}\n')
            time.sleep(2.4)
            break

def corpus_builder():
    '''A function that will turn the data from two given folders
    into a list of tuples containing lyrics and artist names.'''
    band_one = input('\nPlease type in subfolder one for the classifier:\n')
    band_two = input('\nPlease type in subfolder two for the classifier:\n')
    band_one_name = band_one.replace('_', ' ')
    band_two_name = band_two.replace('_', ' ')
    subfolder_one = f'lyrics/{band_one}'
    subfolder_two = f'lyrics/{band_two}'
    corpus = None
    try:
        songs_one = len(os.listdir(subfolder_one))
        songs_two = len(os.listdir(subfolder_two))
        print(f'\n{band_one_name} has {songs_one} songs.')
        time.sleep(1)
        print(f'{band_two_name} has {songs_two} songs.\n')
        time.sleep(1)
        if abs(songs_one - songs_two) > 10:
            print('The difference between the song quantities is over 10.'
                  'Do you want to balance your dataset?')
            choice = input('\nPlease type in yes to balance or no to skip balancing:\n\n')
            if choice == 'yes':
                data_balancer(songs_one, songs_two, subfolder_one, subfolder_two)
            elif choice == 'no':
                print('Okay, let\'s skip that part')
            else:
                print('Sorry, that was no valid answer.')
        else:
            pass
        list_of_texts = []
        for file in os.listdir(subfolder_one):
            # open band one file
            with open(os.path.join(subfolder_one, file)) as song:
                text = song.read()
                text = re.sub("'ve ", " have ", text)   # Pre-NLP
                text = re.sub("'ll ", " will ", text)   # Pre-NLP
                text = re.sub("'s ", " ", text)         # Pre-NLP
                text = re.sub("'t ", " not ", text)     # Pre-NLP
                text = re.sub(r"[^a-zA-Z]", " ", text)  # Replace all non-alphabet by spaces
                text = re.sub(r"\s+", " ", text)         # Replace multiple spaces by single space
            # append the text and bandname (as tuple)
            list_of_texts.append((text, band_one_name))
        for file in os.listdir(subfolder_two):
            # open band two file
            with open(os.path.join(subfolder_two, file)) as song:
                text = song.read()
                text = re.sub("'ve ", ' have ', text)
                text = re.sub("'ll ", ' will ', text)
                text = re.sub("'s ", ' ', text)
                text = re.sub("'t ", ' not ', text)
                text = re.sub(r"[^a-zA-Z]", " ", text)
                text = re.sub(r"\s+", " ", text)
            # append the text and bandname (as tuple)
            list_of_texts.append((text, band_two_name))
        corpus, labels = list(zip(*list_of_texts))
        return corpus, labels
    except FileNotFoundError:
        print('Sorry, you must have typed something wrong.'
              'Make sure to type folder names exactly as listed.')
        

def data_balancer(songs_one, songs_two, subfolder_one, subfolder_two):
    '''A function that will balance out the amount of songs per band
    to avoid class imbalance.'''
    difference = songs_one - songs_two
    if difference > 0:
        for _ in range(difference):
            random_song = random.choice(os.listdir(subfolder_one))
            os.remove(f'{subfolder_one}/{random_song}')
        adjusted_one = len(os.listdir(subfolder_one))
        print(f'{adjusted_one} songs in {subfolder_one}) and {songs_two} songs in {subfolder_two}')
    elif difference < 0:
        for _ in range(abs(difference)):
            random_song = random.choice(os.listdir(subfolder_two))
            os.remove(f'{subfolder_two}/{random_song}')
        adjusted_two = len(os.listdir(subfolder_two))
        print(f'The songs have been adjusted to {adjusted_two} songs in both folders now.')
    # Better create copy of folder that is being reduced

def nlp_pipeline(corpus):
    '''A function that will tokenize, lemmatize and clean the word data
    before it is vectorized.'''
    for entry in corpus:
        entry = re.sub("ll", "will", entry)
    corpus = [string.lower() for string in corpus]
    tokenizer = TreebankWordTokenizer()
    lemmatizer = WordNetLemmatizer()
    clean_corpus = []
    for entry in corpus:
        tokens = tokenizer.tokenize(text=entry)
        clean_doc = " ".join(lemmatizer.lemmatize(token, pos='a')
                             for token in tokens if len(token) > 1)
        clean_doc.replace(' ve ', ' have ')
        clean_doc.replace(' s ', '')
        clean_doc.replace(' t ', 'not')
        tokens = tokenizer.tokenize(text=entry)
        clean_doc = " ".join(lemmatizer.lemmatize(token, pos='v')
                             for token in tokens if len(token) > 1)
        clean_corpus.append(clean_doc)
    return clean_corpus

def vectorizing(x_train, x_test):
    '''A function that will vectorize words of two bands
    before fitting models on that basis.'''
    STOPWORDS = stopwords.words('english')
    vectorizer = TfidfVectorizer(ngram_range=(1, 4),
                                 max_df=0.7, min_df=0.01, stop_words=STOPWORDS)
    vectors_train = vectorizer.fit_transform(x_train)
    vectors_test = vectorizer.transform(x_test)
    return vectors_train, vectors_test, vectorizer

def build_logistic(x_train, x_test, y_train, y_test, vectorizer):
    '''A function that will build a Logistic Regression model based on two folders
    with lyrics of two different bands/artists.'''
    logistic_model = LogisticRegression()
    print('Fitting logistic regression model.')
    logistic_model.fit(x_train, y_train)
    train_score = logistic_model.score(x_train, y_train)*100
    print(f'Logistic regression model scores {train_score}% on train')
    test_score = logistic_model.score(x_test, y_test)*100
    print(f'Logistic regression model scores {test_score}% on test')
    choice = ''
    while choice != 'no':
        choice = ''
        tester_text = input('Please type some words in the style of one band:\n')
        band_predict = logistic_model.predict(vectorizer.transform([f'{tester_text}']))
        print(f'Your text was: {tester_text}.')
        print(f'The lyrics classifier predicts {band_predict} for that text.')
        while choice not in ('yes', 'no'):
            choice = input('Do you want to try another line? Type yes or no:\n')

def build_forest(x_train, x_test, y_train, y_test, vectorizer):
    '''A function that will build a Random Forest model based on two folders
    with lyrics of two different bands/artists.'''
    forest_model = RandomForestClassifier(max_depth=3, n_estimators=20,
                                          random_state=42, class_weight='balanced')
    print('Fitting random forest model.')
    forest_model.fit(x_train, y_train)
    train_score = round(forest_model.score(x_train, y_train)*100, 2)
    print(f'Random Forest model scores {train_score}% on train')
    test_score = round(forest_model.score(x_test, y_test)*100, 2)
    print(f'Random Forest model scores {test_score}% on test')
    choice = ''
    while choice != 'no':
        choice = ''
        tester_text = input('Please type some words in the style of one band:\n')
        band_predict = forest_model.predict(vectorizer.transform([f'{tester_text}']))
        print(f'Your text was: {tester_text}.')
        print(f'The lyrics classifier predicts {band_predict} for that text.')
        while choice not in ('yes', 'no'):
            choice = input('Do you want to try another line? Type yes or no:\n')

def build_bayes(x_train, x_test, y_train, y_test, vectorizer):
    '''A function that will build a Naive Bayes model based on two folders
    with lyrics of two different bands/artists.'''
    bayes_model = MultinomialNB()
    print('Fitting Naive Bayes model.\n')
    time.sleep(2)
    bayes_model.fit(x_train, y_train)
    train_score = round(bayes_model.score(x_train, y_train)*100, 2)
    print(f'Naive Bayes model scores {train_score}% on train.')
    test_score = round(bayes_model.score(x_test, y_test)*100, 2)
    print(f'Naive Bayes model scores {test_score}% on test.\n')
    choice = ''
    while choice != 'no':
        choice = ''
        tester_text = input('Please type some words in the style of one band:\n')
        band_predict = bayes_model.predict(vectorizer.transform([f'{tester_text}']))
        print(f'Your text was: {tester_text}.')
        print(f'The lyrics classifier predicts {band_predict} for that text.')
        while choice not in ('yes', 'no'):
            choice = input('Do you want to try another line? Type yes or no:\n')
