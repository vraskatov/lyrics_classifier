'''This program will scrape lyrics of two artists from lyrics.com and build a
model that will classify user input as the style of one or the other band.
'''

import os
import time
import pickle
from sklearn.model_selection import train_test_split
from utils import get_songs, folder_view, corpus_builder, nlp_pipeline, vectorizing, build_bayes

print('\n')
print('#######################################')
print('## Welcome to the lyrics classifier! ##')
print('#######################################\n')
time.sleep(2.4)

print('This program will build a classifier based on lyrics of bands/artists for you.')
time.sleep(2.4)
print('For that it can scrape them from lyrics.com or analyze already locally existing data.')

time.sleep(5)

try:
    os.makedirs(f'lyrics/', exist_ok=False)
    print('\nThe folder "lyrics" has been automatically created for you next to this py file.')
    time.sleep(2.4)
    print('You will find the scraped folders with lyrics inside of that folder.')
    time.sleep(2.4)
    print('If you already have your data, make sure to put it in subfolders of "lyrics".')
    time.sleep(4)
except FileExistsError:
    pass

answer = ''
while answer not in ('yes', 'no'):
    print('\nWould you like to scrape some lyrics?')
    time.sleep(2.4)
    answer = input('Enter yes to scrape or no if you already have your data:\n\n')
    if answer == 'yes':
        get_songs()
        print('\nYou need two bands to construct the classifier.')
        time.sleep(2.4)
        band_request = ''
        while band_request != 'no':
            print('Would you like to scrape some more lyrics?')
            time.sleep(2.4)
            band_request = input('Please enter yes or no:\n\n')
            if band_request == 'yes':
                get_songs()
            elif band_request == 'no':
                print('Okay, let\'s have a look at your folders then.')
                break
            else:
                print('Sorry, that was no valid input.')
            # just one time, but with another loop ;)

        # You need two bands to make this program work.
        # Do you want to scrape another band?

        # load_data()
    elif answer == 'no':
        print('\nOkay, let\'s get directly to your data then.')
        # show folders (maybe for the other as well)
        break
        # load_data()
    else:
        print('Sorry, that was not a valid answer.')
        print('If you\'re trying to quit the program, you should consider pressing Ctrl+C.')

print('Do you want to have a look at the subfolders of the folder "lyrics"?')
time.sleep(2.4)
folder_view()
print('Going to the corpus builder.')
time.sleep(2)
corpus, labels = corpus_builder() # Can be also written as a function called inside a function
corpus = nlp_pipeline(corpus)
print('Done with NLP.')
time.sleep(1)
print('Splitting corpus in train and test an vectorizing words.')
X_train, X_test, y_train, y_test = train_test_split(corpus, labels, random_state=12)

X_train, X_test, vectorizer = vectorizing(X_train, X_test)
time.sleep(1)
print('\nNow it\'s time to build your model and predict some lyrics.')
time.sleep(1)

bayes_model = build_bayes(X_train, X_test, y_train, y_test, vectorizer)

time.sleep(1)

model_save = ''
while model_save not in ('yes', 'no'):
    model_save = input('\nWould you like to save that model? Enter yes or no:\n\n')
    time.sleep(2.4)
    if model_save == 'yes':
        model_name = input('\nPlease enter a name for your model:\n\n')
        filename = f'{model_name}.sav'
        pickle.dump(bayes_model, open(filename, 'wb'))
        time.sleep(1.5)
        print(f'\nYour model has been saved under the name {model_name}.sav '
              'next to this script.\n')
        time.sleep(2.4)
    elif model_save == 'no':
        time.sleep(1.5)
        break
    else:
        print('\nSorry, that was no valid input.\n')

print('Thank you for using the lyrics classifier!')
