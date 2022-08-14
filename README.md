OVERVIEW:

The lyrics classifier is a command line tool that lets the user download lyrics of artists from lyrics.com or use texts that already exist locally.
After the amount of texts per band is balanced (optional) and some NLP techniques are applied automatically the classifier will fit a Naive Bayes model
and finally ask the user for input in the style of one of the artists. With diverse bands as Bee Gees and Eminem the program reaches an accuracy of
95,71 percent. After playing around with the command line input the classifier will ask the user, whether he wants to save the built model.
The lyrics classifier was produced in week 4 of the bootcamp at Spiced Academy in Berlin and turned into a command line application later.

RUNNING THE CLASSIFIER: 

0. You need to create an environment and install the requirements from requirements.txt in it or pip install the requirements globally.

1. Run the script:

Open your terminal and navigate to the "lyrics_classifier" folder (where lyrics.py and requirements.txt are located).
In the terminal now type: python lyrics.py and press Enter.

2. Navigate through the program

a. Scraping:

The classifier welcomes you, automatically generates a subfolder "lyrics" and asks you, whether you want to scrape some songtexts.
If you enter 'yes' you will be asked to enter the name of an artist/band.
Make sure to type it exactly as it is written originally e.g. ABBA and not Abba.
If you are not sure how to write it, you can directly look it up on lyrics.com.
After hitting enter the program will get a response of the artist site and remove duplicates before downloading the texts.
While downloading you will see the names of the songs in the terminal. The process might take some time.
Empty texts will be removed automatically after the scraping is complete.
You will find the lyrics in a subfolder named after the band, inside of "lyrics".
After all texts have been scraped, the program will ask you, whether you want to scrape another artist/band.
For the classifier you will need to run this process two times. But you can repeat this process as often as you like.

b. Fitting:

After having a (optional) look at your folders you are asked to type in the names of the two folders that your classifier will trained on.
If the difference between the songs in both folders is over ten you will have the possibility to balance the folders (random removal of diffence).
After that the Naive Bayes classifier is built, the accuracy for train and test is printed and you can start to play around with the classifier
by providing text input.

c. Saving:

Finally, when you're done playing aroung, you can keep the model by dumping it.
