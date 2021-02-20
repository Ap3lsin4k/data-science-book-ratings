import flask
import pickle
import sklearn
import numpy as np
import pandas as pd

f_model = open(f'models/tree.pkl', 'rb')  # TODO: make model

model = pickle.load(f_model)

app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('main.html')

    if flask.request.method == 'POST':
        book_pages = np.log10(int(flask.request.form['book_pages']) + 1)
        book_review_count = np.log10(int(flask.request.form['book_review_count']) + 1)
        book_rating_count = np.log10(int(flask.request.form['book_rating_count']) + 1)

        input_variables = pd.DataFrame([[book_pages, book_review_count, book_rating_count]],
                                           columns=['book_pages', 'book_review_count', 'book_rating_count'],
                                           dtype=float)
        prediction = model.predict(input_variables)[0]
        prediction = np.round(prediction, 2)
        return flask.render_template('main.html',
                                         original_input={
                                             'book_pages': book_pages,
                                             'book_review_count': book_review_count,
                                             'book_rating_count': book_rating_count},
                                         result=prediction)


if __name__ == '__main__':
    app.run()
