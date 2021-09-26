from matplotlib.backends.backend_agg import RendererAgg
import streamlit as st
import numpy as np
import pandas as pd
import xmltodict
from pandas import json_normalize
import urllib.request
import seaborn as sns
import matplotlib
from matplotlib.figure import Figure
from PIL import Image
import gender_guesser.detector as gender
from streamlit_lottie import st_lottie
import requests

st.set_page_config(layout="wide")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_book = load_lottieurl('https://assets4.lottiefiles.com/temp/lf20_aKAfIn.json')
st_lottie(lottie_book, speed=1, height=200, key="initial")


matplotlib.use("agg")

_lock = RendererAgg.lock


sns.set_style('darkgrid')
row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (.1, 2, .2, 1, .1))

row0_1.title('Analyzing Your Goodreads Reading Habits')


with row0_2:
    st.write('')

row0_2.subheader(
    'A Streamlit web app by [Tyler Richards](http://www.tylerjrichards.com), get my new book on Streamlit [here!](https://www.amazon.com/Getting-Started-Streamlit-Data-Science/dp/180056550X)')

row1_spacer1, row1_1, row1_spacer2 = st.columns((.1, 3.2, .1))

with row1_1:
    st.markdown("Hey there! Welcome to Tyler's Goodreads Analysis App. This app scrapes (and never keeps or stores!) the books you've read and analyzes data about your book list, including estimating the gender breakdown of the authors, and looking at the distribution of the age and length of book you read. After some nice graphs, it tries to recommend a curated book list to you from a famous public reader, like Barack Obama or Bill Gates. One last tip, if you're on a mobile device, switch over to landscape for viewing ease. Give it a go!")
    st.markdown(
        "**To begin, please enter the link to your [Goodreads profile](https://www.goodreads.com/) (or just use mine!).** 👇")

row2_spacer1, row2_1, row2_spacer2 = st.columns((.1, 3.2, .1))
with row2_1:
    default_username = st.selectbox("Select one of our sample Goodreads profiles", (
        "89659767-tyler-richards", "7128368-amanda", "17864196-adrien-treuille", "133664988-jordan-pierre"))
    st.markdown("**or**")
    user_input = st.text_input(
        "Input your own Goodreads Link (e.g. https://www.goodreads.com/user/show/89659767-tyler-richards)")
    need_help = st.expander('Need help? 👉')
    with need_help:
        st.markdown(
            "Having trouble finding your Goodreads profile? Head to the [Goodreads website](https://www.goodreads.com/) and click profile in the top right corner.")

    if not user_input:
        user_input = f"https://www.goodreads.com/user/show/{default_username}"

user_id = ''.join(filter(lambda i: i.isdigit(), user_input))
user_name = user_input.split(user_id, 1)[1].split('-', 1)[1].replace('-', ' ')


@st.cache
def get_user_data(user_id, key='ZRnySx6awjQuExO9tKEJXw', v='2', shelf='read', per_page='200'):
    api_url_base = 'https://www.goodreads.com/review/list/'
    final_url = api_url_base + user_id + '.xml?key=' + key + \
        '&v=' + v + '&shelf=' + shelf + '&per_page=' + per_page
    contents = urllib.request.urlopen(final_url).read()
    return(contents)


user_input = str(user_input)
contents = get_user_data(user_id=user_id, v='2', shelf='read', per_page='200')
contents = xmltodict.parse(contents)

line1_spacer1, line1_1, line1_spacer2 = st.columns((.1, 3.2, .1))

with line1_1:
    if int(contents['GoodreadsResponse']['reviews']['@total']) == 0:
        st.write("Looks like you did not read any books on Goodreads. Add some books to your profile or try a different profile")
        st.stop()

    st.header('Analyzing the Reading History of: **{}**'.format(user_name))

df = json_normalize(contents['GoodreadsResponse']['reviews']['review'])
u_books = len(df['book.id.#text'].unique())
u_authors = len(df['book.authors.author.id'].unique())
df['read_at_year'] = [i[-4:] if i != None else i for i in df['read_at']]
has_records = any(df['read_at_year'])

st.write('')
row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns(
    (.1, 1, .1, 1, .1))


with row3_1, _lock:
    st.subheader('Books Read')
    if has_records:
        year_df = pd.DataFrame(
            df['read_at_year'].dropna().value_counts()).reset_index()
        year_df = year_df.sort_values(by='index')
        fig = Figure()
        ax = fig.subplots()
        sns.barplot(x=year_df['index'],
                    y=year_df['read_at_year'], color='goldenrod', ax=ax)
        ax.set_xlabel('Year')
        ax.set_ylabel('Books Read')
        st.pyplot(fig)
    else:
        st.markdown(
            "We do not have information to find out _when_ you read your books")

    st.markdown("It looks like you've read a grand total of **{} books with {} authors,** with {} being your most read author! That's awesome. Here's what your reading habits look like since you've started using Goodreads.".format(
        u_books, u_authors, df['book.authors.author.name'].mode()[0]))


with row3_2, _lock:
    st.subheader("Book Age")
    fig = Figure()
    ax = fig.subplots()
    sns.histplot(pd.to_numeric(df['book.publication_year'], errors='coerce').dropna(
    ).astype(np.int64), kde_kws={'clip': (0.0, 2020)}, ax=ax, kde=True)
    ax.set_xlabel('Book Publication Year')
    ax.set_ylabel('Density')
    st.pyplot(fig)

    avg_book_year = str(
        int(np.mean(pd.to_numeric(df['book.publication_year']))))
    row = df.sort_values(by='book.publication_year', ascending=False).head(1)
    oldest_book = row['book.title_without_series'].iloc[0]
    row_young = df.sort_values(by='book.publication_year').head(1)
    youngest_book = row_young['book.title_without_series'].iloc[0]

    st.markdown("Looks like the average publication date is around **{}**, with your oldest book being **{}** and your youngest being **{}**.".format(
        avg_book_year, oldest_book, youngest_book))
    st.markdown("Note that the publication date on Goodreads is the **last** publication date, so the data is altered for any book that has been republished by a publisher.")

st.write('')
row4_space1, row4_1, row4_space2, row4_2, row4_space3 = st.columns(
    (.1, 1, .1, 1, .1))

with row4_1, _lock:
    st.subheader("How Do You Rate Your Reads?")
    rating_df = pd.DataFrame(pd.to_numeric(df[df['rating'].isin(
        ['1', '2', '3', '4', '5'])]['rating']).value_counts(normalize=True)).reset_index()
    fig = Figure()
    ax = fig.subplots()
    sns.barplot(x=rating_df['index'],
                y=rating_df['rating'], color="goldenrod", ax=ax)
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Your Book Ratings')
    st.pyplot(fig)

    df['rating_diff'] = pd.to_numeric(df['book.average_rating']) - pd.to_numeric(
        df[df['rating'].isin(['1', '2', '3', '4', '5'])]['rating'])

    difference = np.mean(df['rating_diff'].dropna())
    row_diff = df[abs(df['rating_diff']) == abs(df['rating_diff']).max()]
    title_diff = row_diff['book.title_without_series'].iloc[0]
    rating_diff = row_diff['rating'].iloc[0]
    pop_rating_diff = row_diff['book.average_rating'].iloc[0]

    if difference > 0:
        st.markdown("It looks like on average you rate books **lower** than the average Goodreads user, **by about {} points**. You differed from the crowd most on the book {} where you rated the book {} stars while the general readership rated the book {}".format(
            abs(round(difference, 3)), title_diff, rating_diff, pop_rating_diff))
    else:
        st.markdown("It looks like on average you rate books **higher** than the average Goodreads user, **by about {} points**. You differed from the crowd most on the book {} where you rated the book {} stars while the general readership rated the book {}".format(
            abs(round(difference, 3)), title_diff, rating_diff, pop_rating_diff))

with row4_2, _lock:
    st.subheader("How do Goodreads Users Rate Your Reads?")
    fig = Figure()
    ax = fig.subplots()
    sns.histplot(pd.to_numeric(df['book.average_rating'], errors='coerce').dropna(
    ), kde_kws={'clip': (0.0, 5.0)}, ax=ax, kde=True)
    ax.set_xlabel('Goodreads Book Ratings')
    ax.set_ylabel('Density')
    st.pyplot(fig)
    st.markdown("Here is the distribution of average rating by other Goodreads users for the books that you've read. Note that this is a distribution of averages, which explains the lack of extreme values!")

st.write('')
row5_space1, row5_1, row5_space2, row5_2, row5_space3 = st.columns(
    (.1, 1, .1, 1, .1))

with row5_1, _lock:
    # page breakdown
    st.subheader('Book Length Distribution')
    fig = Figure()
    ax = fig.subplots()
    sns.histplot(pd.to_numeric(df['book.num_pages'].dropna()), ax=ax, kde=True)
    ax.set_xlabel('Number of Pages')
    ax.set_ylabel('Density')
    st.pyplot(fig)

    book_len_avg = round(np.mean(pd.to_numeric(df['book.num_pages'].dropna())))
    book_len_max = pd.to_numeric(df['book.num_pages']).max()
    row_long = df[pd.to_numeric(df['book.num_pages']) == book_len_max]
    longest_book = row_long['book.title_without_series'].iloc[0]

    st.markdown("Your average book length is **{} pages**, and your longest book read is **{} at {} pages!**.".format(
        book_len_avg, longest_book, int(book_len_max)))


with row5_2, _lock:
    # length of time until completion
    st.subheader('How Quickly Do You Read?')
    if has_records:
        df['days_to_complete'] = (pd.to_datetime(
            df['read_at']) - pd.to_datetime(df['started_at'])).dt.days
        fig = Figure()
        ax = fig.subplots()
        sns.histplot(pd.to_numeric(
            df['days_to_complete'].dropna()), ax=ax, kde=True)
        ax.set_xlabel('Days')
        ax.set_ylabel('Density')
        st.pyplot(fig)
        days_to_complete = pd.to_numeric(df['days_to_complete'].dropna())
        time_len_avg = 0
        if len(days_to_complete):
            time_len_avg = round(np.mean(days_to_complete))
        st.markdown("On average, it takes you **{} days** between you putting on Goodreads that you're reading a title, and you getting through it! Now let's move on to a gender breakdown of your authors.".format(time_len_avg))
    else:
        st.markdown(
            "We do not have information to find out _when_ you finished reading your books")


st.write('')
row6_space1, row6_1, row6_space2, row6_2, row6_space3 = st.columns(
    (.1, 1, .1, 1, .1))


with row6_1, _lock:
    st.subheader('Gender Breakdown')
    # gender algo
    d = gender.Detector()
    new = df['book.authors.author.name'].str.split(" ", n=1, expand=True)

    df["first_name"] = new[0]
    df['author_gender'] = df['first_name'].apply(d.get_gender)
    df.loc[df['author_gender'] == 'mostly_male', 'author_gender'] = 'male'
    df.loc[df['author_gender'] == 'mostly_female', 'author_gender'] = 'female'

    author_gender_df = pd.DataFrame(
        df['author_gender'].value_counts(normalize=True)).reset_index()
    fig = Figure()
    ax = fig.subplots()
    sns.barplot(x=author_gender_df['index'],
                y=author_gender_df['author_gender'], color="goldenrod", ax=ax)
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Gender')
    st.pyplot(fig)
    st.markdown('To get the gender breakdown of the books you have read, this next bit takes the first name of the authors and uses that to predict their gender. These algorithms are far from perfect, and tend to miss non-Western/non-English genders often so take this graph with a grain of salt.')
    st.markdown("Note: the package I'm using for this prediction outputs 'andy', which stands for androgenous, whenever multiple genders are nearly equally likely (at some threshold of confidence). It is not, sadly, a prediction of a new gender called andy.")

with row6_2, _lock:
    st.subheader("Gender Distribution Over Time")

    if has_records:
        year_author_df = pd.DataFrame(df.groupby(['read_at_year'])[
            'author_gender'].value_counts(normalize=True))
        year_author_df.columns = ['Percentage']
        year_author_df.reset_index(inplace=True)
        year_author_df = year_author_df[year_author_df['read_at_year'] != '']
        fig = Figure()
        ax = fig.subplots()
        sns.lineplot(x=year_author_df['read_at_year'], y=year_author_df['Percentage'],
                     hue=year_author_df['author_gender'], ax=ax)
        ax.set_xlabel('Year')
        ax.set_ylabel('Percentage')
        st.pyplot(fig)
        st.markdown(
            "Here you can see the gender distribution over time to see how your reading habits may have changed.")
    else:
        st.markdown(
            "We do not have information to find out _when_ you read your books")
    st.markdown(
        "Want to read more books written by women? [Here](https://www.penguin.co.uk/articles/2019/mar/best-books-by-female-authors.html) is a great list from Penguin that should be a good start (I'm trying to do better at this myself!).")

st.write('')
row7_spacer1, row7_1, row7_spacer2 = st.columns((.1, 3.2, .1))

with row7_1:
    st.header("**Book List Recommendation for {}**".format(user_name))

    reco_df = pd.read_csv('recommendations_df.csv')
    unique_list_books = df['book.title'].unique()
    reco_df['did_user_read'] = reco_df['goodreads_title'].isin(
        unique_list_books)
    most_in_common = pd.DataFrame(reco_df.groupby('recommender_name').sum(
    )).reset_index().sort_values(by='did_user_read', ascending=False).iloc[0][0]
    avg_in_common = pd.DataFrame(reco_df.groupby('recommender_name').mean(
    )).reset_index().sort_values(by='did_user_read', ascending=False).iloc[0][0]
    most_recommended = reco_df[reco_df['recommender_name'] == most_in_common]['recommender'].iloc[0]
    avg_recommended = reco_df[reco_df['recommender_name'] == avg_in_common]['recommender'].iloc[0]

    def get_link(recommended):
        if '-' not in recommended:
            link = 'https://bookschatter.com/books/' + recommended
        elif '-' in recommended:
            link = 'https://www.mostrecommendedbooks.com/' + recommended + '-books'
        return(link)
    st.markdown('For one last bit of analysis, we scraped a few hundred book lists from famous thinkers in technology, media, and government (everyone from Barack and Michelle Obama to Keith Rabois and Naval Ravikant). We took your list of books read and tried to recommend one of their lists to book through based on information we gleaned from your list')
    st.markdown("You read the most books in common with **{}**, and your book list is the most similar on average to **{}**. Find their book lists [here]({}) and [here]({}) respectively.".format(
        most_in_common, avg_in_common, get_link(most_recommended), get_link(avg_recommended)))

    st.markdown('***')
    st.markdown(
        "Thanks for going through this mini-analysis with me! I'd love feedback on this, so if you want to reach out you can find me on [twitter] (https://twitter.com/tylerjrichards) or my [website](http://www.tylerjrichards.com/).")
