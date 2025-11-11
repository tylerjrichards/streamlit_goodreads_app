
import pandas as pd
import pymysql
import os
import numpy as np
from openpyxl import load_workbook
import re
import warnings
import httpx
import json
import time
import chess
import chess.pgn
import io
from datetime import datetime
from pytz import timezone
import pytz
from supabase import create_client, Client



def test_function_test():
    test_return = 5
    return test_return

# Function to parse chess data and insert into a dictionary
def parse_chess_data_to_dict(data_string, data_dict_list=None):
    pattern = re.compile(r'\[(\w+)\s"([^"]+)"\]')
    matches = pattern.findall(data_string)
    game_data_dict = {key: value for key, value in matches}
    if data_dict_list is None:
        data_dict_list = []
    data_dict_list.append(game_data_dict)

    return data_dict_list

def parse_chess_data_to_list(data_string, data_list=None):
    pattern = re.compile(r'\[(\w+)\s"([^"]+)"\]')
    matches = pattern.findall(data_string)
    game_data_list = [value for _, value in matches]
    if data_list is None:
        data_list = []
    data_list.append(game_data_list)

    return data_list

def parse_chess_data_to_dataframe(data_string, df=None):
    pattern = re.compile(r'\[(\w+)\s"([^"]+)"\]')
    matches = pattern.findall(data_string)
    data_dict = {key: value for key, value in matches}
    new_df = pd.DataFrame([data_dict])
    if df is None:
        df = new_df
    else:
        df = pd.concat([df, new_df], ignore_index=True)


    return df

def extract_moves_from_pgn(pgn_string):
    pgn_io = io.StringIO(pgn_string)
    game = chess.pgn.read_game(pgn_io)
    board = game.board()
    moves = []
    for move in game.mainline_moves():
        moves.append(board.san(move))
        board.push(move)
    return moves

    # Create a new column for the extracted moves

def elo_category(elo_opponent, your_elo):
    if elo_opponent < your_elo - 100:
        return 'Weaker'
    elif your_elo - 50 <= elo_opponent <= your_elo + 50:
        return 'Similar'
    else:
        return 'Stronger'

def opening_database():

    data_a = "https://raw.githubusercontent.com/Jayfetra/streamlit_goodreads_app/refs/heads/master/chess-openings-master/a.tsv"
    data_b = "https://raw.githubusercontent.com/Jayfetra/streamlit_goodreads_app/refs/heads/master/chess-openings-master/b.tsv"
    data_c = "https://raw.githubusercontent.com/Jayfetra/streamlit_goodreads_app/refs/heads/master/chess-openings-master/c.tsv"
    data_d = "https://raw.githubusercontent.com/Jayfetra/streamlit_goodreads_app/refs/heads/master/chess-openings-master/d.tsv"
    data_e = "https://raw.githubusercontent.com/Jayfetra/streamlit_goodreads_app/refs/heads/master/chess-openings-master/e.tsv"

    df_a = pd.read_csv(data_a, sep="\t")
    df_b = pd.read_csv(data_b, sep="\t")
    df_c = pd.read_csv(data_c, sep="\t")
    df_d = pd.read_csv(data_d, sep="\t")
    df_e = pd.read_csv(data_e, sep="\t")

    # Combine the DataFrames
    chess_opening_database = pd.concat([df_a, df_b, df_c, df_d, df_e], ignore_index=True)
    chess_moves = []
    # Iterate over the rows of the combined DataFrame
    for index, row in chess_opening_database.iterrows():
        pgn_string = row['pgn']
        moves = extract_moves_from_pgn(pgn_string)
        chess_moves.append(moves)

        # Add the new column of opening_moves name to the DataFrame
    chess_opening_database['opening_moves'] = chess_moves

    # chess_database_opening_df.head()
    return chess_opening_database

def determine_opening(moves, openings):
    best_match = "Unknown Opening"
    max_matched_moves = 0

    for index, row in openings.iterrows():
        opening_name = row['name']
        opening_moves = row['opening_moves']
        matched_moves = 0

        # Count matched moves
        for i in range(min(len(moves), len(opening_moves))):
            if moves[i] == opening_moves[i]:
                matched_moves += 1
            else:
                break

        # Update best match if this opening matches more moves
        if matched_moves > max_matched_moves:
            best_match = opening_name
            max_matched_moves = matched_moves

    return best_match

def get_game_result(row, username):

    if row['winner_user'] == username:
        return 'win'
    elif row['winner_user'] != username and row['winner_user'] != 'draw':
        return 'lose'
    else:
        return 'draw'

# How to use it in your code:
# result_games['termination_category'] = result_games.apply(categorize_termination_robust, username=chesscom_user_id, axis=1)


def categorize_termination(row, username):
    termination_str = str(row['termination']).lower()
    game_result = "" # This will be 'win', 'lose', or 'draw' for the user

    # 1. Determine the game result for the user
    # Check for draw first, as it's typically explicit
    if 'draw' in termination_str or 'drawn' in termination_str  or row['winner_side'] == 'draw' :
        game_result = 'draw'
    elif row['winner_user'] == username:
        game_result = 'win'
    else:
        game_result = 'lose' # If not win or draw, user lost

    # 2. Extract the last word from the termination string
    # Remove common trailing punctuation like periods
    clean_termination_str = re.sub(r'[^\w\s]', '', termination_str) # Remove non-word, non-space characters
    words = clean_termination_str.split()
    
    termination_reason_word = 'unknown' # Default reason

    if words:
        # Get the last word. Handle specific cases where the last word is not descriptive.
        last_word = words[-1]

        if last_word == 'resignation' or last_word == 'checkmate' or last_word == 'abandoned' or last_word == 'repetition' or last_word == 'stalemate':
            termination_reason_word = last_word
        elif last_word == 'time':
            termination_reason_word = 'time_win'
        elif last_word == 'agreement': # 'game drawn by agreement'
             termination_reason_word = last_word
        elif last_word == 'accepted': # 'draw by agreement accepted'
             termination_reason_word = 'agreement'
        elif last_word == 'move': # e.g., 'drawn by insufficient material by move'
            if len(words) >= 3 and words[-3] == 'insufficient':
                 termination_reason_word = 'insufficient material'
            else:
                 termination_reason_word = 'other' # Generic for less common 'move' terminations
        elif last_word == 'rule': # e.g., 'drawn by 50-move rule'
             if len(words) >= 3 and words[-3] == '50-move':
                 termination_reason_word = '50-move rule'
             else:
                 termination_reason_word = 'other'
        elif last_word == 'material': # e.g., 'checkmate by insufficient material'
            termination_reason_word = 'insufficient material'
        elif last_word == 'rule': # e.g., 'checkmate by three-fold repetition rule'
            if len(words) >= 3 and words[-3] == 'three-fold':
                termination_reason_word = 'three-fold repetition'
            else:
                termination_reason_word = 'other'
        else:
            termination_reason_word = 'other' # Catch-all for less common last words

    # 3. Combine game result and termination reason
    return f"{game_result} by {termination_reason_word}"


def simplify_opening(opening_name):
    if not isinstance(opening_name, str):
        return "Other"
    parts = opening_name.split(':')
    if len(parts) > 1:
        return parts[0].strip()  # Return only the main opening
    else:
        return opening_name.strip()

def calculate_duration(start_time_str, end_time_str):
    start = datetime.strptime(start_time_str, '%H:%M:%S')
    end = datetime.strptime(end_time_str, '%H:%M:%S')
    return (end - start).total_seconds()

def download_data_chess_com(chesscom_user_id = 'jay_fh', start_month = 3, last_month = 4,year = 2025):

    # for player in player_list:
    error_counter = 0
    list_games = []
    list_game_id   = []
    list_game_created_date = []
    list_white_username = []
    list_black_username = []
    list_result = []
    list_start_time = []
    list_end_time = []
    list_termination = []
    list_white_elo = []
    list_black_elo = []
    list_time_control = []
    list_pgn = []
    list_eco_opening = []
    list_game_class = []

    list_failed_game_url = []
    
    # year = '2025'
    # start_month = 3
    # last_month = 4
    url = f'https://api.chess.com/pub/player/{chesscom_user_id}/games/{year}/'
    for k in range(start_month,last_month+1):
        try:
            x = '0' + str(k) if (k<10) else str(k)
            game_url = url+x
            # print (game_url)
            response = httpx.get(game_url, timeout=None)
            headers = {
                "User-Agent": "MyChessApp/1.0 - chesscomlysis"  # <-- IMPORTANT
            }
            response = httpx.get(game_url, headers=headers, timeout=30)
            
            # print (str(response.status_code))
            if response.status_code == 200:
                try:
                    df_raw = pd.DataFrame(response.json())
            
                    df_games = df_raw.copy()
                    # display (df_games)
                    for index, row in df_games.iterrows():
                        try:
                            json_raw = pd.json_normalize(row)    

                            list_pgn.append(json_raw['pgn'][0])
                            list_game_class.append(json_raw['time_class'][0])

                            df = parse_chess_data_to_dataframe(json_raw['pgn'][0])                        
                            list_game_id.append(df['Link'][0])
                            list_eco_opening.append(df['ECO'])
                            list_game_created_date.append(df['Date'][0])
                            list_white_username.append(df['White'][0])
                            list_black_username.append(df['Black'][0])
                            list_result.append(df['Result'][0])
                            list_start_time.append(df['StartTime'][0])
                            list_end_time.append(df['EndTime'][0])
                            list_termination.append(df['Termination'][0])
                            list_white_elo.append(df['WhiteElo'][0])
                            list_black_elo.append(df['BlackElo'][0])
                            list_time_control.append(df['TimeControl'][0])
                        except:
                            # print(f"Error during normalize: {response.status_code}")
                            # print (game_url + '  - normalize')
                            list_failed_game_url.append(game_url)
                            # time.sleep(1)
                            
                except ValueError:
                    list_failed_game_url.append(game_url)
                    # time.sleep(0.5)
            else:
                error_counter = error_counter + 1
                list_failed_game_url.append(game_url)
                
        except ValueError:
            print("Failed to get request")
            list_failed_game_url.append(game_url)

    df_chess_game_raw = []
    df_chess_game_raw = pd.DataFrame(
        {
            'pgn' : list_pgn,
            'game_class' : list_game_class,
            'game_id' : list_game_id,
            'created_date' : list_game_created_date,
            'opening' : list_eco_opening,
            'white_username' : list_white_username,
            'black_username' : list_black_username,
            
            'game_result' : list_result,
            'start_time' : list_start_time,
            'end_time' : list_end_time,
            'termination' : list_termination,
            'white_elo' : list_white_elo,
            'black_elo' : list_black_elo,
            'TimeControl': list_time_control
        })

    df_list_failed_url =  pd.DataFrame(list_failed_game_url)
    # df_list_failed_url.to_csv("failed_url_5.csv")
    df_chess_game_all = df_chess_game_raw.copy()
    return df_chess_game_all


def pre_analysis_chessgame(row, chesscom_user_id= 'jay_fh',user_timezone=timezone('Asia/Jakarta')):
   
    row['created_date'] = pd.to_datetime(row['created_date'], format='%Y.%m.%d')
    # row['created_week'] = row['created_date'].dt.isocalendar().week
    row['created_week'] = row['created_date'].dt.strftime('%Y-W%U')  # %U is week number (Sunday as first day)
    row['start_time_dtformat'] = pd.to_datetime(row['start_time'])
    row['end_time_dtformat'] = pd.to_datetime(row['end_time'])
    row['created_datetime'] = pd.to_datetime(row['created_date'].astype(str) + ' ' + row['start_time'],format='%Y-%m-%d %H:%M:%S')
    
    if row['created_datetime'].dt.tz is None:
        # Naive datetime → localize first
        row['local_datetime'] = (
            row['created_datetime']
            .dt.tz_localize('UTC')
            .dt.tz_convert(user_timezone)
        )
    else:
        # Already tz-aware → just convert
        row['local_datetime'] = row['created_datetime'].dt.tz_convert(user_timezone)

    
    row['white_elo'] = row['white_elo'].astype(int)
    row['black_elo'] = row['black_elo'].astype(int)
    row['winner_side'] = 'empty'
    row['winner_side']  = np.where(row['game_result'] == '1-0', 'white', row['winner_side'] )
    row['winner_side']  = np.where(row['game_result'] == '0-1', 'black', row['winner_side'] )
    row['winner_side']  = np.where(row['game_result'] == '1/2-1/2', 'draw', row['winner_side'] )

    row['winner_user'] = 'empty'
    row['winner_user']  = np.where(row['winner_side'] == 'white', row['white_username'], row['winner_user'] )
    row['winner_user']  = np.where(row['winner_side'] == 'black', row['black_username'], row['winner_user'] )
    row['winner_user']  = np.where(row['winner_side'] == 'draw', 'draw',row['winner_user']  )

    row['opponent_elo'] = row.apply(lambda row: row['black_elo'] if row['white_username'] == chesscom_user_id else row['white_elo'], axis=1)
    row['your_elo'] = row.apply(lambda row: row['white_elo'] if row['white_username'] == chesscom_user_id else row['black_elo'], axis=1)

    row['elo_category'] = row.apply(lambda row: elo_category(row['opponent_elo'], row['your_elo']), axis=1)

    chess_database_opening_df = opening_database()
    for index, column in row.iterrows():
        pgn_string = column['pgn']
        moves = extract_moves_from_pgn(pgn_string)
        opening_name = determine_opening(moves, chess_database_opening_df)
        row.at[index, 'opening_name'] = opening_name
        row['winner_side'] = 'empty'

       # Calculate game duration in minutes
    row['duration_minutes'] = (row['end_time_dtformat'] - row['start_time_dtformat']).dt.total_seconds() / 60
    
    # Extract hour of day from end_time
    # row['hour_of_day'] = row['end_time_dtformat'].dt.hour
    row['hour_of_day'] = row['local_datetime'].dt.hour

    # Categorize time of day
    # row['time_of_day'] = pd.cut(row['hour_of_day'],bins=[6, 12, 18, 24],labels=['Night', 'Morning', 'Afternoon', 'Evening'],right=False)
    row['time_of_day'] = pd.cut(row['hour_of_day'],bins=[-1, 7, 18, 23],labels=['evening 6pm - 6am', 'morning 6am - 6 pm', 'evening 6pm - 6am'],right=False,ordered=False)
    

    # Filter games where the user played
    user_games = row[(row['white_username'] == chesscom_user_id) | (row['black_username'] == chesscom_user_id)].copy()

    # Determine the user's side and opponent for each game
    row['user_side'] = row.apply(lambda row: 'white' if row['white_username'] == chesscom_user_id else 'black', axis=1)
    row['opponent_username'] = row.apply(lambda row: row['black_username'] if row['white_username'] == chesscom_user_id else row['white_username'], axis=1)
    # Function to determine game result
    
    row['game_result_user'] = row.apply(lambda row, user=chesscom_user_id: get_game_result(row, user), axis=1)
    row['termination_category'] = row.apply(categorize_termination, username=chesscom_user_id, axis=1) 
    row['termination_type'] = row['termination_category'].apply(lambda x: x.split()[-1])
    row['simplified_opening'] = row['opening_name'].apply(simplify_opening)
    # 
 
    # df['created_date'].astype(str)

    row = row[row['TimeControl'] != '1/86400']
    return row

def analyze_chess_games(df, chesscom_user_id='jay_fh'):
    """
    Analyzes historical chess games for a given username, 
    separating black and white results,
    and refining the analysis of termination reasons and openings.

    Args:
        df (pd.DataFrame): DataFrame containing the game data.
        username (str): The username to analyze.

    Returns:
        dict: A dictionary containing the analysis results.
    """

    analysis = {
        'white': {},
        'black': {}
    }
   
    # print ('')
    # print (user_games['game_result_user'].value_counts())
    # print ('')

    # 1. Win rate per time control (split by color)
    for color in ['white', 'black']:
        color_games = user_games[user_games['user_side'] == color]
        if not color_games.empty:
            time_control_analysis = color_games.groupby('TimeControl')['game_result_user'].value_counts().unstack(fill_value=0)
            time_control_analysis['total'] = time_control_analysis.sum(axis=1)
            time_control_analysis['win_rate'] = time_control_analysis.get('win', 0) / time_control_analysis['total']
            analysis[color]['win_rate_per_time_control'] = time_control_analysis.to_dict('index')
        else:
            analysis[color]['win_rate_per_time_control'] = {}

    #2. Win/Loss rate per termination reason (split by color)
    for color in ['white', 'black']:
        color_games = user_games[user_games['user_side'] == color]
        analysis[color]['termination_category'] = {}
        for game_result in ['draw', 'lose', 'win']:
            # print (game_result)
            result_games = color_games[color_games['game_result_user'] == game_result]
            if not color_games.empty:
                result_games['termination_category'] = result_games.apply(categorize_termination, username=chesscom_user_id, axis=1)  
                # print (result_games['termination_category']) 
                termination_analysis = result_games.groupby('game_result_user')['termination_category'].value_counts().unstack(fill_value=0)
                termination_analysis['total'] = termination_analysis.sum(axis=1)
                # -----------
                fig = px.line(
                df_time_control,
                x='time_control',
                y='win_rate',
                color='color',  # This creates separate lines for 'White' and 'Black'
                markers=True,   # Adds markers to the data points for clarity
                labels={
                "time_control": "Time Control",
                "win_rate": "Win Rate",
                "color": "Player Color"
                },
                title="White vs. Black Win Rate"
                )


                # --------------



                analysis[color]['termination_category'][game_result] = termination_analysis.to_dict('index')

                
            else:
                # print ('color_games empty')
                analysis[game_result]['termination_category'] = {}
                print ('failed')


            
    # 3. Win rate in day and night (split by color)
    
    for color in ['white', 'black']:
        color_games = user_games[user_games['user_side'] == color]
        if not color_games.empty:
            hour_of_day_analysis = color_games.groupby('time_of_day')['game_result_user'].value_counts().unstack(fill_value=0)
            hour_of_day_analysis['total'] = hour_of_day_analysis.sum(axis=1)
            hour_of_day_analysis['win_rate'] = hour_of_day_analysis.get('win', 0) / hour_of_day_analysis['total']
            hour_of_day_analysis['loss_rate'] = hour_of_day_analysis.get('loss', 0) / hour_of_day_analysis['total']
            analysis[color]['win_loss_rate_per_hour_of_day'] = hour_of_day_analysis.to_dict('index')
            
        else:
             analysis[color]['hour_of_day'] = {}
            
    # 4. Win rate per opening (less granular, split by color)
    user_games['simplified_opening'] = user_games['opening_name'].apply(simplify_opening)

    for color in ['white', 'black']:
        color_games = user_games[user_games['user_side'] == color]
        if not color_games.empty:
            opening_analysis = color_games.groupby('simplified_opening')['game_result_user'].value_counts().unstack(fill_value=0)
            opening_analysis['total'] = opening_analysis.sum(axis=1)
            opening_analysis['win_rate'] = opening_analysis.get('win', 0) / opening_analysis['total']

            # Filter out openings with very few plays (e.g., < 5)
            opening_analysis = opening_analysis[opening_analysis['total'] >= 5]
            analysis[color]['win_rate_per_opening'] = opening_analysis.to_dict('index')
        else:
            analysis[color]['win_rate_per_opening'] = {}

    # 5. Game duration in winning/losing condition (split by color)
    

    for color in ['white', 'black']:
        color_games = user_games[user_games['user_side'] == color].copy()
        if not color_games.empty:
            winning_games = color_games[color_games['game_result_user'] == 'win'].copy()
            losing_games = color_games[color_games['game_result_user'] == 'lose'].copy()

            if not winning_games.empty:
                winning_games['duration_seconds'] = winning_games.apply(lambda row: calculate_duration(row['start_time'], row['end_time']), axis=1)
                analysis[color]['average_duration_winning_games_seconds'] = winning_games['duration_seconds'].mean()
            else:
                analysis[color]['average_duration_winning_games_seconds'] = None

            if not losing_games.empty:
                losing_games['duration_seconds'] = losing_games.apply(lambda row: calculate_duration(row['start_time'], row['end_time']), axis=1)
                analysis[color]['average_duration_losing_games_seconds'] = losing_games['duration_seconds'].mean()
            else:
                analysis[color]['average_duration_losing_games_seconds'] = None
        else:
            analysis[color]['average_duration_winning_games_seconds'] = None
            analysis[color]['average_duration_losing_games_seconds'] = None

    # 6. Win rate against similar, weaker, stronger elo (split by color)
    for color in ['white', 'black']:
        color_games = user_games[user_games['user_side'] == color]
        if not color_games.empty:
            elo_category_analysis = color_games.groupby('elo_category')['game_result_user'].value_counts().unstack(fill_value=0)
            elo_category_analysis['total'] = elo_category_analysis.sum(axis=1)
            elo_category_analysis['win_rate'] = elo_category_analysis.get('win', 0) / elo_category_analysis['total']
            analysis[color]['win_rate_per_elo_category'] = elo_category_analysis.to_dict('index')
        else:
            analysis[color]['win_rate_per_elo_category'] = {}
    
    return analysis

def prepare_game_data(df):
    """Convert dataframe to list of dictionaries with proper serialization"""
    games_data = []
    
    for _, row in df.iterrows():
        # Convert any pandas/numpy types to native Python types
        utc_time = pd.to_datetime(row['created_date'])
        local_tz = pytz.timezone('Asia/Jakarta')
        local_time = utc_time.tz_localize('UTC').astimezone(local_tz)
        
        game_data = {
            "pgn": str(row['pgn']),  # Explicit string conversion
            "game_class": str(row['game_class']),
            "game_id": str(row['game_id']),
            "created_date": local_time.isoformat(),
            "opening": str(row['opening']),
            "white_username": str(row['white_username']),
            "black_username": str(row['black_username']),
            "game_result": str(row['game_result']),
            "start_time": str(row['start_time']),
            "end_time": str(row['end_time']),
            "termination": str(row['termination']),
            "white_elo": int(row['white_elo']),  # Convert to native int
            "black_elo": int(row['black_elo']),
            "time_control": str(row['TimeControl']),
            "inserted_at": datetime.now().isoformat()
        }
        games_data.append(game_data)
    
    return games_data

def insert_to_supabase(games_data, batch_size=50):
    """Insert games to Supabase in batches"""
    # Read Supabase credentials from environment — do NOT hardcode secrets in source.
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_KEY must be set in environment variables to insert to Supabase."
        )

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    total_inserted = 0
    
    for i in range(0, len(games_data), batch_size):
        batch = games_data[i:i + batch_size]
        try:
            response = supabase.table("chess_games").insert(batch).execute()
            # response.data can be None or a list depending on the client/version
            if hasattr(response, 'data') and response.data:
                try:
                    total_inserted += len(response.data)
                except Exception:
                    total_inserted += 0
            # print(f"Inserted batch {i//batch_size + 1}: {len(response.data)} games")
        except Exception as e:
            print(f"Error inserting batch {i//batch_size + 1}: {str(e)}")
    
    return total_inserted

def user_download_data(df):
    df_games = df.copy()
      

# Load the data from the text file


# print ('test')
