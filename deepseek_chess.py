import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date, datetime, timedelta
import warnings
import pytz
import streamlit.components.v1 as components
from tzlocal import get_localzone  # pip install tzlocal
import time
import chess
import chess.pgn
import io
import pprint
import openai
from openai import OpenAI
import requests
warnings.filterwarnings('ignore')
import chess_com_download
from streamlit_javascript import st_javascript
from pytz import timezone

from chess_com_download import test_function_test
from chess_com_download import opening_database
from chess_com_download import download_data_chess_com
from chess_com_download import pre_analysis_chessgame
from chess_com_download import analyze_chess_games
from chess_com_download import prepare_game_data
from chess_com_download import insert_to_supabase


# from supabase import create_client, Client
from datetime import datetime
import pytz


import streamlit.runtime.app_session as app_session

def styled_chart(fig, caption=""):
    # Improve chart appearance for dark theme
    fig.update_traces(line=dict(width=3))  # Thicker lines
    fig.update_layout(
        title_font=dict(size=22, color="white"),
        font=dict(size=14, color="white"),
        margin=dict(l=60, r=60, t=70, b=60),
        plot_bgcolor="black",
        paper_bgcolor="black",
        xaxis=dict(
            showgrid=True, gridcolor="gray",
            zeroline=True, zerolinecolor="white", zerolinewidth=2,
            linecolor="white", linewidth=2, ticks="outside", tickcolor="white"
        ),
        yaxis=dict(
            showgrid=True, gridcolor="gray",
            zeroline=True, zerolinecolor="white", zerolinewidth=2,
            linecolor="white", linewidth=2, ticks="outside", tickcolor="white"
        ),
        legend=dict(title="", orientation="h", y=-0.25, font=dict(color="white"))
    )

    # Container with padding + border
    st.markdown(
        """
        <div style="
            border:2px solid #444;
            border-radius:12px;
            padding:20px;
            margin:25px 0;
            background-color:#111;">
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True)
    if caption:
        st.caption(caption)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------

if not hasattr(app_session.AppSession, "_scriptrunner"):
    app_session.AppSession._scriptrunner = None


# Set page config
st.set_page_config(
    page_title="Chess.com Game Analyzer",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    h1 {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 4px;
    }
    .stTextInput>div>div>input {
        border: 1px solid #4CAF50;
    }
    .pie-chart-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 2rem;
    }
    .pie-chart {
        width: 30%;
    }
    </style>
    """, unsafe_allow_html=True)

# App title and description
st.title("♟️ Chess.com Game Analyzer - Last Month Data")
st.markdown("""
Analyze your Chess.com game history to understand your strengths and weaknesses. 
Enter your Chess.com username below to see your performance metrics.
""")

## Timezone Function


# Try to detect local timezone and map it to a valid pytz zone
try:
    import tzlocal  # pip install tzlocal
    local_tz_name = tzlocal.get_localzone_name()  # e.g., 'Asia/Jakarta'
except:
    local_tz_name = "UTC"  # fallback

# If session_state doesn't have timezone, set it to detected
if "user_timezone" not in st.session_state:
    st.session_state.user_timezone = local_tz_name

# Get current time in user's timezone
user_timezone = pytz.timezone(st.session_state.user_timezone)

# Initialize session state for form submission
if 'analyze_clicked' not in st.session_state:
    st.session_state.analyze_clicked = False

# Modify your date range filter to use local time

# Sidebar for user input
with st.sidebar:
    st.header("Input Parameters")
    chesscom_user_id = st.text_input("Chess.com Username", value="")

    # Static date range
    start_date = date(2025, 7, 1)
    end_date = date(2025, 7, 31)

    st.markdown(
        f"**Date Range:** {start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}"
    )
    
    # Add analyze button
    analyze_button = st.button("Go!", key="analyze_button")

# Set analyze clicked state when button is pressed
if analyze_button and chesscom_user_id:
    st.session_state.analyze_clicked = True
elif analyze_button and not chesscom_user_id:
    st.warning("Please enter a Chess.com username")
    st.session_state.analyze_clicked = False


# Only show analysis if user has clicked the analyze button
if st.session_state.analyze_clicked and chesscom_user_id:
    # Load data

    # df_chess_game = download_data_chess_com('jay_fh',3,6,2025)
    df_chess_game = download_data_chess_com(chesscom_user_id,7,8,2025)
    
    # games_data = prepare_game_data(df_chess_game)
    # total_inserted = insert_to_supabase(games_data)
    # print(f"\nSuccessfully inserted {total_inserted} out of {len(df_chess_game)} games")
    df = pre_analysis_chessgame(df_chess_game, chesscom_user_id,user_timezone)

    timecontrol_counts = df['TimeControl'].value_counts()
    # Get the TimeControl values that appear more than 20 times
    popular_timecontrols = timecontrol_counts[timecontrol_counts > 50].index
    df = df[df['TimeControl'].isin(popular_timecontrols)].copy()

    if df.empty:
        st.warning("No games found for this user. Please check the usersname and try again.")
        st.stop()

    # Main content
    st.header(f"Game Analysis for {chesscom_user_id}")


    def create_stats_tab_overview(df,username):
        st.subheader("📊 Player Overview")
        # """Create a tab with numerical statistics overview"""
        # 1st section - overall stats
        # 2nd section - overall stats black and white
        # 3rd section - Elo rating over time
        # 4th section - termination types
        # 5th section - Chess Openingg Performance
        # 6th section - Stronger Weaker Opponent Analysis
        # 7th section - Time of Day Performance

        # 1st section - overall stats
        col1, col2 = st.columns(2)
        col1.metric("Games Played", df["game_id"].count())

        overall_wr = df[df['game_result_user'] == 'win']["game_id"].count() / df["game_id"].count() * 100
        col2.metric("Overall Win Rate", f"{overall_wr:.1f}%")

        # 2nd section - overall stats black and white

        # col1, col2 ,col3 ,col4  = st.columns(4)
        col1, col2 = st.columns(2)

        white_game = df[df['user_side'] == 'white'].copy()
        col1.metric("Games Played", white_game["game_id"].count())

        overall_wr_white = white_game[white_game['game_result_user'] == 'win']["game_id"].count() / white_game["game_id"].count() * 100
        col2.metric("Overall Win Rate - White", f"{overall_wr_white:.1f}%")

        black_game = df[df['user_side'] == 'black']
        col1.metric("Games Played", black_game["game_id"].count())

        overall_wr_black = black_game[black_game['game_result_user'] == 'win']["game_id"].count() / black_game["game_id"].count() * 100
        col2.metric("Overall Win Rate - Black", f"{overall_wr_black:.1f}%")

        st.markdown("---")
        # 3rd section - Elo rating over time
        st.markdown("# 📈 Elo Trend Analysis")
        if 'created_datetime' in df.columns and 'your_elo' in df.columns:
            fig = px.line(df.sort_values('created_datetime'), 
                        x='created_datetime', y='your_elo',
                        color='TimeControl',
                        title='Elo Rating Over Time by Time Control')
            styled_chart(fig, "This chart shows how your win rate changes across different openings.")
            # st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Elo history data not available")

        st.markdown("---")
        # 4th section - termination types
        st.markdown("# 📈 Termination Type Analysis")
        col_termination_1, col_termination_2 = st.columns(2)
        win_game = df[df['game_result_user'] == 'win']
        

        with col_termination_1:
            if not win_game.empty:
                termination_counts_win = win_game['termination_type'].value_counts().reset_index()
                termination_counts_win.columns = ['Termination Type', 'Count']
                fig_draw = px.pie(
                    termination_counts_win,
                    values='Count',
                    names='Termination Type',
                    title=f'wining by method'
                )
                st.plotly_chart(fig_draw, use_container_width=True)
            else:
                st.write(f"Error")

        lose_game = df[df['game_result_user'] == 'lose']
        with col_termination_2:
            if not lose_game.empty:
                termination_counts_lose = lose_game['termination_type'].value_counts().reset_index()
                termination_counts_lose.columns = ['Termination Type', 'Count']
                fig_draw = px.pie(
                    termination_counts_lose,
                    values='Count',
                    names='Termination Type',
                    title=f'losing by method'
                )
                st.plotly_chart(fig_draw, use_container_width=True)
            else:
                st.write(f"Error")


        st.markdown("---")
        # 5th section - Chess Openingg Performance

        st.markdown("# 📈 Chess Opening Trend Analysis")
        top5_opening = df['simplified_opening'].value_counts().head(3).index
        df_opening = df[df['simplified_opening'].isin(top5_opening)].copy()
        df_opening["is_win"] = (df_opening["game_result_user"] == "win").astype(int)
        df_opening["year_month_week"] = df_opening["created_datetime"].dt.strftime("%Y-%m") + "-W" + df_opening["created_datetime"].dt.isocalendar().week.astype(str)

        df_summary_opening = df_opening.groupby(['year_month_week', 'simplified_opening']).agg(
            win_rate=('is_win', 'mean'),
            total_games=('is_win', 'count')
        ).reset_index()


        fig_opening = px.line(
            df_summary_opening,
            x="year_month_week",
            y="win_rate",
            color="simplified_opening",
            markers=True,
            title="Win Rate Over Time per Opening"
        )
        fig.update_layout(
            title=dict(
            text="Win Rate Over Time per Opening",
            font=dict(size=28),   # <-- change size here
            x=0.5,                # center title (0=left, 0.5=center, 1=right)
            xanchor="center"
        )
        )


        # st.plotly_chart(fig_opening, use_container_width=True)
        styled_chart(fig_opening, "This chart shows how your win rate changes across different openings.")

        st.markdown("---")
        # 6th section - Stronger Weaker Opponent Analysis

        st.markdown("# 📈 Chess elo comparison Trend Analysis")
        df_comparison_elo = df.copy()
        df_comparison_elo["is_win"] = (df_comparison_elo["game_result_user"] == "win").astype(int)
        df_comparison_elo["year_month_week"] = df_comparison_elo["created_datetime"].dt.strftime("%Y-%m") + "-W" + df_comparison_elo["created_datetime"].dt.isocalendar().week.astype(str)

        df_summary_comparison_elo = df_comparison_elo.groupby(['year_month_week', 'elo_category']).agg(
            win_rate=('is_win', 'mean'),
            total_games=('is_win', 'count')
        ).reset_index()


        fig_opponent_elo = px.line(
            df_summary_comparison_elo,
            x="year_month_week",
            y="win_rate",
            color="elo_category",
            markers=True,
            title="Win Rate Per Opponent ELO Strength"
        )
        styled_chart(fig_opponent_elo, "This chart shows how your win rate changes across different openings.")
        # st.plotly_chart(fig_opening, use_container_width=True)


        st.markdown("---")
        # 7th section - Time of Day Performance

        st.markdown("# 📈 Winning time of day summary")
        df_time_of_day = df.copy()
        df_time_of_day["is_win"] = (df_time_of_day["game_result_user"] == "win").astype(int)
        df_time_of_day["year_month_week"] = df_time_of_day["created_datetime"].dt.strftime("%Y-%m") + "-W" + df_time_of_day["created_datetime"].dt.isocalendar().week.astype(str)

        df_summary_timeday = df_time_of_day.groupby(['year_month_week', 'time_of_day']).agg(
            win_rate=('is_win', 'mean'),
            total_games=('is_win', 'count')
        ).reset_index()


        fig_time_of_day = px.line(
            df_summary_timeday,
            x="year_month_week",
            y="win_rate",
            color="time_of_day",
            markers=True,
            title="Win Rate Per Time of Day"
        )
        styled_chart(fig_time_of_day, "This chart shows how your win rate changes across different openings.")
        # st.plotly_chart(fig_opening, use_container_width=True)



        st.markdown("---")
        #8th section - Personalized ChatGPT Advice


        st.header("8. AI-Powered Chess Improvement Advice")

        openai.api_key = st.secrets["OPENAI_API_KEY"]
        
        #Prepare insights from sections 1-7
        context = f"""
        Color stats White: {overall_wr_white}
        Color stats black: {overall_wr_black}
        Termination types win: {termination_counts_win.to_dict()}
        Termination types lose: {termination_counts_lose.to_dict()}
        chess opening: {df_summary_opening.to_dict()}
        Opponent ELO analysis: {df_summary_comparison_elo.to_dict()}
        When player play analysis: {df_summary_timeday.to_dict()}
        """

        prompt = f"""
        You are a chess coach. Based only on the data below (no external knowledge), give the user 3-5 actionable
        suggestions to improve their chess. Focus on weaknesses and opportunities.
        And specifically for opening, please provide the video tutorial opening from youtube

        DATA:
        {context}
        """

        try:
            client = OpenAI(api_key=openai.api_key)
            response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful chess coach analyzing player statistics."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
        )

            advice = response.choices[0].message.content
            st.subheader("Your Personalized Advice:")
            st.write(advice)
        except Exception as e:
            st.error(f"Error generating advice: {e}")

        # with st.spinner("Analyzing your games..."):
            
        



    def create_stats_tab_detail(df,username):
        
        """Enhanced stats overview tab with DeepSeek-style analytics"""
        # """Create a tab with numerical statistics overview"""
        # 1 Color
        # 2 Termination reason
        # 3 Time control
        # 4 opening
        # 5 Weaker
        st.subheader(f"♟️ Deep Chess Analysis for {username}")

        st.markdown("") #space
        # Section 2: Win Rate by Color (7-column layout)
        st.markdown("## ♖ vs ♜ Win Rates")
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        
        with col1:
            st.markdown("**Color**")
            st.write("White")
            st.write("Black")
        
        with col2:
            st.markdown("**Games**")
            white_games = len(df[df['user_side'] == 'white'])
            black_games = len(df[df['user_side'] == 'black'])
            st.write(white_games)
            st.write(black_games)
        
        with col3:
            st.markdown("**Wins**")
            white_wins = len(df[(df['user_side'] == 'white') & (df['game_result_user'] == 'win')])
            black_wins = len(df[(df['user_side'] == 'black') & (df['game_result_user'] == 'win')])
            st.write(white_wins)
            st.write(black_wins)
        
        with col4:
            st.markdown("**Losses**")
            white_losses = len(df[(df['user_side'] == 'white') & (df['game_result_user'] == 'lose')])
            black_losses = len(df[(df['user_side'] == 'black') & (df['game_result_user'] == 'lose')])
            st.write(white_losses)
            st.write(black_losses)
        
        with col5:
            st.markdown("**Draws**")
            white_draws = len(df[(df['user_side'] == 'white') & (df['game_result_user'] == 'draw')])
            black_draws = len(df[(df['user_side'] == 'black') & (df['game_result_user'] == 'draw')])
            st.write(white_draws if 'draw' in df['game_result_user'].unique() else 0)
            st.write(black_draws if 'draw' in df['game_result_user'].unique() else 0)
        
        with col6:
            st.markdown("**Win Rate**")
            white_rate = (white_wins/white_games*100) if white_games > 0 else 0
            black_rate = (black_wins/black_games*100) if black_games > 0 else 0
            st.write(f"{white_rate:.1f}%")
            st.write(f"{black_rate:.1f}%")
        
        with col7:
            st.markdown("**Perf Diff**")
            white_diff = (white_wins - white_losses)
            black_diff = (black_wins - black_losses)
            st.write(f"+{white_diff}" if white_diff > 0 else white_diff)
            st.write(f"+{black_diff}" if black_diff > 0 else black_diff)
        
        st.markdown("") #space
        # Section 3: Win Rate by Time Control
        st.markdown("## ⏱ Time Control Performance")
        st.markdown("") #space

        if 'TimeControl' in df.columns:
            time_controls = df['TimeControl'].unique()
            for tc in sorted(time_controls):
                tc_df = df[df['TimeControl'] == tc]
                wins = len(tc_df[tc_df['game_result_user'] == 'win'])
                losses = len(tc_df[tc_df['game_result_user'] == 'lose'])
                rate = (wins/(wins+losses)*100) if (wins+losses) > 0 else 0
                
                cols = st.columns(7)
                cols[0].write(f"**{tc}**")
                cols[1].write(f"{wins + losses}")
                cols[2].write(f"{wins}")
                cols[3].write(f"{losses}")
                cols[4].write(f"{len(tc_df[tc_df['game_result_user'] == 'draw'])}" if 'draw' in tc_df['game_result_user'].unique() else "0")
                cols[5].write(f"{rate:.1f}%")
                cols[6].write(f"{wins-losses:+d}")
        

        st.markdown("") #space

        # Section 4: Chess Openings Performance
        st.markdown("## 🏰 Opening Performance (Top 10)")
        st.markdown("") #space

        if 'simplified_opening' in df.columns:
            top_openings = df['simplified_opening'].value_counts().head(5).index
            for opening in top_openings:
                op_df = df[df['simplified_opening'] == opening]
                wins = len(op_df[op_df['game_result_user'] == 'win'])
                losses = len(op_df[op_df['game_result_user'] == 'lose'])
                rate = (wins/(wins+losses)*100) if (wins+losses) > 0 else 0
                
                cols = st.columns(7)
                cols[0].write(f"**{opening}**")
                cols[1].write(f"{wins + losses}")
                cols[2].write(f"{wins}")
                cols[3].write(f"{losses}")
                cols[4].write(f"{len(op_df[op_df['game_result_user'] == 'draw'])}" if 'draw' in op_df['game_result_user'].unique() else "0")
                cols[5].write(f"{rate:.1f}%")
                cols[6].write(f"{wins-losses:+d}")
        
        # Section 5: Elo Matchup Performance
        st.markdown("") #space
        st.markdown("## 🥊 Opponent Strength Analysis")
        st.markdown("") #space

        if 'elo_category' in df.columns:
            for strength in ['Weaker', 'Similar', 'Stronger']:
                str_df = df[df['elo_category'] == strength]
                wins = len(str_df[str_df['game_result_user'] == 'win'])
                losses = len(str_df[str_df['game_result_user'] == 'lose'])
                rate = (wins/(wins+losses)*100) if (wins+losses) > 0 else 0
                
                cols = st.columns(7)
                cols[0].write(f"**{strength}**")
                cols[1].write(f"{wins + losses}")
                cols[2].write(f"{wins}")
                cols[3].write(f"{losses}")
                cols[4].write(f"{len(str_df[str_df['game_result_user'] == 'draw'])}" if 'draw' in str_df['game_result_user'].unique() else "0")
                cols[5].write(f"{rate:.1f}%")
                cols[6].write(f"{wins-losses:+d}")
        
        # Section 6: Time of Day Performance
        st.markdown("## 🌞🌙 Time of Day Performance")
        if 'time_of_day' in df.columns:
            for tod in ['morning', 'evening']:
                tod_df = df[df['time_of_day'] == tod]
                if len(tod_df) > 0:
                    wins = len(tod_df[tod_df['game_result_user'] == 'win'])
                    losses = len(tod_df[tod_df['game_result_user'] == 'lose'])
                    rate = (wins/(wins+losses)*100) if (wins+losses) > 0 else 0
                    
                    cols = st.columns(7)
                    cols[0].write(f"**{tod}**")
                    cols[1].write(f"{wins + losses}")
                    cols[2].write(f"{wins}")
                    cols[3].write(f"{losses}")
                    cols[4].write(f"{len(tod_df[tod_df['game_result_user'] == 'draw'])}" if 'draw' in tod_df['game_result_user'].unique() else "0")
                    cols[5].write(f"{rate:.1f}%")
                    cols[6].write(f"{wins-losses:+d}")
        
        # Section 7: Improvement Recommendations
        st.markdown("## 🧠 Chess Improvement Insights")
        st.info("""
        **DeepSeek Analysis Suggestions:**
        - Practice your weakest openings (those with <45% win rate)
        - Focus on time controls where your win rate is lowest
        - Review games against stronger opponents to identify improvement areas
        - Consider playing more during your most successful time of day
        """)

    def create_stats_tab_overview_color(color):
        color_df = df[df['user_side'] == color]
        
        if color_df.empty:
            st.warning(f"No {color} games found in the selected filters.")
            return
        
        # 1. Win rate per time control
        st.subheader(f"Win Rate by Time Control ({color})")
        time_control_analysis = color_df.groupby(['created_week', 'TimeControl'])['game_result_user'].value_counts().unstack(fill_value=0)
        time_control_analysis['total'] = time_control_analysis.sum(axis=1)
        time_control_analysis['win_rate'] = time_control_analysis.get('win', 0) / time_control_analysis['total']
        time_control_analysis = time_control_analysis.reset_index()
        
        fig1 = px.line(
            time_control_analysis, 
            x='created_week', 
            y='win_rate', 
            color='TimeControl',
            title=f'Win Rate Over Time by Time Control ({color})',
            labels={'created_week': 'Date', 'win_rate': 'Win Rate'}
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # 2. Termination type - Now 3 pie charts (win/lose/draw)
        st.subheader(f"Game Termination Types ({color})")
        
        # Create columns for the 3 pie charts
        col1, col2, col3 = st.columns(3)
        
        with col1:
            win_df = color_df[color_df['game_result_user'] == 'win']
            if not win_df.empty:
                termination_counts = win_df['termination_type'].value_counts().reset_index()
                termination_counts.columns = ['Termination Type', 'Count']
                fig_win = px.pie(
                    termination_counts,
                    values='Count',
                    names='Termination Type',
                    title=f'How Wins Ended ({color})'
                )
                st.plotly_chart(fig_win, use_container_width=True)
            else:
                st.write(f"No wins found for {color}")
        
        with col2:
            lose_df = color_df[color_df['game_result_user'] == 'lose']
            if not lose_df.empty:
                termination_counts = lose_df['termination_type'].value_counts().reset_index()
                termination_counts.columns = ['Termination Type', 'Count']
                fig_lose = px.pie(
                    termination_counts,
                    values='Count',
                    names='Termination Type',
                    title=f'How Losses Ended ({color})'
                )
                st.plotly_chart(fig_lose, use_container_width=True)
            else:
                st.write(f"No losses found for {color}")
        
        with col3:
            draw_df = color_df[color_df['game_result_user'] == 'draw']
            if not draw_df.empty:
                termination_counts = draw_df['termination_type'].value_counts().reset_index()
                termination_counts.columns = ['Termination Type', 'Count']
                fig_draw = px.pie(
                    termination_counts,
                    values='Count',
                    names='Termination Type',
                    title=f'How Draws Ended ({color})'
                )
                st.plotly_chart(fig_draw, use_container_width=True)
            else:
                st.write(f"No draws found for {color}")
        
        # 3. Win rate by time of day
        st.subheader(f"Win Rate by Time of Day ({color})")
        hour_analysis = color_df.groupby(['created_week', 'time_of_day'])['game_result_user'].value_counts().unstack(fill_value=0)
        hour_analysis['total'] = hour_analysis.sum(axis=1)
        hour_analysis['win_rate'] = hour_analysis.get('win', 0) / hour_analysis['total']
        hour_analysis = hour_analysis.reset_index()
        
        fig3 = px.line(
            hour_analysis,
            x='created_week',
            y='win_rate',
            color='time_of_day',
            title=f'Win Rate by Time of Day ({color})',
            labels={'created_week': 'Date', 'win_rate': 'Win Rate'}
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # 4. Win rate by opening
        st.subheader(f"Win Rate by Opening ({color})")
        opening_analysis = color_df.groupby(['created_week', 'simplified_opening'])['game_result_user'].value_counts().unstack(fill_value=0)
        opening_analysis['total'] = opening_analysis.sum(axis=1)
        opening_analysis['win_rate'] = opening_analysis.get('win', 0) / opening_analysis['total']
        opening_analysis = opening_analysis[opening_analysis['total'] >= 3].reset_index()  # Filter rare openings
        
        fig4 = px.line(
            opening_analysis,
            x='created_week',
            y='win_rate',
            color='simplified_opening',
            title=f'Win Rate by Opening ({color})',
            labels={'created_week': 'Date', 'win_rate': 'Win Rate'}
        )
        st.plotly_chart(fig4, use_container_width=True)
        
        # 5. Win rate by opponent strength
        st.subheader(f"Win Rate by Opponent Strength ({color})")
        elo_analysis = color_df.groupby(['created_week', 'elo_category'])['game_result_user'].value_counts().unstack(fill_value=0)
        elo_analysis['total'] = elo_analysis.sum(axis=1)
        elo_analysis['win_rate'] = elo_analysis.get('win', 0) / elo_analysis['total']
        elo_analysis = elo_analysis.reset_index()
        
        fig5 = px.line(
            elo_analysis,
            x='created_week',
            y='win_rate',
            color='elo_category',
            title=f'Win Rate by Opponent Strength ({color})',
            labels={'created_week': 'Date', 'win_rate': 'Win Rate'}
        )
        st.plotly_chart(fig5, use_container_width=True)
    

    # tab1, tab2, tab3,tab4 = st.tabs(["Stats Overview","Stats Detail","White Performance", "Black Performance"])
    # tab1, tab2 = st.tabs(["Stats Overview","Stats Detail"])
    tab1, = st.tabs(["Overview"])
    


    with tab1:
        create_stats_tab_overview(df, chesscom_user_id)


    # with tab2:
    #     create_stats_tab_detail(df, chesscom_user_id)
        
    # with tab3:
    #     create_stats_tab_overview_color('white')

    # with tab4:
    #     create_stats_tab_overview_color('black')


        
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: gray;">
            <p>Chess.com Game Analyzer - Data fetched from Chess.com API</p>
            <p>Note: This is a demo using sample data. Connect to Chess.com API for real data.</p>
        </div>
        """, unsafe_allow_html=True)
else:
    # Show instructions when no analysis has been run yet
    st.info("Please enter your Chess.com username and click 'Go!' to begin analysis")

# Add a button to reset the analysis
if st.session_state.analyze_clicked:
    if st.button("Reset Analysis"):
        st.session_state.analyze_clicked = False
        st.rerun()