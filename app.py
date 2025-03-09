from flask import Flask, request, jsonify, redirect
import os
import sqlite3
from slack_sdk import WebClient
from slack_sdk.oauth import AuthorizeUrlGenerator, OAuthStateUtils
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv
import requests
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

load_dotenv()

app = Flask(_name_)
app.secret_key = os.getenv("FLASK_SECRET_KEY")

SLACK_CLIENT_ID = os.getenv("SLACK_CLIENT_ID")
SLACK_CLIENT_SECRET = os.getenv("SLACK_CLIENT_SECRET")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
SLACK_REDIRECT_URI = os.getenv("SLACK_REDIRECT_URI", "https://adoption-aid-printers-changing.trycloudflare.com/oauth/callback")
client = WebClient(token=SLACK_BOT_TOKEN)

authorize_url_generator = AuthorizeUrlGenerator(client_id=SLACK_CLIENT_ID, scopes=["chat:write", "commands"], redirect_uri=SLACK_REDIRECT_URI)

# Initialize database
def init_db():
    conn = sqlite3.connect("achievements.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS leaderboard (
            user_id TEXT PRIMARY KEY,
            achievements INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

init_db()

def update_leaderboard(user):
    conn = sqlite3.connect("achievements.db")
    cursor = conn.cursor()
    cursor.execute("SELECT achievements FROM leaderboard WHERE user_id = ?", (user,))
    row = cursor.fetchone()
    if row:
        cursor.execute("UPDATE leaderboard SET achievements = achievements + 1 WHERE user_id = ?", (user,))
    else:
        cursor.execute("INSERT INTO leaderboard (user_id, achievements) VALUES (?, 1)", (user,))
    conn.commit()
    conn.close()

@app.route("/")
def home():
    return "Slack Bot is Running! 🚀"

import re

@app.route("/event", methods=["POST"])
def event_handler():
    data = request.json
    print("Event received:", data)

    if "challenge" in data:
        return jsonify({"challenge": data["challenge"]})

    if "event" in data:
        event = data["event"]
        if event.get("type") == "message" and "subtype" not in event:
            user = event.get("user")
            text = event.get("text", "").lower()
            channel = event.get("channel")

            # Extract all mentioned users (Slack uses <@USERID> format)
            mentioned_users = re.findall(r"<@(\w+)>", text)
            if user not in mentioned_users:
                mentioned_users.append(user)  # Ensure the sender is included

            # Keywords for classification
            class_1_keywords = ["i completed", "i achieved", "i led","i published","i won", "goal reached", "mission accomplished","I completed","I did","I won","I led","I published"]
            class_2_keywords = ["i built", "i coded", "i created", "i managed", "i deployed", "i implemented","I built", "I coded", "I created", "I managed", "I deployed", "I implemented"]
            class_3_keywords = ["i debugged", "i fixed", "i resolved", "task completed", "i did", "i solved","i helped","I debugged", "I fixed", "I resolved", "task completed", "I did", "I solved","I helped"]

            # Recognition filters
            negation_keywords = ["not", "n't", "no", "fail", "failed", "unable", "didn't", "wasn't", "couldn't"]
            uncertain_keywords = ["try", "tried", "attempt", "attempted", "working on", "in progress", "trying"]

            # Task classification
            task_class = None
            if any(keyword in text for keyword in class_1_keywords):
                task_class = "Class 1"
            elif any(keyword in text for keyword in class_2_keywords):
                task_class = "Class 2"
            elif any(keyword in text for keyword in class_3_keywords):
                task_class = "Class 3"

            # Check if the statement is a confirmed achievement
            if task_class and not any(neg_word in text.split() for neg_word in negation_keywords) and not any(uncertain_word in text.split() for uncertain_word in uncertain_keywords):
                for u in mentioned_users:
                    update_leaderboard(u)  # Update leaderboard for all involved

                # Construct the congratulatory message
                if len(mentioned_users) > 1:
                    user_mentions = " and ".join([f"<@{u}>" for u in mentioned_users])
                    response_text = f"🔥 Amazing teamwork, {user_mentions}! You've all earned a {task_class} achievement! 🚀"
                else:
                    response_text = f"🎉 Congratulations <@{user}>! You've earned a {task_class} achievement! Keep climbing that leaderboard! 🏆"

                try:
                    client.chat_postMessage(channel=channel, text=response_text)
                except SlackApiError as e:
                    print(f"Error sending Slack message: {e.response['error']}")

    return jsonify({"status": "Event processed"})


def fetch_leaderboard():
    conn = sqlite3.connect("achievements.db")
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, achievements FROM leaderboard ORDER BY achievements DESC LIMIT 10")
    results = cursor.fetchall()
    conn.close()
    return results

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
dash_app = dash.Dash(_name_, server=app, routes_pathname_prefix="/dashboard/", external_stylesheets=external_stylesheets)

dash_app.layout = html.Div([
    html.H1("Leaderboard", style={"textAlign": "center"}),
    dcc.Graph(id="leaderboard-graph"),
    dcc.Interval(
        id="interval-component",
        interval=5000,  # Refresh every 5 seconds
        n_intervals=0
    )
])

@dash_app.callback(
    dash.dependencies.Output("leaderboard-graph", "figure"),
    [dash.dependencies.Input("interval-component", "n_intervals")]
)
def update_graph(n):
    results = fetch_leaderboard()
    if not results:
        return px.bar(title="No Data Available")
    
    df = pd.DataFrame(results, columns=["User", "Achievements"])
    fig = px.bar(df, x="User", y="Achievements", title="Leaderboard", color="Achievements", text_auto=True)
    fig.update_layout(template="plotly_dark", xaxis_title="User ID", yaxis_title="Achievements")
    return fig

@app.route("/oauth/callback")
def oauth_callback():
    code = request.args.get("code")
    if not code:
        return "Error: Authorization code missing", 400
    
    try:
        response = requests.post("https://slack.com/api/oauth.v2.access", data={
            "client_id": SLACK_CLIENT_ID,
            "client_secret": SLACK_CLIENT_SECRET,
            "code": code,
            "redirect_uri": SLACK_REDIRECT_URI
        }).json()
        
        if not response.get("ok"):
            raise SlackApiError("OAuth error", response)
        
        access_token = response.get("access_token")
        return jsonify({"message": "OAuth successful!", "access_token": access_token})
    except SlackApiError as e:
        return jsonify({"error": f"Error exchanging code for token: {e.response['error']}"}), 400

@app.route("/install")
def install():
    url = authorize_url_generator.generate(state=OAuthStateUtils.generate_random())
    return redirect(url)

if _name_ == "_main_":
    app.run(host='0.0.0.0', port=5000, debug=True)