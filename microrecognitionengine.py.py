import os
import io
import csv
import re
import psycopg2
import requests
import time
import logging
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
import openai
from transformers import pipeline
import logging
import matplotlib
import random
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Set up logging
logger = logging.getLogger(__name__)

# Load the Hugging Face zero-shot classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", framework="tf")# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("achievement_bot")

# Load environment variables from .env file
load_dotenv()

# Slack API Tokens
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")

# Database Credentials
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

# Initialize Slack Client & App
client = WebClient(token=SLACK_BOT_TOKEN)
app = App(token=SLACK_BOT_TOKEN)

# Cache for usernames to reduce API calls
username_cache = {}

# Try to load spaCy and NLTK, but continue even if they fail
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    spacy_available = True
    logger.info("✅ spaCy loaded successfully")
except ImportError:
    logger.warning("⚠ spaCy not available. Using fallback text processing.")
    spacy_available = False
    nlp = None

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    
    # Use quiet download with error handling
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("vader_lexicon", quiet=True)
        sia = SentimentIntensityAnalyzer()
        nltk_available = True
        logger.info("✅ NLTK loaded successfully")
    except Exception as e:
        logger.warning(f"⚠ NLTK download failed: {e}")
        nltk_available = False
        sia = None
except ImportError:
    logger.warning("⚠ NLTK not available. Using fallback sentiment analysis.")
    nltk_available = False
    sia = None

def initialize_database():
    """Create database tables if they don't exist."""
    conn = connect_db()
    if not conn:
        logger.error("❌ Failed to initialize database - connection failed")
        return False
    
    try:
        with conn.cursor() as cur:
            # Check if achievements table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'achievements'
                );
            """)
            
            table_exists = cur.fetchone()[0]
            
            if not table_exists:
                logger.info("🔧 Creating achievements table...")
                # Create achievements table with timestamp
                cur.execute("""
                    CREATE TABLE achievements (
                        id SERIAL PRIMARY KEY,
                        "user" VARCHAR(50) NOT NULL,
                        message TEXT NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.commit()
                logger.info("✅ Database initialized successfully")
            else:
                logger.info("✅ Database already initialized")
                
            return True
    except Exception as e:
        logger.error(f"❌ Database initialization error: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False
    finally:
        conn.close()


def connect_db():
    """Connect to PostgreSQL Database with improved retry and error handling."""
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT
            )
            # Set autocommit to True to avoid transaction issues
            # Each query will be in its own transaction unless explicitly stated
            conn.autocommit = True
            return conn
        except Exception as e:
            retry_count += 1
            logger.error(f"❌ Database connection failed (attempt {retry_count}/{max_retries}): {e}")
            if retry_count < max_retries:
                time.sleep(2)  # Wait before retrying
    
    return None


def get_commits_from_github(repo_owner, repo_name):
    """Fetch commits from a public GitHub repository."""
    url = f"https://api.github.com/link/continued...."
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad responses (4xx, 5xx)
        commits = response.json()  # Parse the response as JSON
        
        # Extract commit data from the response
        commit_data = {}
        for commit in commits:
            author = commit['commit']['author']['name']
            if author in commit_data:
                commit_data[author] += 1  # Increment commit count
            else:
                commit_data[author] = 1  # First commit from this user
        
        return commit_data
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching commits: {e}")
        return {}


def save_commits_to_db(commit_data):
    """Save commit counts to the database."""
    conn = connect_db()
    if not conn:
        print("⚠ Database connection failed!")
        return
    
    try:
        with conn.cursor() as cur:
            for user, commit_count in commit_data.items():
                # You can calculate the points here or set a fixed value
                total_points = commit_count * 2  # Example: 2 points per commit, adjust this logic as needed.
                
                # Save or update the commit count and points in the database
                cur.execute("""
                    INSERT INTO user_commitss (user_name, commit_count, total_points) 
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_name) 
                    DO UPDATE SET commit_count = user_commitss.commit_count + EXCLUDED.commit_count, 
                                  total_points = user_commitss.total_points + EXCLUDED.total_points;
                """, (user, commit_count, total_points))
            
            conn.commit()  # Commit the changes to the database
            print(f"Commits saved to database: {len(commit_data)} users.")
    
    except Exception as e:
        print(f"❌ Error saving commit data: {e}")
    finally:
        conn.close()
import matplotlib.pyplot as plt
import io

@app.command("/gitvisualize")
def visualize_command(ack, respond, command):
    """Handle /gitvisualize command to generate a leaderboard pie chart."""
    ack()
    # Define repository info
    repo_owner = "rails"
    repo_name = "rails"
    # Fetch commits from GitHub
    commit_data = get_commits_from_github(repo_owner, repo_name)
    
    if not commit_data:
        respond("⚠ Error fetching commits from GitHub.")
        return
    # Save commit data to the database
    save_commits_to_db(commit_data)
    # Sort users by commit count (descending)
    sorted_committers = sorted(commit_data.items(), key=lambda x: x[1], reverse=True)
    
    # Identify top contributors and group the rest as "Others"
    top_n = 7  # Show top 7 contributors individually
    if len(sorted_committers) > top_n:
        top_users = sorted_committers[:top_n]
        other_count = sum(count for _, count in sorted_committers[top_n:])
        if other_count > 0:
            users = [user for user, _ in top_users] + ["Others"]
            commit_counts = [count for _, count in top_users] + [other_count]
        else:
            users = [user for user, _ in top_users]
            commit_counts = [count for _, count in top_users]
    else:
        users = [user for user, _ in sorted_committers]
        commit_counts = [count for _, count in sorted_committers]
    
    if not users:
        respond("No commit data available to visualize.")
        return
    
    # Generate a pie chart with hover info
    plt.figure(figsize=(8, 8))
    
    # Calculate percentages for annotations
    total_commits = sum(commit_counts)
    percentages = [(count/total_commits)*100 for count in commit_counts]
    
    # Add labels with percentage and count
    labels = [f"{user}\n{count} commits\n({pct:.1f}%)" for user, count, pct in zip(users, commit_counts, percentages)]
    
    # Create the pie chart with exploded first slice for emphasis
    explode = [0.1] + [0] * (len(users) - 1)  # Explode first slice
    wedges, texts, autotexts = plt.pie(
        commit_counts, 
        explode=explode,
        labels=None,  # No labels directly on pie
        autopct='',  # No percentages directly on pie
        shadow=True, 
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
    )
    
    # Add a legend with detailed information
    plt.legend(
        wedges, 
        labels,
        title="Contributors",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    
    plt.title(f"GitHub Commit Distribution - {repo_owner}/{repo_name}")
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
    
    # Save the chart to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
    buffer.seek(0)
    plt.close()
    
    # Post to the channel where the command was triggered (no DM needed)
    channel_id = command["channel_id"]
    
    try:
        # Upload the image to Slack
        response = client.files_upload_v2(
            channel=channel_id,
            file=buffer,
            filename="git_pie_chart.png",
            title="GitHub Commits Distribution",
            initial_comment="📊 Here's your GitHub commit distribution pie chart!"
        )
        if response.get("ok"):
            respond("📊 Chart posted to this channel!")
        else:
            respond("⚠ Error uploading chart. Try again!")
    except Exception as e:
        respond(f"⚠ Error: {str(e)}. Please check your bot token and permissions.")


def assess_achievement_difficulty(achievement_text):
    """
    Use Hugging Face's BART model to assess the difficulty of an achievement and assign points.
    Returns tuple of (difficulty_level, points)
    """
    try:
        # Define the possible difficulty categories
        labels = ["high", "medium", "low"]

        # Classify the achievement text
        result = classifier(achievement_text, labels)

        # Get the highest scoring category
        difficulty = result["labels"][0].lower()

        # Map difficulties to points
        points_map = {"high": 10, "medium": 5, "low": 2}

        # Default to 'low' if response is unexpected
        if difficulty not in points_map:
            logger.warning(f"⚠ Unexpected difficulty assessment: {difficulty}, defaulting to 'low'")
            difficulty = "low"

        points = points_map[difficulty]
        logger.info(f"🏆 Achievement assessed as {difficulty} difficulty: {points} points")

        return difficulty, points

    except Exception as e:
        logger.error(f"❌ Error assessing achievement difficulty: {e}")
        return "low", 2  # Default to low difficulty if something goes wrongdef save_achievement(user, message):
    """Save an achievement message to the database with difficulty assessment."""
    conn = connect_db()
    if conn:
        try:
            # Assess difficulty and get points
            difficulty, points = assess_achievement_difficulty(message)
            
            with conn.cursor() as cur:
                cur.execute(
                    'INSERT INTO achievements ("user", message, difficulty, points) VALUES (%s, %s, %s, %s);', 
                    (user, message, difficulty, points)
                )
            logger.info(f"✅ Achievement saved: {user} - {message} - {difficulty} ({points} pts)")
            
            # Return the points so we can include them in the response
            return True, points, difficulty
        except Exception as e:
            logger.error(f"❌ Error saving achievement: {e}")
            try:
                if not conn.autocommit:
                    conn.rollback()
            except Exception:
                pass
        finally:
            conn.close()
    return False, 0, "none"

def save_achievement(user, message):
    """Save an achievement message to the database with difficulty assessment."""
    conn = connect_db()
    if conn:
        try:
            # Assess difficulty and get points
            difficulty, points = assess_achievement_difficulty(message)
            
            with conn.cursor() as cur:
                cur.execute(
                    'INSERT INTO achievements ("user", message, difficulty, points) VALUES (%s, %s, %s, %s);', 
                    (user, message, difficulty, points)
                )
            logger.info(f"✅ Achievement saved: {user} - {message} - {difficulty} ({points} pts)")
            
            # Return the points so we can include them in the response
            return True, points, difficulty
        except Exception as e:
            logger.error(f"❌ Error saving achievement: {e}")
            try:
                if not conn.autocommit:
                    conn.rollback()
            except Exception:
                pass
        finally:
            conn.close()
    return False, 0, "none"

def is_achievement(text):
    """
    Enhanced achievement detection using keywords and basic text analysis.
    Falls back to simpler methods if NLP libraries aren't available.
    """
    # Primary achievement keywords with weights
    achievement_keywords = {
        "completed": 2, "achieved": 2, "finished": 2, "done": 1.5, 
        "won": 2, "success": 2, "reached": 1.5, "milestone": 2,
        "accomplished": 2, "delivered": 1.5, "launched": 1.5,
        "solved": 2, "fixed": 1.5, "implemented": 1.5, "shipped": 2,
        "built": 1.5, "created": 1.5, "passed": 1.5, "succeeded": 2
    }
    
    # Calculate keyword score
    keyword_score = 0
    text_lower = text.lower()
    for keyword, weight in achievement_keywords.items():
        if keyword in text_lower:
            keyword_score += weight
    
    # Get sentiment score if NLTK is available
    sentiment_score = 0
    if nltk_available and sia:
        try:
            sentiment_result = sia.polarity_scores(text)
            sentiment_score = sentiment_result["compound"]
        except Exception as e:
            logger.warning(f"⚠ Sentiment analysis failed: {e}")
    
    # Check for personal pronouns manually
    personal_pronouns = ["i", "we", "my", "our"]
    has_personal_pronoun = any(pronoun in text_lower.split() for pronoun in personal_pronouns)
    
    # Check for spaCy enhanced analysis
    spacy_score = 0
    if spacy_available and nlp:
        try:
            doc = nlp(text)
            # Check for past tense verbs
            has_past_tense = any(token.tag_ in ["VBD", "VBN"] for token in doc)
            spacy_score = 0.5 if has_past_tense else 0
        except Exception as e:
            logger.warning(f"⚠ spaCy analysis failed: {e}")
    
    # Final achievement score calculation
    achievement_score = (
        keyword_score * 0.6 +  # Keyword weight - increased to compensate for possible missing NLP
        sentiment_score * 0.2 + # Sentiment weight - reduced due to possible missing NLTK
        (0.5 if has_personal_pronoun else 0) + # Personal pronoun bonus
        spacy_score  # spaCy score (if available)
    )
    
    # Debug info
    logger.info(f"🔍 Achievement analysis: score={achievement_score:.2f}, keywords={keyword_score}")
    
    # Consider it an achievement if the score is above threshold
    return achievement_score > 1.2

def get_username(user_id):
    try:
        response = client.users_info(user=user_id)
        if response["ok"]:
            return response["user"]["real_name"]
        else:
            return f"Unknown ({user_id})"
    except Exception as e:
        logger.error(f"❌ Error fetching username for {user_id}: {e}")
        return f"Unknown ({user_id})"

@app.command("/leaderboard")
def leaderboard_command(ack, say):
    """Handle /leaderboard command to display top users."""
    ack()
    say(get_leaderboard())

def get_leaderboard():
    """Fetch top achievers from the database and return leaderboard text."""
    conn = connect_db()
    if not conn:
        return "⚠ Database connection failed!"
    
    try:
        with conn.cursor() as cur:
            # Use a safer query that doesn't rely on created_at if it might not exist
            try:
                cur.execute('''
                    SELECT "user", COUNT(*) as achievement_count, 
                           MAX(created_at) as last_achievement 
                    FROM achievements 
                    GROUP BY "user" 
                    ORDER BY achievement_count DESC, last_achievement DESC 
                    LIMIT 5;
                ''')
            except Exception as e:
                # Make sure to rollback after exception
                conn.rollback()
                # Fallback query if created_at column doesn't exist
                cur.execute('''
                    SELECT "user", COUNT(*) as achievement_count
                    FROM achievements 
                    GROUP BY "user" 
                    ORDER BY achievement_count DESC
                    LIMIT 5;
                ''')
            results = cur.fetchall()
            
            # Format the leaderboard text
            if not results:
                return "No achievements recorded yet! Be the first to share your achievements! 🏆"
            
            leaderboard_text = "🏆 Achievement Leaderboard 🏆\n\n"
            
            for i, row in enumerate(results):
                user_id = row[0]
                count = row[1]
                username = get_username(user_id)
                
                # Add medal emoji for top 3
                if i == 0:
                    medal = "🥇"
                elif i == 1:
                    medal = "🥈"
                elif i == 2:
                    medal = "🥉"
                else:
                    medal = "🏅"
                
                # Format date if available
                if len(row) >= 3 and row[2]:
                    last_date = row[2].strftime("%Y-%m-%d")
                    leaderboard_text += f"{medal} {i+1}. {username}: {count} achievements (last: {last_date})\n"
                else:
                    leaderboard_text += f"{medal} {i+1}. {username}: {count} achievements\n"
            
            return leaderboard_text
            
    except Exception as e:
        logger.error(f"❌ Error fetching leaderboard: {e}")
        # Ensure transaction is rolled back
        try:
            conn.rollback()
        except Exception:
            pass
        return "⚠ Error fetching leaderboard data!"
    finally:
        conn.close()

@app.command("/visualize")
def visualize_command(ack, respond, command):
    """Handle /visualize command to generate a leaderboard chart."""
    ack()
    user_id = command["user_id"]

    conn = connect_db()
    if not conn:
        respond("⚠ Database connection failed!")
        return
    
    with conn.cursor() as cur:
        cur.execute('SELECT "user", COUNT() FROM achievements GROUP BY "user" ORDER BY COUNT() DESC LIMIT 5;')
        results = cur.fetchall()
    conn.close()

    if not results:
        respond("No achievements recorded yet! 📊")
        return
    
    users = [get_username(row[0]) for row in results]  # Convert IDs to real names
    counts = [row[1] for row in results]

    plt.figure(figsize=(6, 4))
    plt.bar(users, counts, color="blue")
    plt.xlabel("Users")
    plt.ylabel("Achievements")
    plt.title("Top Achievers Leaderboard")
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()
    
    # Open a DM channel with the user
    dm_response = client.conversations_open(users=[user_id])
    channel_id = dm_response["channel"]["id"]
    
    response = client.files_upload_v2(
        channel=channel_id,
        file=buffer,
        filename="leaderboard.png",
        title="Leaderboard Chart",
        initial_comment="📊 Here’s your leaderboard visualization!"
    )
    
    if response.get("ok"):
        respond("📊 Chart sent via DM!")
    else:
        respond("⚠ Error uploading chart. Try again!")

@app.command("/export")
def export_command(ack, respond, command):
    """Handle /export command to generate and send achievements CSV."""
    ack()
    user_id = command["user_id"]

    conn = connect_db()
    if not conn:
        respond("⚠ Database connection failed!")
        return
    
    try:
        with conn.cursor() as cur:
            # Try with created_at field first
            try:
                cur.execute('''
                    SELECT a."user", a.message, a.created_at 
                    FROM achievements a
                    ORDER BY a.created_at DESC;
                ''')
            except Exception:
                # Fallback if created_at doesn't exist
                cur.execute('''
                    SELECT a."user", a.message 
                    FROM achievements a
                ''')
            results = cur.fetchall()
    except Exception as e:
        logger.error(f"❌ Error exporting achievements: {e}")
        respond("⚠ Error exporting achievements data!")
        return
    finally:
        conn.close()

    if not results:
        respond("No achievements recorded yet! 📜")
        return
    
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    
    # Check if we have timestamp data
    if len(results[0]) >= 3:
        writer.writerow(["User", "Achievement", "Date"])
        for row in results:
            writer.writerow([
                get_username(row[0]), 
                row[1], 
                row[2].strftime("%Y-%m-%d %H:%M:%S") if row[2] else "N/A"
            ])
    else:
        writer.writerow(["User", "Achievement"])
        for row in results:
            writer.writerow([get_username(row[0]), row[1]])
            
    buffer.seek(0)
    
    try:
        dm_response = client.conversations_open(users=user_id)
        channel_id = dm_response["channel"]["id"]
        
        response = client.files_upload_v2(
            channel=channel_id,
            content=buffer.getvalue(),
            filename="achievements.csv",
            title="Achievements Export",
            initial_comment="📜 Here's your exported achievements!"
        )
        
        if response.get("ok"):
            respond("📜 CSV file sent via DM!")
        else:
            respond("⚠ Error uploading achievements file. Try again!")
    except Exception as e:
        logger.error(f"❌ Error in file upload: {e}")
        respond("⚠ Error sending export. Please try again later.")

def extract_mentions(text):
    """
    Extract all mentioned user IDs from a message.
    
    In Slack, user mentions appear as <@USER_ID> in the raw message text.
    """
    mentioned_users = []
    
    # First, try using regex to extract user IDs
    import re
    mention_pattern = re.compile(r'<@([A-Z0-9]+)>')
    matches = mention_pattern.findall(text)
    
    if matches:
        mentioned_users.extend(matches)
        logger.info(f"🔍 Found {len(matches)} user mentions via regex")
    
    # Optional: Use a fallback approach to look for common patterns if no regex matches
    if not mentioned_users:
        words = text.split()
        for word in words:
            # Look for the <@ID> pattern manually
            if word.startswith('<@') and word.endswith('>'):
                user_id = word[2:-1]  # Remove the <@ and >
                mentioned_users.append(user_id)
    
    return mentioned_users

def analyze_content(text,user_id):
    """
    Simple content analysis that falls back to basic methods if NLP tools aren't available.
    """
    # Skip empty messages
    if not text or len(text.strip()) < 5:
        return None
    
    # Basic negative keyword detection
    negative_keywords = ["hate", "stupid", "idiot", "kill", "die", "fuck", "shit"]
    if any(word in text.lower() for word in negative_keywords):
        return "🚨 This message contains potentially inappropriate language. Let's keep it respectful!"
    
    # NLTK sentiment analysis if available
    if nltk_available and sia:
        try:
            sentiment = sia.polarity_scores(text)
            if sentiment["compound"] < -0.5:
                return "⚠ This message seems negative. Let's keep our communications positive!"
        except Exception as e:
            logger.warning(f"⚠ NLTK sentiment analysis failed: {e}")
    
    # Try Hugging Face if API key is available
    if HUGGING_FACE_API_KEY:
        hf_result = analyze_sentiment_hf(text,user_id)
        if hf_result:
            return hf_result
            
    return None


def analyze_sentiment_hf(text, user_id):
    """
    Uses Hugging Face API for sentiment analysis.
    """
    if not HUGGING_FACE_API_KEY or len(text.strip()) < 5:
        return None

    url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}

    payload = {
        "inputs": text,
        "parameters": {"candidate_labels": [
            "achievement", "positive", "neutral", "negative",
            "threatening", "abusive", "toxic",
            "identity hate", "obscene", "insulting"
        ]}
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=5)

        if response.status_code == 200:
            predictions = response.json()
            if "labels" in predictions and "scores" in predictions:
                sentiment = predictions["labels"][0]
                score = predictions["scores"][0]

                logger.info(f"🔍 Sentiment Analysis: {sentiment} ({score:.2f})")

                if sentiment == "achievement" and score > 0.7:
                    return "🎉 Great job! Keep up the positive momentum!"

                if sentiment in ["sad", "frustrated", "disappointed", "negative"] and score > 0.7:
                    motivational_msg = get_motivational_message()
                    send_private_message(user_id, motivational_msg)  # Send privately
                    return None

                # Moderate toxicity
                if sentiment in ["toxic", "insulting", "negative"] and score > 0.7:
                    return "🚨 Please ensure your message is respectful and constructive."

                elif sentiment == "threatening" and score > 0.7:
                    return "⚠ This message seems threatening. Let's ensure a safe environment."

                elif sentiment in ["abusive", "obscene"] and score > 0.7:
                    return "⛔ Inappropriate language is not allowed. Let's keep it respectful."

                elif sentiment == "identity hate" and score > 0.6:
                    return "🚫 Hate speech is not tolerated. Please be inclusive and kind."

                return None  # No action required if message is neutral or low-risk

        else:
            logger.warning(f"❌ Hugging Face API request failed: {response.status_code}")

    except requests.exceptions.RequestException as e:
        logger.warning(f"❌ API request failed: {e}")

    return None

def send_private_message(user_id, message):
    """
    Sends a private message to a user on Slack.
    """
    try:
        client.chat_postMessage(channel=user_id, text=message)
        logger.info(f"✅ Private motivation sent to {user_id}: {message}")
    except Exception as e:
        logger.error(f"❌ Failed to send private message: {e}")

def get_motivational_message():
    """
    Returns a random motivational message.
    """
    messages = [
        "Challenges are part of the journey to success. Stay strong!",
        "Every step forward, no matter how small, is progress!",
        "Setbacks are opportunities for growth and improvement.",
        "It's okay to feel overwhelmed sometimes. Take a deep breath.",
        "You're capable of overcoming this challenge. Keep pushing forward!",
        "Persistence is key. Every obstacle builds resilience.",
        "Success is a marathon, not a sprint. Stay committed!",
        "Keep learning and improving. Growth comes from effort.",
        "Every challenge presents an opportunity for growth!",
        "Success is about dedication over time. Keep going!"
    ]
    return random.choice(messages)



@app.event("message")
def handle_message(event, say):
    """Enhanced message handler with improved achievement detection and mention parsing."""
    # Ignore bot messages
    if event.get("subtype") == "bot_message":
        return
        
    user = event.get("user")
    text = event.get("text", "")
    channel = event.get("channel")
    
    # Skip empty messages
    if not text or len(text.strip()) < 3:
        return
    
    # Run content analysis
    content_result = analyze_content(text,user)
    
    # Handle moderation flags
    if content_result and content_result != "achievement":
        say(thread_ts=event.get("ts"), text=content_result)
        return
    
    # Check if this is an achievement post
    if is_achievement(text) or content_result == "achievement":
        username = get_username(user)
        
        # Extract all mentioned users
        mentioned_users = extract_mentions(text)
        
        # Save achievement for the original poster with difficulty assessment
        success, points, difficulty = save_achievement(user, text)  # <-- UPDATED LINE
        
        if success:  # <-- UPDATED CONDITION
            # Also save for each mentioned user if there are any
            for mentioned_user in mentioned_users:
                if mentioned_user != user:  # Avoid duplicate entries for the same user
                    # Use same difficulty for mentioned users, but with lower points
                    mentioned_points = max(1, points // 2)  # Half points, minimum 1
                    save_achievement(mentioned_user, f"Mentioned in achievement: {text}")
            
            # Customize response based on difficulty
            difficulty_emoji = {
                "high": "🔥",
                "medium": "✨",
                "low": "👏", 
                "none": "🎉"
            }
            
            # Get emoji based on difficulty
            reaction = difficulty_emoji.get(difficulty, "🎉")
            
            # React to the message
            try:
                client.reactions_add(
                    channel=channel,
                    timestamp=event.get("ts"),
                    name=reaction.replace(":", "")  # Remove colons if present
                )
            except Exception as e:
                logger.error(f"❌ Error adding reaction: {e}")
            
            # Create acknowledgment text with points information
            acknowledgment = f"{reaction} Achievement unlocked! Great job, {username}"
            if mentioned_users:
                mentioned_names = [get_username(uid) for uid in mentioned_users if uid != user]
                if mentioned_names:
                    acknowledgment += f" and {', '.join(mentioned_names)}"
            
            # Add difficulty and points info
            acknowledgment += f"! You earned {points} points for this {difficulty} achievement! {reaction}"
            
            # Respond in thread
            say(
                thread_ts=event.get("ts"),
                text=acknowledgment
            )
            
            # Log the achievement with difficulty and points
            logger.info(f"🏆 Achievement detected for {username} ({difficulty}, {points} pts) and {len(mentioned_users)} mentioned users: {text}")
@app.command("/achievements")
def achievements_command(ack, say, command):
    """Handle /achievements command to list recent achievements."""
    ack()
    
    conn = connect_db()
    if not conn:
        say("⚠ Database connection failed!")
        return
    
    try:
        with conn.cursor() as cur:
            # Try with created_at field first
            try:
                cur.execute('''
                    SELECT a."user", a.message, a.created_at 
                    FROM achievements a
                    ORDER BY a.created_at DESC
                    LIMIT 5;
                ''')
            except Exception:
                # Fallback if created_at doesn't exist
                cur.execute('''
                    SELECT a."user", a.message 
                    FROM achievements a
                    LIMIT 5;
                ''')
            results = cur.fetchall()
    except Exception as e:
        logger.error(f"❌ Error fetching achievements: {e}")
        say("⚠ Error fetching achievement data!")
        return
    finally:
        conn.close()

    if results:
        achievements_text = "🏆 Recent Achievements:\n"
        for row in results:
            username = get_username(row[0])
            achievement = row[1]
            # Include date if available
            if len(row) >= 3 and row[2]:
                date_str = row[2].strftime("%Y-%m-%d")
                achievements_text += f"• {username} ({date_str}): {achievement}\n"
            else:
                achievements_text += f"• {username}: {achievement}\n"
        say(achievements_text)
    else:
        say("No achievements recorded yet! Be the first to share yours!")

@app.event("app_mention")
def handle_app_mentions(body, say):
    """Handle when the bot is mentioned in a channel."""
    text = body["event"].get("text", "")
    
    # Extract the actual message (removing the mention part)
    mention_text = text.split(">", 1)[1].strip() if ">" in text else ""
    
    if "help" in mention_text.lower():
        help_text = """
Achievement Bot Help
I track and celebrate team achievements! Here's how to use me:

- Simply post your achievements in channels where I'm added
- Use /leaderboard to see top achievers
- Use /achievements to see recent achievements
- Use /export to get a CSV export of all achievements
- Mention me with "help" to see this message
        """
        say(help_text)
    elif any(word in mention_text.lower() for word in ["hi", "hello", "hey"]):
        say("Hello! I'm Achievement Bot. I track team achievements! Mention me with 'help' to learn more.")
    else:
        say("👋 Need help with achievements? Mention me with 'help' to see what I can do!")

def initialize_bot():
    """Initialize the Slack bot with better error handling."""
    logger.info("🚀 Starting Achievement Bot")
    
    # Verify internet connectivity before trying to start
    try:
        # Simple connectivity check to a reliable service
        requests.get("https://www.google.com", timeout=5)
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Internet connectivity issue detected: {e}")
        logger.error("Please check your network connection and try again.")
        return False
    
    # Log NLP status
    logger.info(f"✓ NLP Status: spaCy={'✅' if spacy_available else '❌'}, NLTK={'✅' if nltk_available else '❌'}")
    
    # Initialize the Slack app with retry logic
    retry_count = 0
    max_retries = 3	
    
    while retry_count < max_retries:
        try:
            handler = SocketModeHandler(app, SLACK_APP_TOKEN)
            handler.start()
            logger.info("✅ Bot started successfully")
            return True
        except Exception as e:
            retry_count += 1
            logger.error(f"❌ Error starting bot (attempt {retry_count}/{max_retries}): {e}")
            if retry_count < max_retries:
                logger.info(f"Retrying in 5 seconds...")
                time.sleep(5)
    
    logger.error("❌ Failed to start bot after multiple attempts. Please check your network and credentials.")
    return False

if __name__ == "__main__":
    # Ensure database is initialized
    if not initialize_database():
        logger.error("❌ Failed to initialize database. Exiting.")
        exit(1)
    
    # Start the bot with enhanced error handling
    if not initialize_bot():
        logger.error("❌ Failed to start bot. Exiting.")
        exit(1)