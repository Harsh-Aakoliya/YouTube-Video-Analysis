import logging
import os
import re
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fer import FER
from googleapiclient.discovery import build
from nltk import download as nltk_download
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from pydantic import BaseModel, Field, HttpUrl
from PIL import Image
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YouTube Analysis API", version="0.1.0")

# CORS for local dev + Codespaces public URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Config & global resources ------------------------------------------------


def require_api_key() -> str:
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        raise RuntimeError("YOUTUBE_API_KEY is not set in the environment.")
    return key


@lru_cache(maxsize=1)
def youtube_client():
    return build("youtube", "v3", developerKey=require_api_key())


@lru_cache(maxsize=1)
def emotion_classifier():
    # Lazy-load the EmoRoBERTa pipeline once
    return pipeline("sentiment-analysis", model="arpanghoshal/EmoRoBERTa")


@lru_cache(maxsize=1)
def thumbnail_detector():
    return FER(mtcnn=True)


# NLTK assets
nltk_download("stopwords", quiet=True)
nltk_download("punkt", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

# Simple sample data to fit a quick regression model for view prediction
SAMPLE_VIDEO_ROWS = [
    {"likes": 1200, "dislikes": 40, "subscribers": 150000, "views": 28000},
    {"likes": 5400, "dislikes": 210, "subscribers": 820000, "views": 195000},
    {"likes": 900, "dislikes": 10, "subscribers": 120000, "views": 22000},
    {"likes": 2200, "dislikes": 75, "subscribers": 310000, "views": 64000},
    {"likes": 7500, "dislikes": 320, "subscribers": 1250000, "views": 260000},
    {"likes": 15000, "dislikes": 600, "subscribers": 2000000, "views": 410000},
    {"likes": 4300, "dislikes": 140, "subscribers": 520000, "views": 130000},
    {"likes": 1700, "dislikes": 60, "subscribers": 240000, "views": 52000},
    {"likes": 6200, "dislikes": 190, "subscribers": 980000, "views": 210000},
    {"likes": 2800, "dislikes": 90, "subscribers": 430000, "views": 86000},
]
SAMPLE_DF = pd.DataFrame(SAMPLE_VIDEO_ROWS)
VIEW_MODEL = None


def get_view_model():
    global VIEW_MODEL
    if VIEW_MODEL is None:
        from sklearn.linear_model import LinearRegression

        features = SAMPLE_DF[["likes", "dislikes", "subscribers"]]
        target = SAMPLE_DF["views"]
        model = LinearRegression()
        model.fit(features, target)
        VIEW_MODEL = model
    return VIEW_MODEL


def emoji_regex():
    return re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F700-\U0001F77F"
        u"\U0001F780-\U0001F7FF"
        u"\U0001F800-\U0001F8FF"
        u"\U0001F900-\U0001F9FF"
        u"\U0001FA00-\U0001FA6F"
        u"\U0001FA70-\U0001FAFF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )


def strip_html(text: str) -> str:
    return BeautifulSoup(text, "html.parser").text


def strip_emoji(text: str) -> str:
    return emoji_regex().sub("", text)


def translate_to_english(text: str) -> str:
    try:
        return GoogleTranslator(source="auto", target="english").translate(text)
    except Exception:
        # Best-effort translation; fall back to the original text
        return text


def preprocess_comment(text: str, translate: bool) -> str:
    cleaned = strip_html(text)
    cleaned = strip_emoji(cleaned)
    if translate:
        cleaned = translate_to_english(cleaned)
    return cleaned


def summarize_text(text: str) -> str:
    words = word_tokenize(text)
    freq: Dict[str, int] = {}
    for word in words:
        word_lower = word.lower()
        if word_lower in STOP_WORDS or not word_lower.isalpha():
            continue
        freq[word_lower] = freq.get(word_lower, 0) + 1

    sentences = sent_tokenize(text)
    scores: Dict[str, int] = {}
    for sentence in sentences:
        score = sum(freq.get(word.lower(), 0) for word in word_tokenize(sentence))
        if score:
            scores[sentence] = score

    if not scores:
        return ""

    avg_score = sum(scores.values()) / len(scores)
    summary = " ".join(
        sentence for sentence in sentences if scores.get(sentence, 0) > 1.2 * avg_score
    )
    return summary or sentences[0]


# --- Pydantic models ----------------------------------------------------------


class ChannelStatsRequest(BaseModel):
    channel_ids: List[str] = Field(..., description="One or more YouTube channel IDs")


class ChannelStatsResponse(BaseModel):
    channel_id: str
    title: str
    subscribers: int
    views: int
    total_videos: int
    description: Optional[str] = None


class CommentEmotionRequest(BaseModel):
    video_id: str
    max_pages: int = Field(1, ge=1, le=5, description="How many 100-comment pages to scan")
    translate: bool = Field(True, description="Translate comments to English before labeling")


class CommentEmotion(BaseModel):
    author: Optional[str]
    like_count: int
    published_at: str
    original_text: str
    cleaned_text: str
    emotion: str


class CommentEmotionResponse(BaseModel):
    video_id: str
    comments: List[CommentEmotion]
    totals: Dict[str, int]


class TranscriptSummaryResponse(BaseModel):
    video_id: str
    summary: str
    transcript_text: str


class ViewPredictionRequest(BaseModel):
    likes: int = Field(..., ge=0)
    dislikes: int = Field(..., ge=0)
    subscribers: int = Field(..., ge=0)


class ViewPredictionResponse(BaseModel):
    predicted_views: int


class ThumbnailEmotion(BaseModel):
    box: List[int]
    emotions: Dict[str, float]
    top_emotion: str


class ThumbnailEmotionResponse(BaseModel):
    video_id: str
    thumbnail_url: HttpUrl
    faces: List[ThumbnailEmotion]


class VideoCategoryResponse(BaseModel):
    video_id: str
    title: str
    category_id: str
    category_name: str


# --- Routes -------------------------------------------------------------------


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/channel/stats", response_model=List[ChannelStatsResponse])
def channel_stats(payload: ChannelStatsRequest):
    yt = youtube_client()
    try:
        request = yt.channels().list(
            part="snippet,statistics",
            id=",".join(payload.channel_ids),
        )
        response = request.execute()
    except Exception as exc:
        logger.exception("YouTube channel stats failed")
        raise HTTPException(status_code=502, detail=str(exc))

    items = response.get("items", [])
    results: List[ChannelStatsResponse] = []
    for item in items:
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        results.append(
            ChannelStatsResponse(
                channel_id=item.get("id", ""),
                title=snippet.get("title", ""),
                subscribers=int(stats.get("subscriberCount", 0)),
                views=int(stats.get("viewCount", 0)),
                total_videos=int(stats.get("videoCount", 0)),
                description=snippet.get("description"),
            )
        )
    return results


@app.post("/video/comments/emotions", response_model=CommentEmotionResponse)
def video_comment_emotions(payload: CommentEmotionRequest):
    yt = youtube_client()
    comments: List[CommentEmotion] = []
    next_page_token: Optional[str] = None
    classifier = emotion_classifier()

    for _ in range(payload.max_pages):
        try:
            request = yt.commentThreads().list(
                part="snippet",
                videoId=payload.video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText",
            )
            data = request.execute()
        except Exception as exc:
            logger.exception("Fetching comments failed")
            raise HTTPException(status_code=502, detail=str(exc))

        for item in data.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            original = snippet.get("textDisplay", "")
            cleaned = preprocess_comment(original, translate=payload.translate)
            emotion = classifier(cleaned)[0]["label"]
            comments.append(
                CommentEmotion(
                    author=snippet.get("authorDisplayName"),
                    like_count=int(snippet.get("likeCount", 0)),
                    published_at=snippet.get("publishedAt", ""),
                    original_text=original,
                    cleaned_text=cleaned,
                    emotion=emotion,
                )
            )

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    totals: Dict[str, int] = {}
    for comment in comments:
        totals[comment.emotion] = totals.get(comment.emotion, 0) + 1

    return CommentEmotionResponse(
        video_id=payload.video_id,
        comments=comments,
        totals=totals,
    )


@app.get("/video/{video_id}/summary", response_model=TranscriptSummaryResponse)
def transcript_summary(video_id: str):
    try:
        transcript_parts = YouTubeTranscriptApi.get_transcript(video_id)
    except TranscriptsDisabled:
        raise HTTPException(status_code=404, detail="Transcript not available for this video")
    except Exception as exc:
        logger.exception("Transcript fetch failed")
        raise HTTPException(status_code=502, detail=str(exc))

    transcript_text = " ".join(part["text"] for part in transcript_parts)
    summary = summarize_text(transcript_text)
    return TranscriptSummaryResponse(video_id=video_id, summary=summary, transcript_text=transcript_text)


@app.get("/video/{video_id}/category", response_model=VideoCategoryResponse)
def video_category(video_id: str):
    yt = youtube_client()
    try:
        vid = yt.videos().list(part="snippet", id=video_id).execute()
    except Exception as exc:
        logger.exception("Video lookup failed")
        raise HTTPException(status_code=502, detail=str(exc))

    items = vid.get("items", [])
    if not items:
        raise HTTPException(status_code=404, detail="Video not found")
    snippet = items[0]["snippet"]
    category_id = snippet.get("categoryId")

    try:
        categories = yt.videoCategories().list(part="snippet", regionCode="US").execute()
    except Exception as exc:
        logger.exception("Category lookup failed")
        raise HTTPException(status_code=502, detail=str(exc))

    category_lookup = {
        item["id"]: item["snippet"]["title"] for item in categories.get("items", [])
    }
    return VideoCategoryResponse(
        video_id=video_id,
        title=snippet.get("title", ""),
        category_id=category_id or "",
        category_name=category_lookup.get(category_id, "Unknown"),
    )


@app.post("/predict/views", response_model=ViewPredictionResponse)
def predict_views(payload: ViewPredictionRequest):
    model = get_view_model()
    features = np.array([[payload.likes, payload.dislikes, payload.subscribers]])
    prediction = int(max(0, model.predict(features)[0]))
    return ViewPredictionResponse(predicted_views=prediction)


@app.get("/video/{video_id}/thumbnail/emotions", response_model=ThumbnailEmotionResponse)
def thumbnail_emotions(video_id: str):
    yt = youtube_client()
    try:
        vid = yt.videos().list(part="snippet", id=video_id).execute()
    except Exception as exc:
        logger.exception("Video lookup failed")
        raise HTTPException(status_code=502, detail=str(exc))

    items = vid.get("items", [])
    if not items:
        raise HTTPException(status_code=404, detail="Video not found")

    thumbnails = items[0]["snippet"].get("thumbnails", {})
    preferred_order = ["maxres", "standard", "high", "medium", "default"]
    chosen_url = None
    for key in preferred_order:
        if key in thumbnails:
            chosen_url = thumbnails[key]["url"]
            break
    if not chosen_url:
        raise HTTPException(status_code=404, detail="No thumbnail available")

    resp = requests.get(chosen_url, timeout=15)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Unable to download thumbnail")

    img = np.array(Image.open(BytesIO(resp.content)).convert("RGB"))
    detector = thumbnail_detector()
    faces_raw = detector.detect_emotions(img) or []

    faces = [
        ThumbnailEmotion(
            box=face["box"],
            emotions={k: float(v) for k, v in face.get("emotions", {}).items()},
            top_emotion=max(face.get("emotions", {}), key=face.get("emotions", {}).get, default=""),
        )
        for face in faces_raw
    ]
    return ThumbnailEmotionResponse(video_id=video_id, thumbnail_url=chosen_url, faces=faces)


@app.get("/")
def docs_redirect():
    return {"message": "YouTube Analysis API is running. See /docs for OpenAPI UI."}

