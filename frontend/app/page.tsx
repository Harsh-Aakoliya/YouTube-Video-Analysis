"use client";

import { FormEvent, useMemo, useState } from "react";

type ChannelStats = {
  channel_id: string;
  title: string;
  subscribers: number;
  views: number;
  total_videos: number;
  description?: string;
};

type CommentEmotion = {
  author?: string;
  like_count: number;
  published_at: string;
  original_text: string;
  cleaned_text: string;
  emotion: string;
};

type CommentEmotionResponse = {
  video_id: string;
  comments: CommentEmotion[];
  totals: Record<string, number>;
};

type TranscriptSummaryResponse = {
  video_id: string;
  summary: string;
  transcript_text: string;
};

type VideoCategoryResponse = {
  video_id: string;
  title: string;
  category_id: string;
  category_name: string;
};

type ThumbnailEmotionResponse = {
  video_id: string;
  thumbnail_url: string;
  faces: {
    box: number[];
    emotions: Record<string, number>;
    top_emotion: string;
  }[];
};

const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${apiBase}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || res.statusText);
  }
  return res.json();
}

export default function HomePage() {
  const [channelInput, setChannelInput] = useState("UCnz-ZXXER4jOvuED5trXfEA");
  const [channelStats, setChannelStats] = useState<ChannelStats[] | null>(null);
  const [channelLoading, setChannelLoading] = useState(false);

  const [commentVideoId, setCommentVideoId] = useState("8t0G-9Kls6o");
  const [commentPages, setCommentPages] = useState(1);
  const [translate, setTranslate] = useState(true);
  const [commentsResult, setCommentsResult] = useState<CommentEmotionResponse | null>(null);
  const [commentsLoading, setCommentsLoading] = useState(false);

  const [summaryVideoId, setSummaryVideoId] = useState("cQW2wMf8dEE");
  const [summaryResult, setSummaryResult] = useState<TranscriptSummaryResponse | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(false);

  const [categoryVideoId, setCategoryVideoId] = useState("Nz3Ngt0AMDA");
  const [categoryResult, setCategoryResult] = useState<VideoCategoryResponse | null>(null);
  const [categoryLoading, setCategoryLoading] = useState(false);

  const [likes, setLikes] = useState(1500);
  const [dislikes, setDislikes] = useState(30);
  const [subs, setSubs] = useState(120000);
  const [predictedViews, setPredictedViews] = useState<number | null>(null);
  const [viewsLoading, setViewsLoading] = useState(false);

  const [thumbVideoId, setThumbVideoId] = useState("Nz3Ngt0AMDA");
  const [thumbResult, setThumbResult] = useState<ThumbnailEmotionResponse | null>(null);
  const [thumbLoading, setThumbLoading] = useState(false);

  const channelIds = useMemo(
    () =>
      channelInput
        .split(",")
        .map((c) => c.trim())
        .filter(Boolean),
    [channelInput]
  );

  const handleChannelStats = async (e: FormEvent) => {
    e.preventDefault();
    setChannelLoading(true);
    try {
      const data = await apiFetch<ChannelStats[]>("/channel/stats", {
        method: "POST",
        body: JSON.stringify({ channel_ids: channelIds }),
      });
      setChannelStats(data);
    } catch (err: any) {
      alert(`Channel stats failed: ${err.message}`);
    } finally {
      setChannelLoading(false);
    }
  };

  const handleComments = async (e: FormEvent) => {
    e.preventDefault();
    setCommentsLoading(true);
    try {
      const data = await apiFetch<CommentEmotionResponse>("/video/comments/emotions", {
        method: "POST",
        body: JSON.stringify({
          video_id: commentVideoId.trim(),
          max_pages: commentPages,
          translate,
        }),
      });
      setCommentsResult(data);
    } catch (err: any) {
      alert(`Comment analysis failed: ${err.message}`);
    } finally {
      setCommentsLoading(false);
    }
  };

  const handleSummary = async (e: FormEvent) => {
    e.preventDefault();
    setSummaryLoading(true);
    try {
      const data = await apiFetch<TranscriptSummaryResponse>(`/video/${summaryVideoId}/summary`);
      setSummaryResult(data);
    } catch (err: any) {
      alert(`Summary failed: ${err.message}`);
    } finally {
      setSummaryLoading(false);
    }
  };

  const handleCategory = async (e: FormEvent) => {
    e.preventDefault();
    setCategoryLoading(true);
    try {
      const data = await apiFetch<VideoCategoryResponse>(`/video/${categoryVideoId}/category`);
      setCategoryResult(data);
    } catch (err: any) {
      alert(`Category lookup failed: ${err.message}`);
    } finally {
      setCategoryLoading(false);
    }
  };

  const handleViewPrediction = async (e: FormEvent) => {
    e.preventDefault();
    setViewsLoading(true);
    try {
      const data = await apiFetch<{ predicted_views: number }>("/predict/views", {
        method: "POST",
        body: JSON.stringify({
          likes: Number(likes),
          dislikes: Number(dislikes),
          subscribers: Number(subs),
        }),
      });
      setPredictedViews(data.predicted_views);
    } catch (err: any) {
      alert(`Prediction failed: ${err.message}`);
    } finally {
      setViewsLoading(false);
    }
  };

  const handleThumbnail = async (e: FormEvent) => {
    e.preventDefault();
    setThumbLoading(true);
    try {
      const data = await apiFetch<ThumbnailEmotionResponse>(
        `/video/${thumbVideoId}/thumbnail/emotions`
      );
      setThumbResult(data);
    } catch (err: any) {
      alert(`Thumbnail analysis failed: ${err.message}`);
    } finally {
      setThumbLoading(false);
    }
  };

  return (
    <main className="grid" style={{ gap: 16, maxWidth: 1200, margin: "0 auto" }}>
      <header className="panel">
        <h1>YouTube Analysis Playground</h1>
        <p>
          Point the frontend at your FastAPI instance via{" "}
          <code>NEXT_PUBLIC_API_BASE_URL</code>. Try the built-in demos below.
        </p>
        <div className="badge">Backend: FastAPI</div>{" "}
        <div className="badge">Frontend: Next.js</div>
      </header>

      <section className="panel">
        <h2>Channel stats</h2>
        <form onSubmit={handleChannelStats}>
          <label>Channel IDs (comma separated)</label>
          <input
            value={channelInput}
            onChange={(e) => setChannelInput(e.target.value)}
            placeholder="UCxxxxxxxxx, UCyyyyyyyy"
          />
          <button type="submit" disabled={channelIds.length === 0 || channelLoading}>
            {channelLoading ? "Loading..." : "Fetch stats"}
          </button>
        </form>
        {channelStats && (
          <div className="result">
            {channelStats.map((c) => (
              <div key={c.channel_id} style={{ marginBottom: 12 }}>
                <strong>{c.title}</strong> ({c.channel_id})<br />
                Subs: {c.subscribers.toLocaleString()} · Views: {c.views.toLocaleString()} ·
                Videos: {c.total_videos.toLocaleString()}
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="panel">
        <h2>Comment emotions</h2>
        <form onSubmit={handleComments} className="two-col">
          <div>
            <label>Video ID</label>
            <input value={commentVideoId} onChange={(e) => setCommentVideoId(e.target.value)} />
          </div>
          <div>
            <label>Pages (x100 comments)</label>
            <input
              type="number"
              min={1}
              max={5}
              value={commentPages}
              onChange={(e) => setCommentPages(Number(e.target.value))}
            />
          </div>
          <div>
            <label>
              <input
                type="checkbox"
                checked={translate}
                onChange={(e) => setTranslate(e.target.checked)}
                style={{ width: "auto", marginRight: 8 }}
              />
              Translate to English before labeling
            </label>
          </div>
          <div style={{ alignSelf: "end" }}>
            <button type="submit" disabled={commentsLoading}>
              {commentsLoading ? "Analyzing..." : "Analyze comments"}
            </button>
          </div>
        </form>
        {commentsResult && (
          <div className="result">
            <div style={{ marginBottom: 8 }}>
              Emotion totals:{" "}
              {Object.entries(commentsResult.totals).map(([k, v]) => (
                <span key={k} className="badge">
                  {k}: {v}
                </span>
              ))}
            </div>
            <div style={{ maxHeight: 240, overflowY: "auto" }}>
              {commentsResult.comments.slice(0, 20).map((c, idx) => (
                <div key={`${c.published_at}-${idx}`} style={{ marginBottom: 8 }}>
                  <strong>{c.author ?? "Anon"}</strong> · {c.emotion} · ❤️ {c.like_count}
                  <br />
                  <em>{c.cleaned_text}</em>
                </div>
              ))}
              {commentsResult.comments.length > 20 && (
                <div className="badge">Showing first 20 of {commentsResult.comments.length}</div>
              )}
            </div>
          </div>
        )}
      </section>

      <section className="panel">
        <h2>Transcript summary</h2>
        <form onSubmit={handleSummary} className="two-col">
          <div>
            <label>Video ID</label>
            <input value={summaryVideoId} onChange={(e) => setSummaryVideoId(e.target.value)} />
          </div>
          <div style={{ alignSelf: "end" }}>
            <button type="submit" disabled={summaryLoading}>
              {summaryLoading ? "Summarizing..." : "Summarize transcript"}
            </button>
          </div>
        </form>
        {summaryResult && (
          <div className="result">
            <strong>Summary</strong>
            <p>{summaryResult.summary}</p>
          </div>
        )}
      </section>

      <section className="panel two-col">
        <div>
          <h2>Video category lookup</h2>
          <form onSubmit={handleCategory}>
            <label>Video ID</label>
            <input value={categoryVideoId} onChange={(e) => setCategoryVideoId(e.target.value)} />
            <button type="submit" disabled={categoryLoading}>
              {categoryLoading ? "Loading..." : "Get category"}
            </button>
          </form>
          {categoryResult && (
            <div className="result">
              <div>
                <strong>{categoryResult.title}</strong>
              </div>
              <div>
                {categoryResult.category_name} ({categoryResult.category_id})
              </div>
            </div>
          )}
        </div>

        <div>
          <h2>Views prediction</h2>
          <form onSubmit={handleViewPrediction}>
            <label>Likes</label>
            <input type="number" value={likes} onChange={(e) => setLikes(Number(e.target.value))} />
            <label>Dislikes</label>
            <input
              type="number"
              value={dislikes}
              onChange={(e) => setDislikes(Number(e.target.value))}
            />
            <label>Subscribers</label>
            <input type="number" value={subs} onChange={(e) => setSubs(Number(e.target.value))} />
            <button type="submit" disabled={viewsLoading}>
              {viewsLoading ? "Predicting..." : "Predict views"}
            </button>
          </form>
          {predictedViews !== null && (
            <div className="result">
              Estimated views: <strong>{predictedViews.toLocaleString()}</strong>
            </div>
          )}
        </div>
      </section>

      <section className="panel">
        <h2>Thumbnail emotion detection</h2>
        <form onSubmit={handleThumbnail} className="two-col">
          <div>
            <label>Video ID</label>
            <input value={thumbVideoId} onChange={(e) => setThumbVideoId(e.target.value)} />
          </div>
          <div style={{ alignSelf: "end" }}>
            <button type="submit" disabled={thumbLoading}>
              {thumbLoading ? "Scanning..." : "Scan thumbnail"}
            </button>
          </div>
        </form>
        {thumbResult && (
          <div className="result">
            <div style={{ marginBottom: 8 }}>
              Thumbnail URL:{" "}
              <a href={thumbResult.thumbnail_url} target="_blank" rel="noreferrer">
                {thumbResult.thumbnail_url}
              </a>
            </div>
            {thumbResult.faces.length === 0 && <div>No faces detected.</div>}
            {thumbResult.faces.map((face, idx) => (
              <div key={idx} style={{ marginBottom: 8 }}>
                <strong>Face {idx + 1}</strong> · {face.top_emotion}
                <div className="badge">Box: {face.box.join(", ")}</div>
                <div>
                  {Object.entries(face.emotions).map(([k, v]) => (
                    <span key={k} className="badge">
                      {k}: {(Number(v) * 100).toFixed(1)}%
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}

