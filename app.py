import uuid
import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

from src.agent.agent import build_the_batch_agent


def run_fetch_and_ingest():
    """
    Fetch latest articles, ingest into Chroma, and clean up processed data.
    """
    try:
        from src.the_batch.ingestor import TheBatchIngestor
        from src.scripts.ingest_chroma_db import load_articles_from_jsonl
        from src.config import config
        from src.embeddings.text_embedder import TextEmbedder
        from src.embeddings.image_embedder import ImageEmbedder
        from src.vector_db.chroma_store import TheBatchChromaIndexer
    except Exception as e:
        st.error(f"Failed to import ingestion dependencies: {e}")
        return

    output_path = Path("data/processed/the_batch_articles.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    topics = ["letters", "data-points", "research", "business", "science", "culture", "hardware"]

    try:
        with st.spinner("Fetching latest articles..."):
            ingestor = TheBatchIngestor()
            ingestor.ingest_all_topics(topics=topics, output_jsonl=output_path)
    except Exception as e:
        st.error(f"Fetch failed: {e}")
        return

    try:
        with st.spinner("Indexing articles into Chroma..."):
            articles = load_articles_from_jsonl(output_path)
            if not articles:
                st.warning("No articles loaded; skipping indexing.")
                return

            text_embedder = TextEmbedder(
                model_name=config.text_embed_model_name,
                device=config.device,
            )
            clip_embedder = ImageEmbedder(
                model_name=config.clip_model_name,
                device=config.device,
            )
            indexer = TheBatchChromaIndexer(
                text_embedder=text_embedder,
                clip_embedder=clip_embedder,
            )
            indexer.index(articles)
    except Exception as e:
        st.error(f"Ingestion failed: {e}")
        return
    finally:
        try:
            if output_path.exists():
                output_path.unlink()
            if output_path.parent.exists() and not any(output_path.parent.iterdir()):
                output_path.parent.rmdir()
        except Exception as cleanup_err:
            st.warning(f"Cleanup skipped: {cleanup_err}")

    st.success("Fetched, indexed, and cleaned up processed data.")


def set_section(name: str):
    st.session_state.section = name


def extract_articles_from_intermediate_steps(intermediate_steps):
    """
    Pulls article results from the the_batch_multimodal_search tool calls.
    """
    collected = []

    for step in intermediate_steps:
        if not isinstance(step, (list, tuple)) or len(step) != 2:
            continue

        action, result = step

        tool_name = getattr(action, "tool", None)
        if tool_name not in {"the_batch_multimodal_search", "image_search"}:
            continue

        payload = result
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = None

        if isinstance(payload, dict):
            article_list = payload.get("results") or []
        elif isinstance(payload, list):
            article_list = payload
        else:
            article_list = []
        
        collected.extend(article_list)

    return collected


def render_article_card(art):
    """
    Stylish card for a single retrieved article / image match.
    Shows preview image + all metadata + relevant snippets.
    """
    title = art.get("title", "Untitled article")
    url = art.get("url", "#")
    topic = art.get("topic")
    published_at = art.get("published_at")
    sources = art.get("sources", [])
    text_snippets = art.get("text_snippets", [])
    image_urls = art.get("image_urls", []) or []
    image_alts = art.get("image_alts") or [None] * len(image_urls)
    score = art.get("score")

    meta_bits = []
    if topic:
        meta_bits.append(f"**Topic:** {topic}")
    if published_at:
        meta_bits.append(f"**Published:** {published_at}")
    if sources:
        src_labels = " ".join(f"`{s}`" for s in sorted(sources))
        meta_bits.append(f"**Source type:** {src_labels}")
    if score is not None:
        meta_bits.append(f"**Score:** {score:.3f}")

    is_image_match = "image" in [s.lower() for s in sources]
    is_text_match = "text" in [s.lower() for s in sources]

    if is_image_match and not is_text_match:
        match_badge = "🎨 Image-based match"
    elif is_text_match and not is_image_match:
        match_badge = "📄 Text-based match"
    elif is_image_match and is_text_match:
        match_badge = "🔀 Text + image match"
    else:
        match_badge = "🔎 Retrieved result"

    with st.container(border=True):
        st.markdown(
            f"<div style='margin-bottom: 0.5rem; font-size: 0.85rem; opacity: 0.8;'>{match_badge}</div>",
            unsafe_allow_html=True,
        )

        cols = st.columns([1, 2]) if image_urls else [st]
 
        if image_urls:
            with cols[0]:
                st.image(
                    image_urls[0],
                    caption=image_alts[0] or "",
                    width="stretch"
                )

        right_col = cols[-1]
        with right_col:
            if url and url != "#":
                st.markdown(f"### [{title}]({url})")
            else:
                st.markdown(f"### {title}")

            if meta_bits:
                st.markdown(" • ".join(meta_bits))

            if text_snippets and is_text_match:
                st.markdown("**Relevant snippets:**")
                for snippet in text_snippets[:3]:
                    st.markdown(f"> {snippet}")

            if is_image_match:
                if len(image_urls) > 1:
                    st.markdown("**Matched images from this source:**")
                    img_cols = st.columns(min(3, len(image_urls[1:])))
                    for idx, (u, alt) in enumerate(
                        zip(image_urls[1:], image_alts[1:]), start=1
                    ):
                        with img_cols[(idx - 1) % len(img_cols)]:
                            st.image(u, caption=alt or "", width="stretch")

        
def render_articles_section(articles, section_title: str = "Retrieved sources"):
    """
    Render a list of article/image results with a nice heading.
    """
    if not articles:
        return

    st.markdown("---")
    st.markdown(f"### {section_title}")

    for art in articles:
        render_article_card(art)


def init_agent():
    """
    Lazily initialize and cache the agent in Streamlit session_state.
    """
    if "agent" not in st.session_state:
        st.session_state.agent = build_the_batch_agent()
    return st.session_state.agent


def init_session():
    """
    Initialize session_id and chat messages for this Streamlit session.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"streamlit:{uuid.uuid4()}"

    if "messages" not in st.session_state:
        st.session_state.messages = []

def render_message(msg):
    """
    Render a single message in Streamlit's chat UI.
    """
    role = msg["role"]
    content = msg["content"]
    articles = msg.get("articles") or []

    with st.chat_message(role):
        st.markdown(content)

        if role == "assistant" and articles:
            render_articles_section(articles, section_title="Sources used in this answer")


def render_settings_section():
    """
    Settings tab: fetch and ingest latest articles with double confirmation.
    """
    st.subheader("Settings")
    st.caption("Fetch latest articles, index them into database")

    if "confirm_fetch" not in st.session_state:
        st.session_state.confirm_fetch = False

    if not st.session_state.confirm_fetch:
        if st.button("Fetch articles", type="primary"):
            st.session_state.confirm_fetch = True
            st.rerun()
    else:
        st.warning(
            "This will download the latest articles, re-index the database"
        )
        col_proceed, col_cancel = st.columns([1, 1])
        with col_proceed:
            proceed = st.button("Yes, fetch & ingest", type="primary", use_container_width=True)
        with col_cancel:
            cancel = st.button("Cancel", use_container_width=True)
        
        if proceed:
            run_fetch_and_ingest()
            st.session_state.confirm_fetch = False
        if cancel:
            st.session_state.confirm_fetch = False
            st.rerun()


def main():
    st.set_page_config(page_title="Multimodal RAG System", page_icon="", layout="wide")
    st.title("Multimodal RAG System")
    st.write("Agent capable of retrieving The Batch AI News and Insights from DeepLearning.AI")
    init_session()
    agent = init_agent()

    if "section" not in st.session_state:
        st.session_state.section = "Chat"

    st.sidebar.markdown("### Sections")
    st.sidebar.button(
        "Chat",
        use_container_width=True,
        type="primary" if st.session_state.section == "Chat" else "secondary",
        on_click=set_section,
        args=("Chat",),
    )
    st.sidebar.button(
        "Settings",
        use_container_width=True,
        type="primary" if st.session_state.section == "Settings" else "secondary",
        on_click=set_section,
        args=("Settings",),
    )

    section = st.session_state.section

    if section == "Chat":
        for msg in st.session_state.messages:
            render_message(msg)
            
        with st.sidebar:
            st.markdown("**Attachment**")
            uploaded_img = st.file_uploader(
                "📎 Image",
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed",
            )
        user_input = st.chat_input("Ask about AI news…")
        
        image_path = None

        if user_input:
            st.session_state.messages.append(
                {"role": "user", "content": user_input, "articles": []}
            )
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    combined_input = user_input

                    if uploaded_img:
                        uploads_dir = Path("data/uploads")
                        uploads_dir.mkdir(parents=True, exist_ok=True)
                        image_path = uploads_dir / uploaded_img.name
                        with open(image_path, "wb") as f:
                            f.write(uploaded_img.getbuffer())
                        st.image(str(image_path), caption="Query image", use_column_width=True)

                        combined_input = f"{user_input}\n[Image path]: {image_path}"

                    res = agent.invoke(
                        {"input": combined_input},
                        config={"configurable": {"session_id": st.session_state.session_id}},
                    )

            output_text = res.get("output", "(no output)")
            intermediate_steps = res.get("intermediate_steps", [])
            articles = extract_articles_from_intermediate_steps(intermediate_steps)

            st.session_state.messages.append(
                {"role": "assistant", "content": output_text, "articles": articles}
            )
            st.rerun()
    elif section == "Settings":
        render_settings_section()


if __name__ == "__main__":
    main()
