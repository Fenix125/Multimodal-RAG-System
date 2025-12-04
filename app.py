import uuid
import json

import streamlit as st

from src.agent.agent import build_the_batch_agent


def extract_articles_from_intermediate_steps(intermediate_steps):
    """
    Pulls article results from the the_batch_multimodal_search tool calls.
    """
    for step in intermediate_steps:
        if not isinstance(step, (list, tuple)) or len(step) != 2:
            continue

        action, result = step

        tool_name = getattr(action, "tool", None)
        if tool_name != "the_batch_multimodal_search":
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
        
        return article_list

    return []


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


def main():
    st.set_page_config(page_title="Multimodal RAG System", page_icon="", layout="wide")
    st.title("Multimodal RAG System")
    st.write("Agent capable of retrieving The Batch AI News and Insights from DeepLearning.AI")
    init_session()
    agent = init_agent()

    for msg in st.session_state.messages:
        render_message(msg)

    user_input = st.chat_input("Ask about AI news, robotics, agents, etc.")
    if user_input:
        st.session_state.messages.append(
            {"role": "user", "content": user_input, "articles": []}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = agent.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": st.session_state.session_id}},
                )

        output_text = res.get("output", "(no output)")
        intermediate_steps = res.get("intermediate_steps", [])
        articles = extract_articles_from_intermediate_steps(intermediate_steps)

        st.session_state.messages.append(
            {"role": "assistant", "content": output_text, "articles": articles}
        )
        st.rerun()


if __name__ == "__main__":
    main()
