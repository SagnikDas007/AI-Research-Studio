
import os
import json
import re
import html
from io import BytesIO

import streamlit as st

from dotenv import load_dotenv
from groq import Groq
from ddgs import DDGS

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Research Studio",
    page_icon="🔬",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #0f172a,
            #111827,
            #172554
        );
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        color: white;
        margin-bottom: 8px;
    }

    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .badge {
        width: fit-content;
        margin: auto;
        padding: 8px 18px;
        border-radius: 20px;
        background: #1e3a8a;
        color: white;
        font-size: 14px;
        margin-bottom: 30px;
    }

    .workflow {
        text-align: center;
        color: #cbd5e1;
        font-size: 17px;
        margin: 25px 0;
    }

    .stButton > button {
        border-radius: 10px;
        height: 48px;
        font-size: 17px;
        font-weight: 600;
    }

    section[data-testid="stSidebar"] {
        background: #020617;
    }

    h1, h2, h3, h4 {
        color: white !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔬 AI Research Studio</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Research, Analyze, Verify and Generate Reports using AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="badge">🤖 Multi-Agent AI System</div>',
    unsafe_allow_html=True
)


# ============================================================
# AI FUNCTION
# ============================================================

def ask_ai(prompt, system_message):

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=8000
    )

    return response.choices[0].message.content


# ============================================================
# WEB SEARCH
# ============================================================

def search_web(topic, max_results=6):

    results = []

    try:

        with DDGS() as ddgs:

            search_results = ddgs.text(
                topic,
                max_results=max_results
            )

            for result in search_results:

                results.append({
                    "title": result.get(
                        "title",
                        "Unknown Source"
                    ),
                    "url": result.get(
                        "href",
                        ""
                    ),
                    "content": result.get(
                        "body",
                        ""
                    )
                })

    except Exception as e:

        results.append({
            "title": "Search Error",
            "url": "",
            "content": str(e)
        })

    return results


# ============================================================
# RESEARCH AGENT
# ============================================================

def research_agent(topic):

    sources = search_web(topic)

    source_text = ""

    for i, source in enumerate(sources, 1):

        source_text += f"""
SOURCE {i}
Title: {source["title"]}
URL: {source["url"]}
Content: {source["content"]}

"""

    prompt = f"""
Research the following topic:

{topic}

Use the following web search results:

{source_text}

Create detailed research notes.

Focus on:

1. Definition
2. Background
3. Important developments
4. Key facts
5. Statistics and measurable findings
6. Benefits
7. Challenges
8. Real-world applications
9. Different viewpoints
10. Important sources

Do not invent statistics.

Clearly distinguish between verified facts,
reported claims, and general observations.
"""

    research = ask_ai(
        prompt,
        """
You are a professional research assistant.

Your job is to collect accurate,
structured and balanced research information.

Never invent facts, statistics or sources.
"""
    )

    return research, sources


# ============================================================
# ANALYSIS AGENT
# ============================================================

def analysis_agent(topic, research):

    prompt = f"""
Topic:

{topic}

Research notes:

{research}

Analyze the research.

Discuss:

1. Major positive impacts
2. Major negative impacts
3. Important trends
4. Practical implications
5. Risks
6. Opportunities
7. Areas of disagreement
8. Context-dependent factors
9. Overall interpretation of the evidence

Do not invent information that is not supported
by the research.
"""

    return ask_ai(
        prompt,
        """
You are an analytical research expert.

Provide balanced, evidence-based analysis.
Do not exaggerate conclusions.
""",
    )


# ============================================================
# FACT CHECKER AGENT
# ============================================================

def fact_checker_agent(
    topic,
    research,
    analysis,
    sources
):

    source_text = ""

    for i, source in enumerate(sources, 1):

        source_text += f"""
SOURCE {i}
Title: {source["title"]}
URL: {source["url"]}
Content: {source["content"]}

"""

    prompt = f"""
Topic:

{topic}

RESEARCH:

{research}

ANALYSIS:

{analysis}

SOURCES:

{source_text}

Fact-check the important claims.

For each important claim, provide:

Claim:
Status:
Evidence:
Source:

Use only these statuses:

- Supported
- Partially Supported
- Not Supported
- Cannot Be Determined

Do not invent evidence.
"""

    return ask_ai(
        prompt,
        """
You are a strict fact-checking assistant.

Only mark claims as supported when
the available evidence actually supports them.
"""
    )


# ============================================================
# REPORT WRITER AGENT
# ============================================================

def report_writer_agent(
    topic,
    research,
    analysis,
    fact_check,
    sources
):

    source_text = ""

    for i, source in enumerate(sources, 1):

        source_text += f"""
{i}. {source["title"]}
URL: {source["url"]}
"""

    prompt = f"""
Create a complete professional research report
on the following topic:

{topic}

RESEARCH:
{research}

ANALYSIS:
{analysis}

FACT CHECK:
{fact_check}

SOURCES:
{source_text}

Write the COMPLETE report.

IMPORTANT:

- Do not stop halfway.
- Do not truncate the conclusion.
- Cover every section completely.
- Use clear paragraphs.
- Use bullet points where useful.
- Use Markdown headings.
- Use Markdown tables only when useful.
- Keep tables compact.
- Do not create extremely wide tables.
- Do not invent statistics.
- Do not invent sources.
- Preserve important numbers accurately.

Use EXACTLY this structure:

# {topic}

## 1. Introduction

Explain the topic and why it is important.

## 2. Background

Explain the background and context.

## 3. Key Findings

Present the most important findings.
Use a compact Markdown table if appropriate.

## 4. Analysis

Analyze the findings and explain what they mean.

## 5. Benefits

Explain the major benefits.

## 6. Challenges

Explain limitations, risks and challenges.

## 7. Fact-Checked Information

Summarize important claims and their verification status.

## 8. Conclusion

Provide a complete balanced conclusion.

## 9. Sources

List the important sources with their URLs.

The report must be complete from beginning to end.
"""

    return ask_ai(
        prompt,
        """
You are a professional technical report writer.

Write complete, clear, structured and
evidence-based reports.

Never intentionally stop early.
Never invent facts or sources.
"""
    )


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u25a0": "-",
        "\u2022": "-"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("**", "")
    text = text.replace("*", "")

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🔬 AI Research Studio")

    st.write(
        "Multi-agent research workflow"
    )

    st.divider()

    st.write("### Agents")

    st.write("🔎 Research Agent")
    st.write("📊 Analysis Agent")
    st.write("✅ Fact Checker")
    st.write("📝 Report Writer")

    st.divider()

    st.write("### Technologies")

    st.write("🐍 Python")
    st.write("🎈 Streamlit")
    st.write("🧠 Groq")
    st.write("🌐 DuckDuckGo Search")
    st.write("📄 ReportLab")


# ============================================================
# INPUT
# ============================================================

st.subheader("Enter Research Topic")

topic = st.text_input(
    "Research Topic",
    placeholder=(
        "Example: Impact of Generative AI on "
        "Software Development"
    ),
    label_visibility="collapsed"
)

research_button = st.button(
    "🚀 Start Research",
    type="primary",
    use_container_width=True
)


# ============================================================
# WORKFLOW
# ============================================================

st.markdown(
    """
    <div class="workflow">
    🔎 Research → 📊 Analyze → ✅ Verify → 📝 Report
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# RUN RESEARCH
# ============================================================

if research_button:

    if not topic.strip():

        st.warning(
            "Please enter a research topic."
        )

        st.stop()

    # ----------------------------------------
    # RESEARCH
    # ----------------------------------------

    with st.status(
        "🔎 Research Agent is working...",
        expanded=True
    ) as status:

        research, sources = research_agent(
            topic
        )

        st.write(
            f"Collected {len(sources)} sources."
        )

        status.update(
            label="🔎 Research completed",
            state="complete"
        )

    # ----------------------------------------
    # ANALYSIS
    # ----------------------------------------

    with st.status(
        "📊 Analysis Agent is working...",
        expanded=True
    ) as status:

        analysis = analysis_agent(
            topic,
            research
        )

        status.update(
            label="📊 Analysis completed",
            state="complete"
        )

    # ----------------------------------------
    # FACT CHECK
    # ----------------------------------------

    with st.status(
        "✅ Fact Checker is working...",
        expanded=True
    ) as status:

        fact_check = fact_checker_agent(
            topic,
            research,
            analysis,
            sources
        )

        status.update(
            label="✅ Fact checking completed",
            state="complete"
        )

    # ----------------------------------------
    # REPORT
    # ----------------------------------------

    with st.status(
        "📝 Report Writer is working...",
        expanded=True
    ) as status:

        report = report_writer_agent(
            topic,
            research,
            analysis,
            fact_check,
            sources
        )

        status.update(
            label="📝 Report generated",
            state="complete"
        )

    # ----------------------------------------
    # SAVE SESSION
    # ----------------------------------------

    st.session_state["topic"] = topic
    st.session_state["research"] = research
    st.session_state["analysis"] = analysis
    st.session_state["fact_check"] = fact_check
    st.session_state["sources"] = sources
    st.session_state["report"] = report


# ============================================================
# DISPLAY REPORT
# ============================================================

if "report" in st.session_state:

    st.divider()

    st.header("📄 Final Research Report")

    st.markdown(
        st.session_state["report"]
    )

    st.divider()

    # ========================================================
    # PDF GENERATION
    # ========================================================

    st.subheader("📥 Download Report")

    report_text = st.session_state["report"]


    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=18
    )

    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        spaceBefore=10,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        spaceAfter=7
    )

    table_style = ParagraphStyle(
        "TableText",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        spaceAfter=0
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=table_style,
        fontName="Helvetica-Bold"
    )

    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    story = []

    lines = report_text.splitlines()

    i = 0

    while i < len(lines):

        line = lines[i].strip()

        # ----------------------------------------
        # Empty line
        # ----------------------------------------

        if not line:

            story.append(
                Spacer(1, 5)
            )

            i += 1
            continue

        # ----------------------------------------
        # Main title
        # ----------------------------------------

        if line.startswith("# ") and not line.startswith("## "):

            title = clean_text(
                line[2:].strip()
            )

            story.append(
                Paragraph(
                    html.escape(title),
                    title_style
                )
            )

            i += 1
            continue

        # ----------------------------------------
        # Heading
        # ----------------------------------------

        if line.startswith("## "):

            heading = clean_text(
                line[3:].strip()
            )

            story.append(
                Paragraph(
                    html.escape(heading),
                    heading_style
                )
            )

            i += 1
            continue

        # ----------------------------------------
        # Table
        # ----------------------------------------

        if line.startswith("|"):

            table_data = []

            while (
                i < len(lines)
                and lines[i].strip().startswith("|")
            ):

                row = lines[i].strip()

                # Ignore Markdown separator
                if not re.match(
                    r"^\|[\s\-:|]+\|?$",
                    row
                ):

                    cells = [
                        clean_text(cell.strip())
                        for cell in row.strip("|").split("|")
                    ]

                    formatted_cells = []

                    for cell_index, cell in enumerate(cells):

                        escaped_cell = html.escape(
                            cell
                        )

                        if len(table_data) == 0:

                            formatted_cells.append(
                                Paragraph(
                                    escaped_cell,
                                    table_header_style
                                )
                            )

                        else:

                            formatted_cells.append(
                                Paragraph(
                                    escaped_cell,
                                    table_style
                                )
                            )

                    table_data.append(
                        formatted_cells
                    )

                i += 1

            if table_data:

                num_columns = len(
                    table_data[0]
                )

                available_width = 515

                column_widths = [
                    available_width / num_columns
                ] * num_columns

                pdf_table = Table(
                    table_data,
                    colWidths=column_widths,
                    repeatRows=1,
                    hAlign="LEFT"
                )

                pdf_table.setStyle(
                    TableStyle([
                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, 0),
                            colors.lightgrey
                        ),
                        (
                            "TEXTCOLOR",
                            (0, 0),
                            (-1, 0),
                            colors.black
                        ),
                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            0.5,
                            colors.grey
                        ),
                        (
                            "VALIGN",
                            (0, 0),
                            (-1, -1),
                            "TOP"
                        ),
                        (
                            "LEFTPADDING",
                            (0, 0),
                            (-1, -1),
                            5
                        ),
                        (
                            "RIGHTPADDING",
                            (0, 0),
                            (-1, -1),
                            5
                        ),
                        (
                            "TOPPADDING",
                            (0, 0),
                            (-1, -1),
                            5
                        ),
                        (
                            "BOTTOMPADDING",
                            (0, 0),
                            (-1, -1),
                            5
                        )
                    ])
                )

                story.append(
                    pdf_table
                )

                story.append(
                    Spacer(1, 12)
                )

            continue

        # ----------------------------------------
        # Bullet points
        # ----------------------------------------

        if (
            line.startswith("- ")
            or line.startswith("* ")
            or line.startswith("•")
        ):

            bullet = clean_text(
                line[2:].strip()
                if not line.startswith("• ")
                else line[2:].strip()
            )
            # Remove Markdown formatting symbols
            bullet = bullet.replace("**", "")
            bullet = bullet.replace("*", "")

            story.append(
                Paragraph(
                    "• " + html.escape(bullet),
                    body_style
                )
            )

            i += 1
            continue

        # ----------------------------------------
        # Numbered list
        # ----------------------------------------

        if re.match(
            r"^\d+\.\s+",
            line
        ):

            number_match = re.match(
                r"^(\d+)\.\s+(.*)",
                line
            )

            if number_match:

                number = number_match.group(1)
                content = clean_text(
                    number_match.group(2)
                )

                story.append(
                    Paragraph(
                        f"{number}. "
                        f"{html.escape(content)}",
                        body_style
                    )
                )

                i += 1
                continue

        # ----------------------------------------
        # Normal paragraph
        # ----------------------------------------

        paragraph = clean_text(
            line
        )

        # Basic Markdown bold cleanup
        paragraph = paragraph.replace("**", "")
        paragraph = paragraph.replace("*", "")

        # Escape ampersands that are not HTML tags
        paragraph = paragraph.replace(
            "&",
            "&amp;"
        )

        paragraph = paragraph.replace(
            "&amp;lt;",
            "&lt;"
        )

        paragraph = paragraph.replace(
            "&amp;gt;",
            "&gt;"
        )

        paragraph = paragraph.replace(
            "&amp;b>",
            "<b>"
        )

        paragraph = paragraph.replace(
            "&amp;/b>",
            "</b>"
        )

        paragraph = paragraph.replace(
            "&amp;i>",
            "<i>"
        )

        paragraph = paragraph.replace(
            "&amp;/i>",
            "</i>"
        )

        story.append(
            Paragraph(
                paragraph,
                body_style
            )
        )

        i += 1

    # ========================================================
    # BUILD PDF
    # ========================================================

    doc.build(story)
    pdf_buffer.seek(0)

    st.download_button(
        label="📄 Download Research Report PDF",
        data=pdf_buffer,
        file_name="research_report.pdf",
        mime="application/pdf",            
        use_container_width=True
        )


# ============================================================
# SOURCES
# ============================================================

if "sources" in st.session_state:

    st.divider()

    st.subheader("🌐 Research Sources")

    for i, source in enumerate(
        st.session_state["sources"],
        1
    ):

        title = source.get(
            "title",
            "Unknown Source"
        )

        url = source.get(
            "url",
            ""
        )

        if url:

            st.markdown(
                f"{i}. [{title}]({url})"
            )

        else:

            st.write(
                f"{i}. {title}"
            )


# ============================================================
# AGENT OUTPUTS
# ============================================================

if "research" in st.session_state:

    st.divider()

    st.subheader("🤖 Agent Outputs")

    with st.expander(
        "🔎 Research Agent"
    ):

        st.markdown(
            st.session_state["research"]
        )

    with st.expander(
        "📊 Analysis Agent"
    ):

        st.markdown(
            st.session_state["analysis"]
        )

    with st.expander(
        "✅ Fact Checker"
    ):

        st.markdown(
            st.session_state["fact_check"]
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Multi-Agent AI Research System | "
    "Python + Streamlit + Groq"
)
