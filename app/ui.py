import streamlit as st
import plotly.express as px


def styled_chart(fig: px.line, caption: str = ""):
    """Improve chart appearance for dark theme and show with a bordered container."""
    try:
        fig.update_traces(line=dict(width=2))  # Thicker lines
        fig.update_layout(
            title_font=dict(size=22, color="white"),
            font=dict(size=14, color="white"),
            margin=dict(l=60, r=60, t=70, b=60),
            plot_bgcolor="black",
            paper_bgcolor="black",
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.15)",
                gridwidth=0.2,
                zeroline=True, zerolinecolor="white", zerolinewidth=2,
                linecolor="white", linewidth=2, ticks="outside", tickcolor="white"
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.15)",
                gridwidth=0.2,
                zeroline=True, zerolinecolor="white", zerolinewidth=2,
                linecolor="white", linewidth=2, ticks="outside", tickcolor="white"
            ),
            legend=dict(title="", orientation="h", y=-0.25, font=dict(color="white"))
        )

        st.markdown(
            """
            <div style=""
                border:0px solid #444;
                border-radius:12px;
                padding:20px;
                margin:20px 0;
                background-color:#111;">
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True)
        if caption:
            st.caption(caption)
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception:
        # gracefully fallback to default plotting
        st.plotly_chart(fig, use_container_width=True)


hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;} /* Hide the hamburger menu */
    footer {visibility: hidden;}   /* Hide Streamlit footer */
    header {visibility: hidden;}   /* Hide Streamlit header */
    </style>
"""
