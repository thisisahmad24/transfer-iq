"""Streamlit dashboard for TransferIQ."""

import streamlit as st

from src.models.predict import (
    MODEL_PATH,
    build_prediction_features,
    load_model,
    predict_transfer_fee,
)


st.set_page_config(
    page_title="TransferIQ | Transfer Fee Predictor",
    page_icon="⚽",
    layout="wide",
)


@st.cache_resource
def get_model():
    """Load the saved model once per app session."""
    return load_model()


def main():
    st.title("⚽ TransferIQ")
    st.subheader("Premier League Transfer Fee Predictor")

    st.write(
        "Estimate a footballer's potential transfer fee using "
        "age, playing time, position, and previous-season performance."
    )

    st.caption(
        "Research and demonstration tool. Predictions are estimates, "
        "not guaranteed transfer fees or official valuations."
    )

    if not MODEL_PATH.exists():
        st.error(
            "The trained model was not found. "
            "Run the model training pipeline first."
        )
        st.stop()

    try:
        model = get_model()
    except Exception as exc:
        st.error(f"Could not load the trained model: {exc}")
        st.stop()

    st.divider()
    st.header("Player information")

    with st.form("player_prediction_form"):
        player_name = st.text_input(
            "Player name (optional)",
            placeholder="Enter a player name",
        )

        position_options = [
            "Goalkeeper",
            "Defence",
            "Midfield",
            "Attack",
        ]

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Age (years)",
                min_value=15,
                max_value=45,
                value=24,
            )
            position = st.selectbox(
                "Position group",
                position_options,
                index=2,
            )
            minutes = st.number_input(
                "Minutes played",
                min_value=0,
                max_value=6000,
                value=1800,
            )
            goals_scored = st.number_input(
                "Goals",
                min_value=0,
                value=10,
            )
            assists = st.number_input(
                "Assists",
                min_value=0,
                value=5,
            )
            total_points = st.number_input(
                "Total FPL points",
                min_value=0,
                value=150,
            )

        with col2:
            element_type = st.selectbox(
                "FPL position type",
                options=["1", "2", "3", "4"],
                index=2,
                format_func=lambda value: {
                    "1": "Goalkeeper",
                    "2": "Defender",
                    "3": "Midfielder",
                    "4": "Forward",
                }[value],
            )
            goals_conceded = st.number_input(
                "Goals conceded",
                min_value=0,
                value=20,
            )
            clean_sheets = st.number_input(
                "Clean sheets",
                min_value=0,
                value=8,
            )
            bonus = st.number_input(
                "Bonus points",
                min_value=0,
                value=12,
            )
            bps = st.number_input(
                "Bonus points system (BPS)",
                min_value=0,
                value=400,
            )
            now_cost = st.number_input(
                "FPL cost (tenths of £m)",
                min_value=0.0,
                value=75.0,
                step=0.5,
            )

        with col3:
            influence = st.number_input(
                "Influence",
                min_value=0.0,
                value=500.0,
            )
            creativity = st.number_input(
                "Creativity",
                min_value=0.0,
                value=450.0,
            )
            threat = st.number_input(
                "Threat",
                min_value=0.0,
                value=600.0,
            )
            ict_index = st.number_input(
                "ICT index",
                min_value=0.0,
                value=155.0,
            )
            red_cards = st.number_input(
                "Red cards",
                min_value=0,
                value=0,
            )
            yellow_cards = st.number_input(
                "Yellow cards",
                min_value=0,
                value=3,
            )
            selected_by_percent = st.number_input(
                "Selected by (%)",
                min_value=0.0,
                max_value=100.0,
                value=12.5,
                step=0.5,
            )

        submitted = st.form_submit_button(
            "Estimate transfer fee",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        stats = {
            "age": age,
            "minutes": minutes,
            "goals_scored": goals_scored,
            "assists": assists,
            "total_points": total_points,
            "goals_conceded": goals_conceded,
            "clean_sheets": clean_sheets,
            "bonus": bonus,
            "bps": bps,
            "influence": influence,
            "creativity": creativity,
            "threat": threat,
            "ict_index": ict_index,
            "red_cards": red_cards,
            "yellow_cards": yellow_cards,
            "selected_by_percent": selected_by_percent,
            "now_cost": now_cost,
            "position": position,
            "element_type": element_type,
        }

        try:
            prediction = predict_transfer_fee(
                stats,
                model=model,
            )

            st.divider()
            st.header("Prediction result")

            if player_name.strip():
                st.write(f"Player: **{player_name.strip()}**")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    "Estimated transfer fee",
                    f"€{prediction / 1_000_000:.2f} million",
                )

            with result_col2:
                st.metric(
                    "Estimated fee in euros",
                    f"€{prediction:,.0f}",
                )

            st.info(
                "This is a model estimate based on the statistics "
                "entered. It does not account for every factor, such "
                "as contract length, injuries, negotiations, or "
                "competition between buying clubs."
            )

        except (ValueError, FileNotFoundError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(
                "The prediction could not be completed. "
                f"Technical details: {exc}"
            )


if __name__ == "__main__":
    main()