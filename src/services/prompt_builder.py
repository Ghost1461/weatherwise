class PromptBuilderService:

    def _fmt(self, val):
        if val is True:
            return "yes"
        if val is False:
            return "no"
        return "unknown"

    def _effective(self, val, personalization_on: bool):
        if not personalization_on:
            return None
        return val

    def _effective_tone(self, tone: str, personalization_on: bool) -> str:
        if not personalization_on:
            return "friendly"
        if tone in ("friendly", "formal"):
            return tone
        return "friendly"

    def build_prompt(self, weather: dict, analysis: dict, user_request: dict) -> str:
        personalization_on = user_request["personalization_on"]

        has_dog = self._effective(user_request["has_dog"], personalization_on)
        has_car = self._effective(user_request["has_car"], personalization_on)
        near_sea = self._effective(user_request["near_sea"], personalization_on)
        sun_allergy = self._effective(user_request["sun_allergy"], personalization_on)
        tone = self._effective_tone(user_request["tone"], personalization_on)

        return f"""
    You are a weather recommendation assistant.

    Main signals:
    - umbrella_needed: {analysis["umbrella"]["prediction"]}
    - umbrella_confidence: {analysis["umbrella"]["confidence"]["label"]}
    - clothing: {analysis["clothing"]["prediction"]}
    - clothing_confidence: {analysis["clothing"]["confidence"]["label"]}
    - outside_score: {analysis["suitability"]["prediction"]}
    - outside_confidence: {analysis["suitability"]["confidence"]["label"]}

    Extra context:
    - rain_present: {"yes" if (weather.get("precipitation_mm", 0) or 0) > 0 else "no"}
    - windy: {"yes" if (weather.get("wind_speed_kmh", 0) or 0) >= 25 else "no"}
    - cloudy: {"yes" if (weather.get("cloud_cover_pct", 0) or 0) >= 60 else "no"}

    User preferences:
    - personalization: {"on" if personalization_on else "off"}
    - has_dog: {self._fmt(has_dog)}
    - has_car: {self._fmt(has_car)}
    - near_sea: {self._fmt(near_sea)}
    - sun_allergy: {self._fmt(sun_allergy)}
    - tone: {tone}

    Rules:
    - Write exactly 2 complete sentences.
    - Start directly with the advice.
    - Focus only on clothing, umbrella, and whether being outside is a good idea.
    - No numbers.
    - No greeting.
    - No raw weather data.
    - If personalization is off, ignore personal preferences.
    - If tone is friendly, be warm but direct.
    - If tone is formal, be clear and direct.
    - Output only the final recommendation.
    """.strip()

    def build_daily_prompt(self, daily_weather_summary: dict, daily_analysis: dict, user_request: dict) -> str:
            personalization_on = user_request["personalization_on"]

            has_dog = self._effective(user_request["has_dog"], personalization_on)
            has_car = self._effective(user_request["has_car"], personalization_on)
            near_sea = self._effective(user_request["near_sea"], personalization_on)
            sun_allergy = self._effective(user_request["sun_allergy"], personalization_on)
            tone = self._effective_tone(user_request["tone"], personalization_on)

            return f"""
    You are a weather recommendation assistant.

    Main signals:
    - umbrella_needed_today: {daily_analysis["umbrella_needed"]}
    - clothing_today: {daily_analysis["dominant_clothing"]}
    - outside_today: {daily_analysis["average_suitability_score"]}
    - condition: {daily_weather_summary.get("general_condition")}

    User preferences:
    - personalization: {"on" if personalization_on else "off"}
    - has_dog: {self._fmt(has_dog)}
    - has_car: {self._fmt(has_car)}
    - near_sea: {self._fmt(near_sea)}
    - sun_allergy: {self._fmt(sun_allergy)}
    - tone: {tone}

     Rules:
    - Write exactly 2 complete sentences.
    - Start directly with the advice.
    - Focus only on clothing, umbrella, and whether being outside is a good idea.
    - No numbers.
    - No greeting.
    - No raw weather data.
    - If personalization is off, ignore personal preferences.
    - If tone is friendly, be warm but direct.
    - If tone is formal, be clear and direct.
    - Output only the final recommendation.
    """.strip()