import sqlite3
import json
from config import AI_MODEL, CLIENT
from vector_store import VectorStore
from guardrails import is_medical_query, SAFETY_MESSAGE


SYSTEM_PROMPT_DIET = """You are a knowledgeable diet planning assistant. You help plan meals consistent with the user's goal and any potential injuries.
Rules:
- Do not provide medical advice (diagnoses, medications). You can offer nutritional advice for injuries (protein, omega-3, vitamins).
- Use knowledge of products from the OpenFoodFacts database and general principles of nutrition.
- If the user provided age, weight, height, gender, and activity level – calculate their caloric requirement and macronutrients.
- When suggesting meals, provide specific products (name, brand, portion size) and their nutritional values.
- Be concise, but provide justification.
"""

class DietAssistant:
    def __init__(self, vector_store: VectorStore, db_path="database/food.db"):
        self.vs = vector_store
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.user_profile = {}  # tymczasowe dane: wiek, waga, wzrost, płeć, aktywność, cel, kontuzja

    def _get_full_context(self, history: list):
        return history[-10:]

    def _calculate_bmr(self, weight_kg, height_cm, age, sex):
        if sex == "male":
            return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
        else:
            return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

    def _calculate_tdee(self, bmr, activity_factor):
        return bmr * activity_factor

    def _activity_factor(self, level):
        mapping = {
            "sedentary": 1.2,
            "light": 1.375,
            "moderate": 1.55,
            "active": 1.725,
            "very_active": 1.9
        }
        return mapping.get(level, 1.2)

    def _extract_profile_from_message(self, user_msg: str, history: list) -> dict:
        """Use LLM to extract profile fields from the user message.
        Returns a dict with any of: weight_kg, height_cm, age, sex, activity_level, goal, injury.
        """
        context = self._get_full_context(history)
        # Build a prompt that lists already known fields (to avoid asking again)
        known = {k: v for k, v in self.user_profile.items() if v is not None}
        system = f"""You are a profile extraction assistant. The user is talking about diet and training.
    Extract the following fields from the conversation if present, otherwise leave null.
    Fields: weight_kg (float, kg), height_cm (float, cm), age (int, years), sex ('male' or 'female'),
    activity_level (one of: sedentary, light, moderate, active, very_active),
    goal (one of: loss, maintenance, gain),
    injury (string, short description of any injury that affects diet, e.g. 'knee injury', 'broken bone').

    Already known fields: {known}
    Output ONLY valid JSON, e.g. {{"weight_kg": 75.5, "height_cm": null, ...}}.
    """
        resp = CLIENT.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=150,
            temperature=0.0
        )
        try:
            extracted = json.loads(resp.choices[0].message.content)
            # Keep only valid keys
            valid_keys = {"weight_kg", "height_cm", "age", "sex", "activity_level", "goal", "injury"}
            return {k: v for k, v in extracted.items() if k in valid_keys and v is not None}
        except:
            return {}

    def search_products(self, query=None, min_proteins=None, max_fat=None, max_carbs=None,
                        min_fiber=None, nutriscore_max=None, category_contains=None, limit=10):
        import re
        params = []
        conditions = []

        # Clean the query to extract only alphanumeric words, avoiding FTS5 syntax crashes
        clean_query = " ".join(re.findall(r'\w+', query)) if query else ""

        if clean_query:
            # Use FTS5 with the actual table name (no alias for MATCH)
            sql = """
                  SELECT p.code, \
                         p.product_name, \
                         p.brands, \
                         p.energy_kcal,
                         p.proteins, \
                         p.fat, \
                         p.carbs, \
                         p.fiber, \
                         p.nutriscore
                  FROM products_fts
                           JOIN products p ON products_fts.code = p.code \
                  """
            conditions.append("products_fts MATCH ?")
            params.append(clean_query)
            # All columns in conditions must use 'p.' prefix because of join
            col_prefix = "p."
        else:
            # No text search (or query was only punctuation): query the products table directly
            sql = """
                  SELECT code, \
                         product_name, \
                         brands, \
                         energy_kcal,
                         proteins, \
                         fat, \
                         carbs, \
                         fiber, \
                         nutriscore
                  FROM products \
                  """
            col_prefix = ""

        # Numeric filters – add with correct prefix
        if min_proteins is not None:
            conditions.append(f"{col_prefix}proteins >= ?")
            params.append(min_proteins)
        if max_fat is not None:
            conditions.append(f"{col_prefix}fat <= ?")
            params.append(max_fat)
        if max_carbs is not None:
            conditions.append(f"{col_prefix}carbs <= ?")
            params.append(max_carbs)
        if min_fiber is not None:
            conditions.append(f"{col_prefix}fiber >= ?")
            params.append(min_fiber)
        if nutriscore_max is not None:
            conditions.append(f"{col_prefix}nutriscore <= ?")
            params.append(nutriscore_max)
        if category_contains:
            conditions.append(f"{col_prefix}categories LIKE ?")
            params.append(f"%{category_contains}%")

        where = " AND ".join(conditions) if conditions else "1=1"
        order_col = f"{col_prefix}proteins" if col_prefix else "proteins"
        sql += f" WHERE {where} ORDER BY {order_col} DESC LIMIT ?"
        params.append(limit)

        cursor = self.conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_product_by_code(self, code):
        cursor = self.conn.execute("SELECT * FROM products WHERE code = ?", (code,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def generate_meal_plan(self, calories, proteins_g, fats_g, carbs_g, injury=None):
        # Get high‑protein foods (lean)
        protein_foods = self.search_products(min_proteins=15, max_fat=15, limit=20)
        # Get carb sources (whole grains, vegetables)
        carb_foods = self.search_products(category_contains="cereal,vegetable", max_fat=10, limit=20)
        # Get healthy fat sources (nuts, seeds, oils)
        fat_foods = self.search_products(category_contains="nuts,seeds,oil", min_fiber=2, limit=10)

        # Format for LLM
        def fmt(products):
            return "\n".join(
                f"- {p['product_name']} ({p['proteins']:.1f}g protein, {p['fat']:.1f}g fat, "
                f"{p['carbs']:.1f}g carbs, {p['energy_kcal']:.0f} kcal per 100g)"
                for p in products[:8]
            )

        # Retrieve injury‑specific diet knowledge
        injury_context = ""
        if injury:
            knowledge = self.vs.search_diet_knowledge(f"diet for {injury} injury", n_results=2)
            if knowledge["documents"]:
                injury_context = "\n".join(knowledge["documents"][0])

        prompt = f"""
    You are a dietitian. Create a one‑day meal plan (breakfast, lunch, dinner, one snack) that meets:
    - Total energy: {calories:.0f} kcal
    - Protein: {proteins_g:.0f} g
    - Fat: {fats_g:.0f} g
    - Carbohydrates: {carbs_g:.0f} g

    Injury / condition: {injury if injury else 'None'}
    Nutritional advice: {injury_context}

    Available foods (per 100g):
    {fmt(protein_foods)}
    --- carb sources ---
    {fmt(carb_foods)}
    --- fat sources ---
    {fmt(fat_foods)}

    For each meal:
    - Suggest specific foods from the lists above (or similar realistic foods).
    - Indicate the serving size in grams.
    - Calculate approximate macros and calories for the meal.
    - Ensure the daily totals are close to the targets.

    Output in plain English, friendly and concise.
    """
        resp = CLIENT.chat.completions.create(
            model=AI_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT_DIET},
                      {"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.6
        )
        return resp.choices[0].message.content.strip()

    def generate_response(self, user_msg: str, history: list) -> str:
        if is_medical_query(user_msg):
            return SAFETY_MESSAGE

        # 2. Update profile from this message
        new_profile = self._extract_profile_from_message(user_msg, history)
        self.user_profile.update(new_profile)

        # 3. Check if we have enough to answer
        required = {"weight_kg", "height_cm", "age", "sex", "activity_level", "goal"}
        missing = [k for k in required if k not in self.user_profile]
        if missing:
            # Ask for missing fields one by one
            ask = f"To give you a personalised diet plan, I need your {', '.join(missing)}. Could you provide it?"
            return ask

        # 3. Sprawdź, czy użytkownik prosi o konkretny plan
        if "plan" in user_msg.lower() or "meal" in user_msg.lower() or "diet" in user_msg.lower():
            # Oblicz zapotrzebowanie
            bmr = self._calculate_bmr(
                self.user_profile["weight_kg"],
                self.user_profile["height_cm"],
                self.user_profile["age"],
                self.user_profile["sex"]
            )
            tdee = self._calculate_tdee(bmr, self._activity_factor(self.user_profile["activity_level"]))
            goal = self.user_profile.get("goal", "maintenance")
            if goal == "loss":
                calories = tdee - 500
            elif goal == "gain":
                calories = tdee + 300
            else:
                calories = tdee

            # Ustal makro (przykład: 1.8g białka/kg, 1g tłuszczu/kg, reszta węgle)
            weight = self.user_profile["weight_kg"]
            proteins_g = 1.8 * weight
            fats_g = 1.0 * weight
            carbs_g = (calories - (proteins_g * 4 + fats_g * 9)) / 4
            carbs_g = max(100, carbs_g)  # minimum 100g

            injury = self.user_profile.get("injury", None)
            plan = self.generate_meal_plan(calories, proteins_g, fats_g, carbs_g, injury)

            answer = f"Your daily requirement is ~{calories:.0f} kcal.\n{plan}"
        else:
            # 4. Zwykła rozmowa – wyszukaj produkty lub odpowiedz ogólnie
            products = self.search_products(query=user_msg, limit=5)
            if products:
                prod_info = "\n".join([f"- {p['product_name']} (proteins {p['proteins']:.1f}g, fat {p['fat']:.1f}g, {p['energy_kcal']:.0f}kcal/100g)" for p in products])
                answer = f"I found products fitting your request:\n{prod_info}\nWould you like for me to prepare a meal plan?"
            else:
                # Użyj LLM z kontekstem
                messages = [{"role": "system", "content": SYSTEM_PROMPT_DIET}]
                messages.extend(history[-6:])
                messages.append({"role": "user", "content": user_msg})
                resp = CLIENT.chat.completions.create(
                    model=AI_MODEL,
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7
                )
                answer = resp.choices[0].message.content.strip()

        return answer

    def update_profile(self, **kwargs):
        self.user_profile.update(kwargs)