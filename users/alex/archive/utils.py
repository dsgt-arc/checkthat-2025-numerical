import re
import dateparser

def tokenizer(text, id=None):
    """
    This is a custom preprocessing (normalization) for a word-based tokenizer to handle:
    1) "covid-19" normalization.
    2) Dates in various formats converted to YYYY-MM-DD.
    3) Numbers with "crore" or commas converted to scientific notation.
    4) Ensures numbers are processed only once.
    """


    # ----------------------------------------------------------------
    # Supporting Functions
    # ----------------------------------------------------------------
    def convert_crore_to_numeric(matchobj):
        """
        Converts numbers followed by 'crore' to their numeric equivalent.
        Example: '35,000 crore' → '350000000000'
        """
        number_part = matchobj.group(1).replace(",", "")
        numeric_value = float(number_part) * 10**7  # 1 crore = 10 million
        return f"PROCESSED_{numeric_value}"  # Mark as processed to avoid reprocessing

    def number_repl(matchobj):
        """
        Converts a matched number to scientific notation.
        Skips numbers that are already processed by "crore".
        """
        raw_number = matchobj.group(0)
        if raw_number.startswith("PROCESSED_"):
            return raw_number[len("PROCESSED_"):]  # Return already-processed number
        # Remove commas and convert to scientific notation
        normalized_number = raw_number.replace(",", "")
        num = float(normalized_number)
        return f"{num:.9e}"  # Scientific notation

    def apply_scientific_notation(line):
        """
        Applies scientific notation conversion to all numbers in a line.
        Excludes numbers marked as already processed.
        """
        number_pattern = r"\b(PROCESSED_\d+(\.\d+)?|\d{1,3}(,\d{3})*(\.\d+)?)\b"
        return re.sub(number_pattern, number_repl, line)

    # ----------------------------------------------------------------
    # Main Tokenizer Logic
    # ----------------------------------------------------------------

    # 1) Normalize case
    text_lower = text.lower()

    # 2) Normalize "covid-19" → "covid19"
    text_lower = re.sub(r"covid\s?[-]?\s?19", "covid19", text_lower, flags=re.IGNORECASE)

    # 3) Handle "crore" conversions FIRST
    text_lower = re.sub(r"(\d[\d,]*)\s+crore", convert_crore_to_numeric, text_lower)

    # ----------------------------------------------------------------
    # Regex for Dates
    # ----------------------------------------------------------------
    pattern_iso = r"\b\d{4}-\d{1,2}-\d{1,2}\b"  # e.g., "2023-12-12"
    pattern_md_comma = r"\b\w{3,9}\s+\d{1,2},\s*\d{4}\b"  # e.g., "July 28, 2022"
    pattern_dmy = r"\b\d{1,2}\s+\w{3,9}\.?\s+\d{4}\b"  # e.g., "28 July 2022"
    pattern_dayname = (
        r"\b(?:mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?|"
        r"lun(?:di)?|mar(?:di)?|mer(?:credi)?|jeu(?:di)?|ven(?:dredi)?|sam(?:edi)?|dim(?:anche)?|"
        r"lun(?:edì)?|mar(?:tedì)?|mer(?:coledì)?|gio(?:vedì)?|ven(?:erdì)?|sab(?:ato)?|dom(?:enica)?|"
        r"mo(?:ntag)?|di(?:enstag)?|mi(?:ttwoch)?|do(?:nnerstag)?|fr(?:eitag)?|sa(?:mstag)?|so(?:nntag)?|"
        r"lun(?:es)?|mar(?:tes)?|mié(?:rcoles)?|jue(?:ves)?|vie(?:rnes)?|sáb(?:ado)?|dom(?:ingo)?"
        r")\s*,?\s*\d{1,2}\s+\w{3,9}\.?\s+\d{4}\b"
    )
    combined_pattern = f"{pattern_iso}|{pattern_md_comma}|{pattern_dmy}|{pattern_dayname}"
    date_candidate_pattern = re.compile(combined_pattern, flags=re.IGNORECASE)

    # ----------------------------------------------------------------
    # Date Parsing
    # ----------------------------------------------------------------
    placeholders = []
    replaced_text = text_lower

    # Replace date-like patterns with placeholders
    for i, candidate in enumerate(date_candidate_pattern.findall(text_lower)):
        parsed = dateparser.parse(candidate)
        if parsed:
            placeholder = f"DATEPH{i}"
            iso_date = parsed.strftime("%Y-%m-%d")
            placeholders.append((placeholder, iso_date))
            replaced_text = replaced_text.replace(candidate, placeholder, 1)

    # ----------------------------------------------------------------
    # Apply Scientific Notation
    # ----------------------------------------------------------------
    replaced_text = apply_scientific_notation(replaced_text)

    # ----------------------------------------------------------------
    # Replace Placeholders with Actual Dates Before Tokenization
    # ----------------------------------------------------------------
    for placeholder, iso_date in placeholders:
        replaced_text = replaced_text.replace(placeholder, iso_date)

    # ----------------------------------------------------------------
    # Tokenization
    # ----------------------------------------------------------------
    # Updated tokenization regex to include dates, scientific notation, and words
    token_pattern = r"[a-zA-Z]+|[\d]+\.[\d]+e[+-]?\d+|\d{4}-\d{2}-\d{2}|\w+"
    tokens = re.findall(token_pattern, replaced_text)
 
    return (id, tokens)
